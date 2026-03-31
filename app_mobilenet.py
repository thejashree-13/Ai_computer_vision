from flask import Flask, render_template, request, url_for, send_from_directory
from PIL import Image
import numpy as np
import os
import config
import cv2

# Ensure TensorFlow exists
try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

MODEL = None
FRAMEWORK = None
LOAD_ERROR = None


# ==========================================================
# Grad-CAM Function
# ==========================================================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="Conv_1"):

    # Backbone
    base_model = model.get_layer("mobilenetv2_1.00_224")
    last_conv_layer = base_model.get_layer(last_conv_layer_name)

    # Build a new functional model from backbone input to final output
    x = base_model.output
    x = model.layers[1](x)
    x = model.layers[2](x)
    x = model.layers[3](x)
    final_output = model.layers[4](x)

    grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[last_conv_layer.output, final_output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        #
        class_index = tf.argmax(predictions[0])
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()

def overlay_heatmap(img_path, heatmap, alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    colormap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    colormap = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)

    superimposed_img = cv2.addWeighted(img, 1 - alpha, colormap, alpha, 0)

    return superimposed_img


# ==========================================================
# Model Loader
# ==========================================================
def load_model():
    global MODEL, FRAMEWORK, LOAD_ERROR
    if MODEL is not None:
        return

    path = config.MODEL_PATH
    LOAD_ERROR = None

    if not HAS_TF:
        LOAD_ERROR = "TensorFlow not installed in this environment."
        return

    if not os.path.exists(path):
        LOAD_ERROR = f"Model file not found: {path}"
        return

    try:
        from tensorflow.keras.models import load_model
        import numpy as np

        MODEL = load_model(path)
        FRAMEWORK = 'keras'

        print("MODEL LOADED SUCCESSFULLY")
        #print("Top-level layers:")
        #print([layer.name for layer in MODEL.layers])

        #  Force-build once
        dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
        MODEL.predict(dummy)

        print("MODEL BUILT SUCCESSFULLY")

        # DEBUG BACKBONE INFO
        #backbone = MODEL.layers[0]
        #print("Backbone type:", type(backbone))
        #print("Backbone built:", backbone.built)
        #print("Backbone input:", backbone.input)
        #print("Backbone output:", backbone.output)

    except Exception as e:
        LOAD_ERROR = f"Model load failed: {e}"

# ==========================================================
# Image Preprocessing
# ==========================================================
def preprocess_image(img: Image.Image, size: int):
    img = img.convert('RGB').resize((size, size))
    arr = np.asarray(img).astype('float32') / 255.0
    return np.expand_dims(arr, 0)


# ==========================================================
# Flask Routes
# ==========================================================
@app.route('/', methods=['GET', 'POST'])
def index():
    global LOAD_ERROR
    load_model()

    result = None
    img_url = None
    gradcam_img_url = None
    load_error = LOAD_ERROR

    if request.method == 'POST':
        f = request.files.get('image')

        if f and f.filename:
            filename = f.filename
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            f.save(save_path)

            img_url = url_for('uploaded_file', filename=filename)

            try:
                img = Image.open(save_path)
                inp = preprocess_image(img, config.IMAGE_SIZE)

                # ================= Prediction =================
                preds = MODEL.predict(inp, verbose=0)
                prob = float(preds[0][0])
                pred_idx = 1 if prob >= 0.5 else 0
                confidence = prob if pred_idx == 1 else (1 - prob)

                # ================= Grad-CAM =================
                heatmap = make_gradcam_heatmap(inp, MODEL)
                gradcam_img = overlay_heatmap(save_path, heatmap)

                gradcam_filename = "gradcam_result.jpg"
                gradcam_path = os.path.join("static", gradcam_filename)

                cv2.imwrite(
                    gradcam_path,
                    cv2.cvtColor(gradcam_img, cv2.COLOR_RGB2BGR)
                )

                gradcam_img_url = url_for('static', filename=gradcam_filename)

                result = {
                    'label': config.LABELS[pred_idx],
                    'prob': f"{confidence * 100:.2f}%",
                    'binary': True,
                    'raw_prob': prob
                }

            except Exception as e:
                result = {'label': 'Error', 'prob': str(e)}

    return render_template(
        'index.html',
        title=config.TITLE,
        quote=config.QUOTE,
        result=result,
        img_url=img_url,
        gradcam_img_url=gradcam_img_url,
        load_error=load_error,
        LABELS=config.LABELS
    )


@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(port=5000, debug=True)