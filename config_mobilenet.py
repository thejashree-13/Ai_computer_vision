# Edit these to match your trained model and labels
# Point `MODEL_PATH` to the actual saved model file or the directory containing it.
# If you set it to a directory, the app will pick the newest model file with
# extensions like .h5, .pt, .pth, .ckpt, or .onnx inside that directory.
MODEL_PATH = r"C:\Users\Sree lekha M\Desktop\ML_project\ML Project\waste_model_final\waste_model_biodegradable.h5"
# IMAGE_SIZE used for resizing input images (change if your model expects a different size)
IMAGE_SIZE = 224
# Labels in the same order your model outputs (index 0 -> LABELS[0])
LABELS = ["Biodegradable", "Non-Biodegradable"]

TITLE = "Transparent Waste Sorting Vision System"
QUOTE = (
    "Efficient waste management is crucial for environmental sustainability and public health."
)
