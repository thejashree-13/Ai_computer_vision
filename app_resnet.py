from flask import Flask, render_template, request, url_for, send_from_directory
from PIL import Image
import numpy as np
import os
import config

import torch
import torch.nn as nn
from torchvision import models, transforms

app = Flask(__name__)

# Upload folder
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# =========================
# LOAD MODEL
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)

model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

print("Model loaded successfully!")

# =========================
# TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
])

# =========================
# PREDICT FUNCTION
# =========================
def predict_image(img):
    img = img.convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    label = config.LABELS[predicted.item()]
    confidence = confidence.item() * 100

    return label, f"{confidence:.2f}%"

# =========================
# ROUTES
# =========================
@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    img_url = None

    if request.method == 'POST':
        file = request.files.get('image')

        if file and file.filename:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            img_url = url_for('uploaded_file', filename=file.filename)

            img = Image.open(filepath)
            label, prob = predict_image(img)

            result = {
                'label': label,
                'prob': prob
            }

    return render_template(
        'index.html',
        title=config.TITLE,
        quote=config.QUOTE,
        result=result,
        img_url=img_url
    )

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# =========================
# RUN
# =========================
if __name__ == '__main__':
    app.run(debug=True)