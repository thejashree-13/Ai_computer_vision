# Transparent Waste Sorting Vision System — UI

This folder contains a small Flask-based UI that lets you upload an image and get a prediction from your trained model.

Quick start

1. Place your trained model file at the path configured in `config.py` (default: `C:\Users\Swetha\Desktop\ML Project\model.h5`).
   - If your model is a PyTorch file, update `MODEL_PATH` in `config.py` accordingly.
2. (Optional) Adjust `LABELS` and `IMAGE_SIZE` in `config.py` to match your model's output and input size.
3. Create a virtual environment and install dependencies:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1    # be sure you see the `(venv)` prompt
pip install -r requirements.txt
# If your model is TensorFlow/Keras:
pip install tensorflow
# Or for PyTorch models:
pip install torch
```

> **Important:** always activate the `venv` before running `python app.py`.  The error
> "TensorFlow not installed" means the server was started with a different
> Python interpreter that doesn't have the packages installed.

4. Run the app:

```powershell
python app.py
```

Open http://localhost:5000 in your browser, upload an image, and hit Submit.

