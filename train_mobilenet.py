import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils import class_weight
import numpy as np

# =========================
# Hyperparameters
# =========================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
DATASET_DIR = 'dataset/'  # Path to your dataset folder

# =========================
# Data Preprocessing + Augmentation
# =========================
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    shear_range=0.2
)

train_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True
)

val_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

print("Class Indices:", train_data.class_indices)

# =========================
# Load Pretrained MobileNetV2
# =========================
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False  # Freeze base for faster training

# =========================
# Build Model
# =========================
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# =========================
# Handle Class Imbalance
# =========================
class_weights_array = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_data.classes),
    y=train_data.classes
)
class_weights = dict(enumerate(class_weights_array))
print("Class Weights:", class_weights)

# =========================
# Train Model
# =========================
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    class_weight=class_weights
)

# =========================
# Save Model
# =========================
model.save("waste_model_biodegradable1.h5")
print("Model saved as waste_model_biodegradable1.h5")

# =========================
# Performance Metrics with Sample Counts
# =========================
val_data.reset()
predictions = model.predict(val_data)

# -------------------------
# Class-wise thresholds
# -------------------------
threshold_biodegradable = 0.45  # class 0
threshold_non_biodegradable = 0.55  # class 1

pred_labels = []
for p in predictions:
    if p > threshold_non_biodegradable:
        pred_labels.append(1)  # non-biodegradable
    elif p < threshold_biodegradable:
        pred_labels.append(0)  # biodegradable
    else:
        # borderline → mark as non-biodegradable
        pred_labels.append(1)

pred_labels = np.array(pred_labels)

# -------------------------
# Overall metrics
# -------------------------
overall_acc = accuracy_score(val_data.classes, pred_labels)
cm = confusion_matrix(val_data.classes, pred_labels)
cr = classification_report(val_data.classes, pred_labels)

print("\nOverall Accuracy:", overall_acc)
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", cr)

# -------------------------
# Sample counts and correct/wrong predictions per class
# -------------------------
total_samples = len(val_data.classes)
biodegradable_samples = np.sum(val_data.classes == 0)
non_biodegradable_samples = np.sum(val_data.classes == 1)

correct_biodegradable = np.sum((val_data.classes == 0) & (pred_labels == 0))
wrong_biodegradable = biodegradable_samples - correct_biodegradable

correct_non_biodegradable = np.sum((val_data.classes == 1) & (pred_labels == 1))
wrong_non_biodegradable = non_biodegradable_samples - correct_non_biodegradable

print("\nTotal Validation Samples:", total_samples)
print("Biodegradable Samples:", biodegradable_samples)
print("Non-Biodegradable Samples:", non_biodegradable_samples)
print("\nCorrectly Predicted Biodegradable:", correct_biodegradable)
print("Incorrectly Predicted Biodegradable:", wrong_biodegradable)
print("Correctly Predicted Non-Biodegradable:", correct_non_biodegradable)
print("Incorrectly Predicted Non-Biodegradable:", wrong_non_biodegradable)