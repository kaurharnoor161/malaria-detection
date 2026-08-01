import os
import ssl
import certifi
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from PIL import Image
from sklearn.model_selection import train_test_split

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D
)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# ==========================================
# SSL FIX FOR MAC
# ==========================================
ssl._create_default_https_context = lambda: ssl.create_default_context(
    cafile=certifi.where()
)

# ==========================================
# SETTINGS
# ==========================================
IMG_SIZE = 96
BATCH_SIZE = 32
EPOCHS = 25

DATASET_PATH = "dataset"

# ==========================================
# LOAD DATASET
# ==========================================
images = []
labels = []

classes = ["Uninfected", "Parasitized"]

print("Loading dataset...")

for label, class_name in enumerate(classes):

    folder_path = os.path.join(DATASET_PATH, class_name)

    files = os.listdir(folder_path)

    print(f"\nLoading {class_name}: {len(files)} files")

    for file in files:

        # Ignore hidden/system files
        if file.startswith(".") or file == "Thumbs.db":
            continue

        if not file.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        img_path = os.path.join(folder_path, file)

        try:

            img = Image.open(img_path).convert("RGB")

            img = img.resize((IMG_SIZE, IMG_SIZE))

            img = np.array(img)

            images.append(img)

            labels.append(label)

        except Exception as e:
            print(f"Error loading {img_path}: {e}")

# ==========================================
# CONVERT TO NUMPY
# ==========================================
X = np.array(images, dtype="float32")

y = np.array(labels)

print(f"\nTotal Images: {len(X)}")
print(f"Dataset Shape: {X.shape}")

# ==========================================
# PREPROCESS
# ==========================================
X = preprocess_input(X)

# ==========================================
# TRAIN TEST SPLIT
# ==========================================
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.235,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.638,
    random_state=42,
    stratify=y_temp
)

print(f"\nTrain Samples: {len(X_train)}")
print(f"Validation Samples: {len(X_val)}")
print(f"Test Samples: {len(X_test)}")

# ==========================================
# BUILD MODEL
# ==========================================
def build_model():

    base = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )

    # Fine-tuning
    base.trainable = True

    for layer in base.layers[:-30]:
        layer.trainable = False

    model = Sequential([

        base,

        GlobalAveragePooling2D(),

        Dense(256, activation='relu'),

        Dropout(0.5),

        Dense(128, activation='relu'),

        Dropout(0.3),

        Dense(1, activation='sigmoid')

    ])

    model.compile(

        optimizer=Adam(learning_rate=0.0001),

        loss='binary_crossentropy',

        metrics=['accuracy']

    )

    return model

model = build_model()

model.summary()

# ==========================================
# EARLY STOPPING
# ==========================================
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# ==========================================
# TRAIN
# ==========================================
history = model.fit(

    X_train,
    y_train,

    validation_data=(X_val, y_val),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=[early_stop]

)

# ==========================================
# EVALUATE
# ==========================================
loss, accuracy = model.evaluate(X_test, y_test)

print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

# ==========================================
# SAVE MODEL
# ==========================================
os.makedirs("model", exist_ok=True)

model.save("model/malaria_mobilenetv2.keras")

print("\nModel saved successfully!")

# ==========================================
# PLOTS
# ==========================================
plt.figure(figsize=(12,5))

# Accuracy
plt.subplot(1,2,1)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title("Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.legend(["Train", "Validation"])

# Loss
plt.subplot(1,2,2)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title("Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend(["Train", "Validation"])

plt.tight_layout()

plt.show()