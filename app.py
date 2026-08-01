import streamlit as st
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# =====================================
# SETTINGS
# =====================================
IMG_SIZE = 96

# =====================================
# LOAD MODEL
# =====================================
model = load_model("model/malaria_mobilenetv2.keras")

# =====================================
# PAGE TITLE
# =====================================
st.title("🦠 Malaria Detection System")

st.write(
    "Upload a blood cell image to detect malaria infection."
)

# =====================================
# FILE UPLOAD
# =====================================
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

# =====================================
# PREDICTION
# =====================================
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize((IMG_SIZE, IMG_SIZE))

    img = np.array(img)

    img = preprocess_input(img)

    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img)[0][0]

    # Result
    if prediction > 0.5:

        st.error("⚠️ Parasitized Cell Detected")

        confidence = prediction * 100

    else:

        st.success("✅ Uninfected Cell")

        confidence = (1 - prediction) * 100

    st.write(f"Confidence: {confidence:.2f}%")