import os

# Suppress TensorFlow informational logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
from PIL import Image, ImageOps
import streamlit as st
import tensorflow as tf

# Set page layout and configuration
st.set_page_config(
    page_title="Tomato Health Classifier", page_icon="🌱", layout="centered"
)

# Application title and overview
st.title("🌱 Tomato Health Classification App")
st.write(
    "This AI application uses a Convolutional Neural Network (CNN) to detect "
    "whether a tomato leaf is **Healthy** or affected by **Tomato Bacterial Spot**."
)


# Load trained model with caching to optimize performance
@st.cache_resource
def load_trained_model():
    try:
        # Ensure 'tomato_model.keras' is placed in the same directory as app.py
        model = tf.keras.models.load_model("tomato_model.keras")
        return model
    except Exception as e:
        st.sidebar.error(
            "Error loading model. Ensure the model file 'tomato_model.keras' "
            "exists in the project directory."
        )
        return None


model = load_trained_model()

if model is not None:
    st.sidebar.success("Model loaded successfully!")

# File uploader widget
uploaded_file = st.file_uploader(
    "Choose a tomato leaf image...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None and model is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Leaf Image", use_container_width=True)
    st.write("---")
    st.write("### Analysis Result")

    with st.spinner("Classifying image..."):
        # Preprocessing matching standard CNN input (224x224, normalized [0, 1])
        target_size = (224, 224)
        resized_image = ImageOps.fit(
            image, target_size, Image.Resampling.LANCZOS
        )
        img_array = np.asarray(resized_image, dtype=np.float32) / 255.0
        img_reshape = np.expand_dims(img_array, axis=0)

        # Make prediction
        prediction = model.predict(img_reshape)

        # DEBUG: Displays raw probability in the sidebar so you can verify values
        st.sidebar.write("🔍 Raw Model Output:", prediction)

        # FIXED CLASS MAPPING: Keras sorts folders alphabetically:
        # Index 0 = 'Tomato Bacterial Spot' (B comes first)
        # Index 1 = 'Healthy Tomato' (H comes second)
        CLASS_NAMES = ["Tomato Bacterial Spot", "Healthy Tomato"]

        if prediction.shape[-1] == 1:
            raw_score = float(prediction[0][0])

            # Standard binary sigmoid thresholding:
            # Score > 0.5 -> Class 1 ('Healthy Tomato')
            # Score <= 0.5 -> Class 0 ('Tomato Bacterial Spot')
            if raw_score > 0.5:
                predicted_class = CLASS_NAMES[1]  # Healthy Tomato
                score = raw_score
            else:
                predicted_class = CLASS_NAMES[0]  # Tomato Bacterial Spot
                score = 1.0 - raw_score
        else:
            # For 2-output categorical (softmax) models:
            class_idx = int(np.argmax(prediction[0]))
            predicted_class = CLASS_NAMES[class_idx]
            score = float(np.max(prediction[0]))

        # Confidence Threshold Check (Flags non-leaf or uncertain images)
        CONFIDENCE_THRESHOLD = 0.65

        if score < CONFIDENCE_THRESHOLD:
            st.error(
                "⚠️ **Uncertain Image:** The model is not confident this is a tomato leaf. "
                "Please upload a clear, well-lit image of a tomato leaf."
            )
            st.caption(f"Highest confidence was only {score * 100:.2f}%.")
        else:
            if predicted_class == "Healthy Tomato":
                st.success(f"**Prediction:** {predicted_class}")
            else:
                st.warning(f"**Prediction:** {predicted_class}")

            st.info(f"**Confidence Score:** {score * 100:.2f}%")
