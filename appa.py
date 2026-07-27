import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
from PIL import Image, ImageOps
import streamlit as st
import tensorflow as tf

# Set page layout and configuration
st.set_page_config(
    page_title="Tomato Health Classifier", page_icon="🍅", layout="centered"
)

# Application title and overview
st.title("🍅 Tomato Health Classification App")
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

# Class labels mapping
CLASS_NAMES = ["Healthy Tomato", "Tomato Bacterial Spot"]

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
        img_array = np.asarray(resized_image) / 255.0
        img_reshape = np.expand_dims(img_array, axis=0)

        # Make prediction
        prediction = model.predict(img_reshape)

        if prediction.shape[-1] == 1:
            confidence = float(prediction[0][0])
            # Measure distance from uncertainty (0.5)
            certainty = abs(confidence - 0.5) * 2  # Scales score from 0.0 to 1.0

            if certainty < 0.5:  # Less than 75% confident
                st.error(
                    "⚠️ **Uncertain Image:** This does not appear to be a valid tomato leaf. "
                    "Please upload a clear image of a tomato leaf."
                )
            else:
                if confidence > 0.5:
                    predicted_class = CLASS_NAMES[1]
                    score = confidence
                else:
                    predicted_class = CLASS_NAMES[0]
                    score = 1.0 - confidence

                if predicted_class == "Healthy Tomato":
                    st.success(f"**Prediction:** {predicted_class}")
                else:
                    st.warning(f"**Prediction:** {predicted_class}")

                st.info(f"**Confidence Score:** {score * 100:.2f}%")

        # Display prediction result
        if predicted_class == "Healthy Tomato":
            st.success(f"**Prediction:** {predicted_class}")
        else:
            st.warning(f"**Prediction:** {predicted_class}")

        st.info(f"**Confidence Score:** {score * 100:.2f}%")
