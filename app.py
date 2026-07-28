import os
import time

# Suppress TensorFlow C++ logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

t0 = time.time()
print("Starting app...")

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

print(f"TensorFlow imported in {time.time() - t0:.2f}s")

# Set page layout and configuration
st.set_page_config(
    page_title="Tomato Health Classifier",
    page_icon="🌱",
    layout="centered"
)

# Application title and overview
st.title("🌱 Tomato Health Classification App")
st.write(
    "This AI application uses a MobileNetV2 Convolutional Neural Network to detect "
    "whether a tomato leaf is **Healthy** or affected by **Tomato Bacterial Spot**."
)


# Load trained model with caching
@st.cache_resource
def load_trained_model(model_path: str = "tomato_model.keras"):
    if not os.path.exists(model_path):
        st.sidebar.error(f"Model file not found: '{model_path}'. Ensure it exists in the app folder.")
        return None
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except (ValueError, OSError) as e:
        st.sidebar.error(f"Error loading model: {e}")
        return None


model = load_trained_model()

if model is not None:
    st.sidebar.success("Model loaded successfully!")

# File uploader widget
uploaded_file = st.file_uploader(
    "Upload a tomato leaf image...",
    type=["jpg", "jpeg", "png"]
)

# CLASS MAPPING MATCHING KERAS' DEFAULT ALPHABETICAL ORDER:
# Index 0: Bacterial Spot
# Index 1: Healthy Tomato
CLASS_NAMES = ['Tomato Bacterial Spot', 'Healthy Tomato']

if uploaded_file is not None and model is not None:
    def is_likely_leaf(pil_img):
        """
        Checks if an uploaded image contains prominent leaf colors (green/brown/yellow)
        using PIL and Numpy directly without OpenCV.
        """
        # Convert PIL Image to HSV format using Pillow
        hsv_img = pil_img.convert('HSV')
        hsv_array = np.array(hsv_img)

        H = hsv_array[:, :, 0]  # Hue (0-255 in PIL)
        S = hsv_array[:, :, 1]  # Saturation
        V = hsv_array[:, :, 2]  # Value/Brightness

        # PIL Hue ranges (0-255): Green ~ [35, 120], Yellow/Brown ~ [15, 45]
        green_mask = (H >= 35) & (H <= 120) & (S >= 40) & (V >= 30)
        brown_yellow_mask = (H >= 15) & (H <= 45) & (S >= 40) & (V >= 30)

        leaf_mask = green_mask | brown_yellow_mask
        leaf_pixel_ratio = np.sum(leaf_mask) / (hsv_array.shape[0] * hsv_array.shape[1])

        # Returns True if at least 15% of pixels match leaf tones
        return leaf_pixel_ratio >= 0.15
    # 1. Display uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Leaf Image", width=350)
    st.write("---")
    st.write("### Analysis Result")

    with st.spinner("Classifying image..."):
        # 2. Resizing & Preprocessing matching MobileNetV2 training pipeline
        target_size = (224, 224)
        resized_image = ImageOps.fit(image, target_size, Image.Resampling.LANCZOS)

        img_array = np.asarray(resized_image, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)

        # Apply MobileNetV2 scaling [-1, 1]
        img_preprocessed = preprocess_input(img_array)

        # 3. Model Prediction
        raw_prediction = model.predict(img_preprocessed, verbose=0)

        if raw_prediction.shape[-1] == 1:
            prob_healthy = float(raw_prediction[0][0])
            prob_bacterial = 1.0 - prob_healthy
        else:
            prob_bacterial = float(raw_prediction[0][0])
            prob_healthy = float(raw_prediction[0][1])

    # 4. Out-of-Domain / Non-Tomato Detection Logic
    if 0.42 <= prob_healthy <= 0.58:
        st.warning("⚠️ **Uncertain Image Uploaded**")
        st.write("The image uploaded does not clearly match a Healthy or Bacterial Spot Tomato leaf.")
        st.info(f"**Healthy Tomato:** {prob_healthy * 100:.2f}% confidence")
        st.info(f"**Tomato Bacterial Spot:** {prob_bacterial * 100:.2f}% confidence")

    # 5. Clear Classifications
    elif prob_healthy > prob_bacterial:
        st.success(f"**Prediction:** {CLASS_NAMES[1]}")  # Healthy Tomato
        st.info(f"**Confidence Score:** {prob_healthy * 100:.2f}%")
        st.progress(int(prob_healthy * 100), text=f"Healthy Probability: {prob_healthy * 100:.1f}%")

        # Care recommendations for Healthy Plants
        with st.expander("🌱 **Maintenance & Prevention Tips**"):
            st.markdown("""
            * **Watering:** Water at the base of the plant (drip irrigation) to keep leaves dry.
            * **Airflow:** Prune lower foliage and space plants adequately to increase air circulation.
            * **Monitoring:** Inspect lower leaves weekly for early signs of spots or yellowing.
            """)

    else:
        st.error(f"**Prediction:** {CLASS_NAMES[0]}")  # Tomato Bacterial Spot
        st.info(f"**Confidence Score:** {prob_bacterial * 100:.2f}%")
        st.progress(int(prob_bacterial * 100), text=f"Bacterial Spot Probability: {prob_bacterial * 100:.1f}%")

        # Treatment Guidelines
        st.warning("⚠️ **Immediate Action Recommended**")

        with st.expander("🛠️ **View Treatment Guidelines & Management Remedies**", expanded=True):
            st.subheader("1. Cultural & Immediate Management")
            st.markdown("""
            * **Isolate Infected Plants:** Remove and destroy severely infected leaves or entire plants. **Do not compost infected plant debris.**
            * **Avoid Overhead Watering:** Bacterial spot spreads easily via splashing water. Switch to drip irrigation or hand-water strictly at soil level.
            * **Sanitize Tools:** Clean shears, stakes, and equipment with a 10% bleach solution or 70% alcohol between plants.
            * **Refrain from Working in Wet Fields:** Do not prune or harvest when foliage is wet to prevent mechanical transmission.
            """)

            st.subheader("2. Chemical & Organic Control")
            st.markdown("""
            * **Copper-Based Sprays:** Apply fixed copper fungicides/bactericides early in the disease outbreak.
            * **Copper + Mancozeb Combo:** Mixing fixed copper with a mancozeb-based fungicide enhances control against copper-resistant bacterial strains.
            * **Biological Control:** Apply products containing *Bacillus subtilis* or *Bacillus amyloliquefaciens* for organic suppression.
            """)

            st.subheader("3. Long-Term Prevention")
            st.markdown("""
            * **Crop Rotation:** Rotate tomato crops with non-solanaceous crops (e.g., corn, beans) for at least 2–3 years.
            * **Certified Disease-Free Seeds:** Always start with certified disease-free seeds or transplants.
            * **Resistant Varieties:** Choose tomato cultivars bred with resistance to bacterial spot races where available.
            """)