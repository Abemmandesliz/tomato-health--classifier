# 🍅 Tomato Health Classifier App (Group CE16)

This repository contains an AI-powered Streamlit web application that classifies tomato leaf images into two main categories: **Healthy Tomato** and **Tomato Bacterial Spot**.

---

## 🌟 Key Features
- **Real-time Image Classification:** Detects disease symptoms on uploaded leaf images.
- **Confidence Thresholding:** Flags low-confidence or non-leaf uploads to prevent false predictions.
- **Interactive Web Interface:** Lightweight, intuitive UI built with Streamlit and TensorFlow/Keras.

---

## 📁 Repository Structure

```text
tomato-health--classifier/
├── .gitignore              # Files and directories ignored by Git (e.g., venv)
├── README.md               # Project documentation and setup guide
├── app.py                  # Main Streamlit web application script
├── requirements.txt        # Python package dependencies
└── tomato_model.keras      # Trained Convolutional Neural Network (CNN) model