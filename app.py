import streamlit as st
import numpy as np
import joblib

# Load trained objects .
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("lebel.pkl")

st.set_page_config(page_title="Dry Bean Classifier", layout="centered")

# Custom CSS for attractive background
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .main .block-container {
        background-color: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
    }
    h1 {
        color: #000000 !important;
        text-align: center;
        font-weight: bold;
    }
    .stSlider > div > div > div > div {
        background-color: #2c3e50;
    }
    .stSlider > div > div > div > div > div {
        background-color: #34495e;
    }
    p, .stMarkdown, label, .stSlider label {
        color: #2c3e50 !important;
    }
    .stButton > button {
        color: white !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        font-weight: bold;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    .stButton > button:focus {
        outline: none;
        box-shadow: none;
    }
    div[data-testid="stMarkdownContainer"] p {
        color: #2c3e50 !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🌱 Dry Bean Type Classification")
st.write("Enter physical measurements of a dry bean to predict its class.")

st.divider()

# Feature input fields with sliders
features = []

# Define feature ranges based on typical dry bean measurements
feature_config = {
    "Area": (20000.0, 100000.0, 50000.0),
    "Perimeter": (500.0, 1500.0, 1000.0),
    "Major Axis Length": (150.0, 500.0, 300.0),
    "Minor Axis Length": (100.0, 300.0, 200.0),
    "Aspect Ratio": (1.0, 3.0, 1.5),
    "Eccentricity": (0.5, 1.0, 0.8),
    "Convex Area": (20000.0, 100000.0, 50000.0),
    "Equivalent Diameter": (150.0, 400.0, 250.0),
    "Extent": (0.5, 1.0, 0.7),
    "Solidity": (0.9, 1.0, 0.98),
    "Roundness": (0.5, 1.0, 0.7),
    "Compactness": (0.5, 1.0, 0.8),
    "ShapeFactor1": (0.001, 0.01, 0.005),
    "ShapeFactor2": (0.0001, 0.005, 0.002),
    "ShapeFactor3": (0.3, 1.0, 0.6),
    "ShapeFactor4": (0.9, 1.0, 0.98)
}

for feature, (min_val, max_val, default_val) in feature_config.items():
    value = st.slider(
        label=feature,
        min_value=min_val,
        max_value=max_val,
        value=default_val,
        step=(max_val - min_val) / 1000
    )
    features.append(value)

st.divider()

# Prediction button.
if st.button("Predict Bean Type"):
    input_array = np.array(features).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)
    predicted_class = label_encoder.inverse_transform(prediction)[0]

    st.success(f"✅ Predicted Bean Type: **{predicted_class}**")
