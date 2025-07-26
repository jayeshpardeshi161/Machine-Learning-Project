# app.py

import streamlit as st
import numpy as np
import joblib
import os

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")
st.title("💳 Credit Card Fraud Detection")
st.markdown("Enter or auto-fill transaction details to check if it's **Fraudulent or Normal**")

# --- Load Trained Model with Caching and Error Handling ---
@st.cache_resource
def load_model():
    if not os.path.exists("credit_card_model.pkl"):
        st.error("❌ Model file not found. Make sure 'credit_card_model.pkl' exists in the app directory.")
        st.stop()
    model = joblib.load("credit_card_model.pkl")
    return model

model = load_model()

# --- Define Feature Names (Same order as training data) ---
feature_names = [f'V{i}' for i in range(1, 29)] + ['Amount']

# --- Validate Feature Order ---
try:
    model_features = model.feature_names_in_.tolist()
    if model_features != feature_names:
        st.warning("⚠️ Feature mismatch between input and model. Check the training pipeline.")
except AttributeError:
    st.info("ℹ️ Model does not expose 'feature_names_in_'. Skipping feature order check.")

# --- Sample Transactions for Quick Fill ---
sample_normal = [-1.35981, -0.07278, 2.53635, 1.37815, -0.33832, 0.46239, 0.2396, 0.0987, 0.36379, 0.0908,
                 -0.5516, -0.6178, -0.9913, -0.3111, 1.4681, -0.4704, 0.2071, 0.0257, 0.40399, 0.25141,
                 -0.0183, 0.2778, -0.1105, 0.0669, 0.1285, -0.1891, 0.1335, -0.0210, 149.62]

sample_fraud = [-2.3122, 1.9511, -1.6099, 3.9979, -4.6062, 0.4226, -0.7982, 0.3255, -0.9927, -1.1800,
                -2.0554, -2.2619, -0.3586, -1.1407, -0.3947, -1.7427, -1.8615, 0.0185, 0.2778, 0.0851,
                -0.0332, 0.1232, 0.0494, 0.1047, 0.0830, 0.0320, -0.0750, 0.0210, 0.0]

# --- Initialize Session State ---
if "inputs" not in st.session_state:
    st.session_state.inputs = [0.0] * len(feature_names)
    st.session_state.mode = "Manual Input"

# --- Input Mode Selection ---
st.subheader("📝 Choose Input Mode")
input_mode = st.radio("Select input mode:", ["Manual Input", "Normal Transaction", "Fraudulent Transaction"],
                      index=["Manual Input", "Normal Transaction", "Fraudulent Transaction"].index(st.session_state.mode))

# --- Update Input Based on Mode ---
if input_mode != st.session_state.mode:
    if input_mode == "Normal Transaction":
        st.session_state.inputs = sample_normal.copy()
    elif input_mode == "Fraudulent Transaction":
        st.session_state.inputs = sample_fraud.copy()
    else:
        st.session_state.inputs = [0.0] * len(feature_names)
    st.session_state.mode = input_mode

# --- Transaction Input Form ---
st.subheader("📥 Transaction Details Input")
features = []
for i, name in enumerate(feature_names):
    val = st.number_input(
        f"{name}",
        value=float(st.session_state.inputs[i]),
        format="%.5f",
        key=f"input_{name}",
        disabled=input_mode != "Manual Input",
        help="Enter a numerical value (PCA-transformed features from the original dataset)"
    )
    features.append(val)
    if input_mode == "Manual Input":
        st.session_state.inputs[i] = val

# --- Predict and Reset Buttons ---
col1, col2 = st.columns(2)

with col1:
    if st.button("🔍 Predict", key="predict_button"):
        prediction = model.predict([features])[0]
        probability = model.predict_proba([features])[0][1]

        if prediction == 0:
            st.success("✅ This is a **Normal Transaction**")
        else:
            st.error("⚠️ This is a **Fraudulent Transaction**")

        st.info(f"🔎 Fraud Probability: **{probability:.4f}**")

with col2:
    if st.button("🔄 Reset"):
        st.session_state.inputs = [0.0] * len(feature_names)
        st.session_state.mode = "Manual Input"
        st.experimental_rerun()

# --- Footer ---
st.markdown("---")
st.markdown("🔐 **Note**: This app is for demonstration purposes. The model was trained on PCA-transformed features from the Kaggle credit card fraud dataset.")
