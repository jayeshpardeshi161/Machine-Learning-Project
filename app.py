# app.py

import streamlit as st
import numpy as np
import joblib

# Title
st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")
st.title("💳 Credit Card Fraud Detection")
st.markdown("Enter the transaction details below to check if it's **Fraudulent or Normal**")

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("credit_card_model.pkl")

model = load_model()

# Input features
features = []
feature_names = [f'V{i}' for i in range(1, 29)] + ['Amount']

st.subheader("📝 Transaction Details Input")
for name in feature_names:
    val = st.number_input(f"{name}", value=0.0, format="%.5f")
    features.append(val)

# Predict button
if st.button("🔍 Predict"):
    prediction = model.predict([features])[0]
    if prediction == 0:
        st.success("✅ This is a **Normal Transaction**")
    else:
        st.error("⚠️ This is a **Fraudulent Transaction**")
