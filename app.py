import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests

# ✅ GitHub URL for the trained model
MODEL_URL = "https://raw.githubusercontent.com/sangambhamare/House-Price-Prediction-Regression-Model/master/lightgbm_model.h5"

# ✅ Load the trained model from GitHub
@st.cache_resource
def load_model():
    response = requests.get(MODEL_URL)
    if response.status_code == 200:
        with open("model.h5", "wb") as f:
            f.write(response.content)
        model = joblib.load("model.h5")
        return model
    else:
        st.error("⚠️ Failed to load the model from GitHub. Please check the URL.")
        return None

model = load_model()

# ✅ Streamlit App
st.title("🏡 House Price Prediction")
st.write("This app predicts house prices using a pre-trained LightGBM model.")

# Input fields for features
def user_input_features():
    st.sidebar.header("Enter House Details")
    bedrooms = st.sidebar.slider("🏡 Number of Bedrooms", 1, 10, 3)
    bathrooms = st.sidebar.slider("🚿 Number of Bathrooms", 1, 10, 2)
    sqft_living = st.sidebar.number_input("📏 Living Area (sqft)", min_value=300, max_value=10000, value=1500)
    sqft_lot = st.sidebar.number_input("🌳 Lot Size (sqft)", min_value=500, max_value=50000, value=5000)
    floors = st.sidebar.slider("🏢 Number of Floors", 1, 3, 1)
    waterfront = st.sidebar.radio("🌊 Waterfront View", ["No", "Yes"])
    view = st.sidebar.slider("👀 View Quality (0 - 4)", 0, 4, 1)
    condition = st.sidebar.slider("🏚️ Condition (1 - 5)", 1, 5, 3)
    sqft_above = st.sidebar.number_input("🏠 Above Ground Square Footage", min_value=300, max_value=10000, value=1500)
    sqft_basement = st.sidebar.number_input("🏡 Basement Square Footage", min_value=0, max_value=5000, value=0)
    house_age = st.sidebar.number_input("📅 Age of the House (Years)", min_value=0, max_value=200, value=20)
    was_renovated = st.sidebar.radio("🔨 Was the House Renovated?", ["No", "Yes"])

    # Convert categorical inputs
    waterfront = 1 if waterfront == "Yes" else 0
    was_renovated = 1 if was_renovated == "Yes" else 0

    # Placeholder values for city and statezip encoding
    city_encoded = 0
    statezip_encoded = 0

    # Create DataFrame for prediction
    data = {
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "sqft_living": sqft_living,
        "sqft_lot": sqft_lot,
        "floors": floors,
        "waterfront": waterfront,
        "view": view,
        "condition": condition,
        "sqft_above": sqft_above,
        "sqft_basement": sqft_basement,
        "house_age": house_age,
        "was_renovated": was_renovated,
        "city_encoded": city_encoded,
        "statezip_encoded": statezip_encoded
    }

    return pd.DataFrame([data])

# Prediction logic
if model:
    user_input = user_input_features()
    if st.button("📢 Predict Price"):
        prediction = model.predict(user_input)[0]
        st.success(f"🏡 **Estimated House Price:** ${prediction:,.2f}")
else:
    st.error("⚠️ Model is not available.")
