import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ✅ Correct GitHub raw CSV file link
GITHUB_CSV_URL = "https://raw.githubusercontent.com/sangambhamare/House-Price-Prediction-Regression-Model/master/data.csv"

# Function to load data from GitHub
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(GITHUB_CSV_URL)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load dataset
df = load_data()

if df is not None:
    st.title("🏡 House Price Prediction App")
    st.write("This tool predicts house prices based on key features like square footage, number of bedrooms, location, and more.")

    # Display dataset preview
    if st.checkbox("🔍 Show dataset preview"):
        st.write(df.head())

    # Drop irrelevant columns
    df = df.drop(columns=['date', 'street', 'country'])

    # Encode categorical variables
    label_encoders = {}
    for col in ['city', 'statezip']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Feature Engineering
    df['house_age'] = 2024 - df['yr_built']
    df['was_renovated'] = df['yr_renovated'].apply(lambda x: 1 if x > 0 else 0)

    # Drop original year columns
    df = df.drop(columns=['yr_built', 'yr_renovated'])

    # Define features and target
    X = df.drop(columns=['price'])
    y = df['price']

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest Model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Make predictions
    y_pred_rf = rf_model.predict(X_test)

    # Evaluate the model
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    r2_rf = r2_score(y_test, y_pred_rf)

    # Display model performance
    st.write("## 📊 Model Performance")
    st.write(f"**📉 Mean Absolute Error (MAE):** ${mae_rf:,.2f}")
    st.write(f"**📏 Root Mean Squared Error (RMSE):** ${rmse_rf:,.2f}")
    st.write(f"**📈 R² Score:** {r2_rf:.4f}")

    # Feature Importance
    feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': rf_model.feature_importances_})
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

    # Plot Feature Importance
    st.write("## 🔥 Most Important Factors Affecting Price")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(feature_importance['Feature'], feature_importance['Importance'], color='skyblue')
    ax.set_xlabel("Feature Importance Score")
    ax.set_ylabel("Features")
    ax.set_title("Feature Importance in House Price Prediction")
    ax.invert_yaxis()
    st.pyplot(fig)

    # User Input for Predictions
    st.write("## 🏠 Predict Your House Price")
    st.write("Fill in the details below to get an estimate of your house price:")

    input_features = {}

    input_features['bedrooms'] = st.number_input("🏡 How many **bedrooms** does the house have?", 
                                                 min_value=int(df['bedrooms'].min()), 
                                                 max_value=int(df['bedrooms'].max()), 
                                                 value=int(df['bedrooms'].median()))

    input_features['bathrooms'] = st.number_input("🚿 How many **bathrooms** are there?", 
                                                  min_value=float(df['bathrooms'].min()), 
                                                  max_value=float(df['bathrooms'].max()), 
                                                  value=float(df['bathrooms'].median()))

    input_features['sqft_living'] = st.number_input("📏 What is the **total living area (sqft)**?", 
                                                    min_value=int(df['sqft_living'].min()), 
                                                    max_value=int(df['sqft_living'].max()), 
                                                    value=int(df['sqft_living'].median()))

    input_features['sqft_lot'] = st.number_input("🌳 What is the **total lot size (sqft)**?", 
                                                 min_value=int(df['sqft_lot'].min()), 
                                                 max_value=int(df['sqft_lot'].max()), 
                                                 value=int(df['sqft_lot'].median()))

    input_features['floors'] = st.number_input("🏢 How many **floors** does the house have?", 
                                               min_value=float(df['floors'].min()), 
                                               max_value=float(df['floors'].max()), 
                                               value=float(df['floors'].median()))

    input_features['waterfront'] = st.radio("🌊 Does the house have a **waterfront view**?", ["No", "Yes"])
    input_features['waterfront'] = 1 if input_features['waterfront'] == "Yes" else 0

    input_features['view'] = st.slider("👀 How **good is the view** of the house? (0 = Worst, 4 = Best)", 
                                       min_value=int(df['view'].min()), 
                                       max_value=int(df['view'].max()), 
                                       value=int(df['view'].median()))

    input_features['condition'] = st.slider("🏚️ How **good is the overall condition** of the house? (1 = Poor, 5 = Excellent)", 
                                            min_value=int(df['condition'].min()), 
                                            max_value=int(df['condition'].max()), 
                                            value=int(df['condition'].median()))

    input_features['sqft_above'] = st.number_input("🏠 What is the **total above-ground square footage**?", 
                                                   min_value=int(df['sqft_above'].min()), 
                                                   max_value=int(df['sqft_above'].max()), 
                                                   value=int(df['sqft_above'].median()))

    input_features['sqft_basement'] = st.number_input("🏡 What is the **basement size (sqft)**?", 
                                                      min_value=int(df['sqft_basement'].min()), 
                                                      max_value=int(df['sqft_basement'].max()), 
                                                      value=int(df['sqft_basement'].median()))

    input_features['house_age'] = st.number_input("📅 How **old is the house** (years)?", 
                                                  min_value=int(df['house_age'].min()), 
                                                  max_value=int(df['house_age'].max()), 
                                                  value=int(df['house_age'].median()))

    input_features['was_renovated'] = st.radio("🔨 Has the house been **renovated**?", ["No", "Yes"])
    input_features['was_renovated'] = 1 if input_features['was_renovated'] == "Yes" else 0

    # Predict Price
    if st.button("📢 Get Predicted House Price"):
        input_data = pd.DataFrame([input_features])
        predicted_price = rf_model.predict(input_data)[0]
        st.success(f"🏡 **Estimated House Price:** ${predicted_price:,.2f}")

    # Copyright notice
    st.markdown("All rights reserved to Mr. Sangam Sanjay Bhamare, 2025")
