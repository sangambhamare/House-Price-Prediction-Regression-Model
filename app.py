import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
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
    st.title("🏡 House Price Prediction - Machine Learning Model")
    st.write("This application predicts house prices based on key features like square footage, bedrooms, location, and more.")

    # Display dataset preview
    if st.checkbox("Show dataset"):
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
    st.write(f"**Mean Absolute Error (MAE):** ${mae_rf:,.2f}")
    st.write(f"**Root Mean Squared Error (RMSE):** ${rmse_rf:,.2f}")
    st.write(f"**R² Score:** {r2_rf:.4f}")

    # Feature Importance
    feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': rf_model.feature_importances_})
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

    # Plot Feature Importance
    st.write("## 🔥 Feature Importance")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(feature_importance['Feature'], feature_importance['Importance'], color='skyblue')
    ax.set_xlabel("Feature Importance Score")
    ax.set_ylabel("Features")
    ax.set_title("Feature Importance in House Price Prediction")
    ax.invert_yaxis()
    st.pyplot(fig)

    # User Input for Predictions
    st.write("## 🏠 Predict House Price")
    input_features = {}

    for feature in X.columns:
        value = st.number_input(f"Enter {feature}", float(df[feature].min()), float(df[feature].max()), float(df[feature].median()))
        input_features[feature] = value

    if st.button("Predict Price"):
        input_data = pd.DataFrame([input_features])
        predicted_price = rf_model.predict(input_data)[0]
        st.success(f"🏡 **Predicted House Price:** ${predicted_price:,.2f}")

