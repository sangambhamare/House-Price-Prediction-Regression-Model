import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# GitHub raw CSV file link (Replace with your actual raw link)
GITHUB_CSV_URL = "data.csv"

# Function to load data from GitHub
@st.cache_data
def load_data():
    response = requests.get(GITHUB_CSV_URL)
    if response.status_code == 200:
        csv_data = StringIO(response.text)
        df = pd.read_csv(csv_data)
        return df
    else:
        st.error("Failed to load data from GitHub. Please check the URL.")
        return None

# Load dataset
df = load_data()

if df is not None:
    st.title("🏡 House Price Prediction")
    st.write("This app predicts house prices based on input features.")

    # Display dataset preview
    if st.checkbox("Show dataset"):
        st.write(df.head())

    # Selecting features and target variable
    target_column = "Price"  # Change this to match your dataset
    features = [col for col in df.columns if col != target_column]

    # Handle missing values
    df = df.dropna()

    # Split data into training and testing sets
    X = df[features]
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Model Evaluation
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    st.write("### Model Performance")
    st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
    st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.2f}")
    st.write(f"**R² Score:** {r2:.2f}")

    # User input section for predictions
    st.write("### Predict House Price")
    input_features = {}

    for feature in features:
        value = st.number_input(f"Enter {feature}", float(df[feature].min()), float(df[feature].max()), float(df[feature].median()))
        input_features[feature] = value

    if st.button("Predict Price"):
        input_data = pd.DataFrame([input_features])
        predicted_price = model.predict(input_data)[0]
        st.success(f"Predicted House Price: ${predicted_price:,.2f}")

