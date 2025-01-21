# 🚀 Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor
import joblib
import requests
from io import StringIO

# ✅ GitHub Raw CSV URL (Updated)
GITHUB_CSV_URL = "https://raw.githubusercontent.com/sangambhamare/House-Price-Prediction-Regression-Model/master/data.csv"

# ✅ Load dataset from GitHub
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

df = load_data()

# ✅ Preprocess dataset
def preprocess_data(df):
    if df is None:
        return None

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

    # Remove Outliers (Top 1% most expensive homes)
    price_threshold = df['price'].quantile(0.99)
    df = df[df['price'] <= price_threshold]

    return df

df = preprocess_data(df)

if df is not None:
    # ✅ Split dataset
    X = df.drop(columns=['price'])
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ✅ Train and Optimize LightGBM Model
    @st.cache_data
    def train_model():
        param_grid = {
            'n_estimators': [300, 500],
            'learning_rate': [0.05, 0.1],
            'max_depth': [6, 8],
            'num_leaves': [31, 50],
            'min_child_samples': [10, 20],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }

        # Perform Grid Search CV
        grid_search = GridSearchCV(LGBMRegressor(random_state=42), param_grid, cv=5, n_jobs=-1, verbose=1, scoring='r2')
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_

        # Train best model
        best_lgbm_model = LGBMRegressor(**best_params, random_state=42)
        best_lgbm_model.fit(X_train, y_train)

        # Save the model
        joblib.dump(best_lgbm_model, "best_lgbm_model.pkl")

        return best_lgbm_model, best_params

    model, best_params = train_model()

    # ✅ Evaluate Model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # ✅ Streamlit UI
    st.title("🏡 Optimized House Price Prediction")
    st.write("This app predicts house prices using an optimized LightGBM model.")

    # Display dataset preview
    if st.checkbox("🔍 Show dataset preview"):
        st.write(df.head())

    st.write("### 📊 Model Performance")
    st.write(f"✅ **Mean Absolute Error (MAE):** ${mae:,.2f}")
    st.write(f"✅ **Root Mean Squared Error (RMSE):** ${rmse:,.2f}")
    st.write(f"✅ **R² Score:** {r2:.4f}")

    st.write("### 🔥 Feature Importance")
    feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
    feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance, palette="Blues_r")
    ax.set_xlabel("Feature Importance Score")
    ax.set_ylabel("Features")
    ax.set_title("Feature Importance in House Price Prediction")
    st.pyplot(fig)

    # ✅ User Input for Prediction
    st.write("### 🏠 Predict House Price")
    input_features = {}

    for feature in X.columns:
        value = st.number_input(f"Enter {feature}", float(df[feature].min()), float(df[feature].max()), float(df[feature].median()))
        input_features[feature] = value

    if st.button("📢 Get Predicted House Price"):
        input_data = pd.DataFrame([input_features])
        predicted_price = model.predict(input_data)[0]
        st.success(f"🏡 **Estimated House Price:** ${predicted_price:,.2f}")

    st.markdown("All rights reserved to Mr. Sangam Sanjay Bhamare, 2025")
else:
    st.error("⚠️ Dataset could not be loaded. Please check the GitHub URL.")

