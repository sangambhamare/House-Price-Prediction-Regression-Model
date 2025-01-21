import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import joblib
import requests

# Load the model from GitHub
MODEL_URL = "https://raw.githubusercontent.com/sangambhamare/House-Price-Prediction-Regression-Model/master/lightgbm_model.h5"

def load_model():
    try:
        response = requests.get(MODEL_URL)
        response.raise_for_status()
        with open("model.h5", "wb") as f:
            f.write(response.content)
        return joblib.load("model.h5")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load the model: {e}")
        return None

model = load_model()

# GUI Layout
class HousePriceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🏡 House Price Prediction")
        self.root.geometry("500x600")

        # Create input fields
        self.create_widgets()

    def create_widgets(self):
        ttk.Label(self.root, text="🏡 House Price Prediction", font=("Arial", 16)).pack(pady=10)

        # Input fields
        self.inputs = {}
        fields = [
            ("Bedrooms", 1, 10, 3),
            ("Bathrooms", 1, 10, 2),
            ("Living Area (sqft)", 300, 10000, 1500),
            ("Lot Size (sqft)", 500, 50000, 5000),
            ("Floors", 1, 3, 1),
            ("Waterfront (0 = No, 1 = Yes)", 0, 1, 0),
            ("View Quality (0-4)", 0, 4, 1),
            ("Condition (1-5)", 1, 5, 3),
            ("Above Ground (sqft)", 300, 10000, 1500),
            ("Basement (sqft)", 0, 5000, 0),
            ("House Age (Years)", 0, 200, 20),
            ("Was Renovated (0 = No, 1 = Yes)", 0, 1, 0)
        ]

        for field, min_val, max_val, default in fields:
            frame = ttk.Frame(self.root)
            frame.pack(pady=5, fill="x")
            ttk.Label(frame, text=field, width=25, anchor="w").pack(side="left", padx=5)
            entry = ttk.Entry(frame, width=10)
            entry.pack(side="right", padx=5)
            entry.insert(0, default)
            self.inputs[field] = entry

        # Predict button
        ttk.Button(self.root, text="📢 Predict Price", command=self.predict_price).pack(pady=20)

        # Output field
        self.result_label = ttk.Label(self.root, text="", font=("Arial", 14))
        self.result_label.pack(pady=10)

    def predict_price(self):
        if not model:
            messagebox.showerror("Error", "Model is not loaded. Cannot predict.")
            return

        try:
            # Collect inputs
            input_data = {
                "bedrooms": float(self.inputs["Bedrooms"].get()),
                "bathrooms": float(self.inputs["Bathrooms"].get()),
                "sqft_living": float(self.inputs["Living Area (sqft)"].get()),
                "sqft_lot": float(self.inputs["Lot Size (sqft)"].get()),
                "floors": float(self.inputs["Floors"].get()),
                "waterfront": float(self.inputs["Waterfront (0 = No, 1 = Yes)"].get()),
                "view": float(self.inputs["View Quality (0-4)"].get()),
                "condition": float(self.inputs["Condition (1-5)"].get()),
                "sqft_above": float(self.inputs["Above Ground (sqft)"].get()),
                "sqft_basement": float(self.inputs["Basement (sqft)"].get()),
                "house_age": float(self.inputs["House Age (Years)"].get()),
                "was_renovated": float(self.inputs["Was Renovated (0 = No, 1 = Yes)"].get()),
                "city_encoded": 0,  # Placeholder
                "statezip_encoded": 0  # Placeholder
            }

            # Convert input data into DataFrame
            input_df = pd.DataFrame([input_data])

            # Predict
            predicted_price = model.predict(input_df)[0]

            # Display result
            self.result_label.config(text=f"🏡 Estimated Price: ${predicted_price:,.2f}")
        except Exception as e:
            messagebox.showerror("Error", f"Invalid input or prediction failed: {e}")

# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = HousePriceApp(root)
    root.mainloop()
