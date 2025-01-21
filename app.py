# 🚀 Import necessary libraries
from dash import Dash, html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import joblib

# Load the trained model
MODEL_URL = "https://raw.githubusercontent.com/sangambhamare/House-Price-Prediction-Regression-Model/master/lightgbm_model.h5"

# Load the model
def load_model():
    import requests
    response = requests.get(MODEL_URL)
    if response.status_code == 200:
        with open("model.h5", "wb") as f:
            f.write(response.content)
        model = joblib.load("model.h5")
        return model
    else:
        raise ValueError("Failed to load the model. Please check the URL.")

model = load_model()

# Define feature names
FEATURES = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
    "waterfront", "view", "condition", "sqft_above", "sqft_basement",
    "house_age", "was_renovated", "city_encoded", "statezip_encoded"
]

# App initialization
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "🏡 House Price Prediction"

# App Layout
app.layout = html.Div(
    children=[
        html.H1("🏡 House Price Prediction", style={"textAlign": "center", "marginTop": "20px"}),
        html.P("Enter the details below to estimate the price of your house:", style={"textAlign": "center"}),
        
        # Input form
        dbc.Container(
            [
                dbc.Row([
                    dbc.Col(dbc.InputGroup([dbc.InputGroupText("Bedrooms"), dcc.Input(id="bedrooms", type="number", min=1, max=10, value=3)])),
                    dbc.Col(dbc.InputGroup([dbc.InputGroupText("Bathrooms"), dcc.Input(id="bathrooms", type="number", min=1, max=10, value=2)])),
                ], className="mb-3"),

                dbc.Row([
                    dbc.Col(dbc.InputGroup([dbc.InputGroupText("Living Area (sqft)"), dcc.Input(id="sqft_living", type="number", min=300, max=10000, value=1500)])),
                    dbc.Col(dbc.InputGroup([dbc.InputGroupText("Lot Size (sqft)"), dcc.Input(id="sqft_lot", type="number", min=500, max=50000, value=5000)])),
                ], className="mb-3"),

                dbc.Row([
                    dbc.Col(dbc.InputGroup([dbc.InputGroupText("Floors"), dcc.Input(id="floors", type="number", min=1, max=3, value=1)])),
                    dbc.Col(dbc.InputGroup([dbc.InputGroupText("Waterfront"), dcc.Dropdown(id="waterfront", options=[{"label": "No", "value": 0}, {"label": "Yes", "value": 1}], value=0)])),
                ], className="mb-3"),

                dbc.Row([
                    dbc.Col(dbc.InputGroup([dbc.InputGroupText("View Quality (0-4)"), dcc.Slider(id="view", min=0, max=4, step=1, value=1)])),
                    dbc.Col(dbc.InputGroup([dbc.InputGroupText("Condition (1-5)"), dcc.Slider(id="condition", min=1, max=5, step=1, value=3)])),
                ], className="mb-3"),

                dbc.Row([
                    dbc.Col(dbc.InputGroup([dbc.InputGroupText("Above Ground (sqft)"), dcc.Input(id="sqft_above", type="number", min=300, max=10000, value=1500)])),
                    dbc.Col(dbc.InputGroup([dbc.InputGroupText("Basement (sqft)"), dcc.Input(id="sqft_basement", type="number", min=0, max=5000, value=0)])),
                ], className="mb-3"),

                dbc.Row([
                    dbc.Col(dbc.InputGroup([dbc.InputGroupText("House Age (Years)"), dcc.Input(id="house_age", type="number", min=0, max=200, value=20)])),
                    dbc.Col(dbc.InputGroup([dbc.InputGroupText("Renovated"), dcc.Dropdown(id="was_renovated", options=[{"label": "No", "value": 0}, {"label": "Yes", "value": 1}], value=0)])),
                ], className="mb-3"),
                
                # Predict Button
                dbc.Button("📢 Predict Price", id="predict-btn", color="primary", className="mt-3"),
            ]
        ),

        # Prediction Result
        html.Div(id="prediction-result", style={"textAlign": "center", "marginTop": "20px", "fontSize": "24px", "fontWeight": "bold"}),
    ]
)

# Callback for prediction
@app.callback(
    Output("prediction-result", "children"),
    [Input("predict-btn", "n_clicks")],
    [
        State("bedrooms", "value"), State("bathrooms", "value"), State("sqft_living", "value"),
        State("sqft_lot", "value"), State("floors", "value"), State("waterfront", "value"),
        State("view", "value"), State("condition", "value"), State("sqft_above", "value"),
        State("sqft_basement", "value"), State("house_age", "value"), State("was_renovated", "value")
    ]
)
def predict_price(n_clicks, bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view, condition, sqft_above, sqft_basement, house_age, was_renovated):
    if n_clicks:
        # Prepare the input for prediction
        input_data = pd.DataFrame([{
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
            "city_encoded": 0,  # Placeholder
            "statezip_encoded": 0  # Placeholder
        }])

        # Predict
        predicted_price = model.predict(input_data)[0]
        return f"🏡 Estimated House Price: ${predicted_price:,.2f}"
    return ""

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
