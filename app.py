import joblib
import pickle
import requests

# ✅ GitHub URL for the trained model
MODEL_URL = "https://raw.githubusercontent.com/sangambhamare/House-Price-Prediction-Regression-Model/master/lightgbm_model.h5"

def load_model():
    response = requests.get(MODEL_URL)
    if response.status_code == 200:
        with open("model.h5", "wb") as f:
            f.write(response.content)

        # Try loading with joblib
        try:
            model = joblib.load("model.h5")
            print("✅ Model loaded successfully with Joblib.")
            return model
        except Exception as e:
            print(f"⚠️ Joblib failed: {e}. Trying Pickle...")

        # Try loading with pickle
        try:
            with open("model.h5", "rb") as f:
                model = pickle.load(f)
            print("✅ Model loaded successfully with Pickle.")
            return model
        except Exception as e:
            print(f"⚠️ Pickle also failed: {e}. The model file may be corrupted.")

    else:
        print("⚠️ Failed to download model from GitHub.")

    return None

# Load the model
model = load_model()

if model:
    print("🚀 Model is ready for predictions!")
else:
    print("❌ Model could not be loaded. Please check the file format.")
