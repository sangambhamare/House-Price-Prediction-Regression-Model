# 🏡 **House Price Prediction - Machine Learning Model**  

## 📌 **Project Overview**  
This **House Price Prediction App** uses **machine learning** to estimate house prices based on key features such as **square footage, number of bedrooms, location, view, and more**. The app is built using **Python, Streamlit, and Scikit-learn** and is designed to provide an **interactive user experience** for predicting house prices.

---

## 🚀 **Key Features**
✅ **Load real estate dataset** from GitHub  
✅ **Preprocess data** (handle categorical & numerical features)  
✅ **Feature Engineering** (house age, renovation status)  
✅ **Train a Random Forest Model** for accurate predictions  
✅ **Evaluate Model Performance** (MAE, RMSE, R² Score)  
✅ **Visualize Feature Importance**  
✅ **User-friendly Interface** to input house details  
✅ **Live House Price Prediction**  

---

## 📊 **Project Structure**
```bash
House-Price-Prediction/
│── app.py                # Main Streamlit app
│── requirements.txt      # Dependencies
│── data.csv              # Dataset (Stored in GitHub)
│── README.md             # Documentation
```

---

## 💻 **How to Run the Project**
### **🔹 1. Clone the Repository**
```bash
git clone https://github.com/your-username/House-Price-Prediction.git
cd House-Price-Prediction
```

### **🔹 2. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **🔹 3. Run the Streamlit App**
```bash
streamlit run app.py
```
The app will open in your **web browser**.

---

## 🔥 **Technologies Used**
- **Python** 🐍  
- **Streamlit** 🎨 *(for interactive UI)*  
- **Pandas & NumPy** 📊 *(for data preprocessing)*  
- **Scikit-learn** 🤖 *(for machine learning models)*  
- **Matplotlib** 📈 *(for visualizations)*  

---

## 🎯 **How to Use the App**
1. **View Dataset:** Click the checkbox to explore the data.  
2. **Predict House Price:**  
   - Enter details like **bedrooms, bathrooms, house size, view, floors, renovation status, etc.**  
   - Click **"Get Predicted House Price"** to see the estimated price.  
3. **View Feature Importance:** See which features impact house prices the most.

---

## 📉 **Model Performance**
- **Mean Absolute Error (MAE):** 🏠 **\$165,051.21**  
- **Root Mean Squared Error (RMSE):** 📏 **\$988,778.64**  
- **R² Score:** 📈 **0.041 (Low, but improvable with advanced models)**  
- **Most Important Features:** `sqft_living`, `statezip`, `house_age`, `city`, `sqft_above`  

---

## 🛠️ **Future Improvements**
- ✅ Implement **Gradient Boosting (XGBoost, LightGBM)**
- ✅ Optimize **Hyperparameters for better accuracy**
- ✅ Remove **outliers** to improve predictions
- ✅ Add **Geolocation data** for better price estimation

---



