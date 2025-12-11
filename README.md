# 🌾 Crop Production Prediction Web App

This project is a **Streamlit-based Machine Learning application** that predicts **crop production in India** using historical agricultural data.  
It uses **Random Forest Regression**, along with preprocessing, encoding, scaling, and interactive data visualizations.  
The repository also contains a **Jupyter Notebook** (`Crop.ipynb`) used for exploration, cleaning, and model experimentation.

---

## 📁 Project Structure

```
├── Crop.ipynb              # Jupyter notebook for data exploration & visualizations
├── demo.py                 # Main Streamlit web application
├── crop_production.csv     # Dataset used for training & insights
├── requirements.txt        # Full dependency list (Streamlit + Notebook)
└── README.md               # Project documentation
```

---

## 🚀 Features

### ✅ **1. Crop Production Prediction**
- Predicts production using **RandomForestRegressor**
- Clean sidebar-based input system
- Displays results with `st.metric` and animations
- Shows model accuracy with **R² score**

### ✅ **2. Interactive Data Insights**
- Top-performing states by production  
- Top crops by total yield  
- Production changes over the years  
- Interactive Plotly charts  
- Sample dataset preview

### ✅ **3. Modern Streamlit Interface**
- Two-tab layout (Prediction / Insights)
- Optional `header.jpg` banner image support
- Cached model loading for faster performance

### ✅ **4. Jupyter Notebook Included**
`Crop.ipynb` contains:
- Data cleaning  
- Exploratory Data Analysis  
- Visualizations (Matplotlib, Seaborn, Plotly)  
- Model experiments  

---

## 🛠 Installation

### 1️⃣ Install all dependencies
```
pip install -r requirements.txt
```

### 2️⃣ Run the Streamlit App
```
streamlit run demo.py
```

### 3️⃣ (Optional) Open the Notebook
```
jupyter notebook Crop.ipynb
```

---

## 📦 requirements.txt (Used by Both App + Notebook)

```
streamlit
pandas
numpy
scikit-learn
plotly
matplotlib
seaborn
jupyter
```

---

## 🧠 How It Works

### 🔹 Data Preprocessing
- Removes missing values  
- Drops unnecessary columns (e.g., `District_Name`)  
- Encodes `State_Name`, `Season`, `Crop` using LabelEncoding  
- Scales numerical features with **StandardScaler**

### 🔹 Model Training
- Train/Test split  
- RandomForestRegressor (optimized depth & estimators)  
- Calculates **R² score** for performance  
- Caches training results for faster Streamlit loading

### 🔹 Prediction Pipeline
User inputs:
- State  
- Season  
- Crop  
- Crop Year  
- Total Area  

→ Encoded → Scaled → Model Predicts Production

---

## 📊 Data Insights Provided

- Top 10 states by total production  
- Top 10 crops  
- Yearly production trends  
- Sample dataset view (1000 rows)

All charts are interactive and built with **Plotly**.

---

## ⚠ Important Notes

- Ensure `crop_production.csv` is in the same folder as `demo.py`.  
- Add a `header.jpg` file if you want a banner in the app.  
- Some rare combinations may produce lower accuracy if unseen in training.

---

## ✨ Future Enhancements

- Add XGBoost / CatBoost models  
- Geo-visualization using maps  
- Downloadable prediction reports (PDF/CSV)  
- LSTM/ARIMA for time-series forecasting  

---

