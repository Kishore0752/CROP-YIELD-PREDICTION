# 🌾 Indian Crop Production Predictor

This project is a machine learning model built to predict crop production in India. It uses historical data to train a Random Forest Regressor and is deployed as an interactive web application using Streamlit.



## 📝 Overview

The goal of this project is to forecast the total production (in tonnes) of a specific crop based on several key factors:
* State
* Crop Type
* Season
* Crop Year
* Area of cultivation (in hectares)

The model was trained on the "Crop Production in India" dataset (`crop_production.csv`).

## ✨ Features

* **Prediction Tool:** A web-based UI where users can input parameters and get an instant production forecast.
* **Data Insights:** An interactive dashboard with Plotly charts to visualize key findings, such as:
    * Top 10 States by Total Production
    * Top 10 Crops by Total Production
    * Total Production by Year
* **Machine Learning Model:** A highly accurate **`RandomForestRegressor`** model that achieved a **90.2% R-squared score** on the test data.

## 🛠️ Technologies Used

* **Machine Learning:** Scikit-learn (`RandomForestRegressor`, `StandardScaler`, `LabelEncoder`)
* **Data Analysis:** Pandas, NumPy
* **Web Framework:** Streamlit
* **Data Visualization:** Plotly, Seaborn, Matplotlib

## 🚀 How to Run This Project Locally

Follow these steps to run the web application on your own machine.

**1. Clone the Repository:**
```bash
git clone [https://github.com/Kishore0752/CROP-PRODUCTION-PREDICTION.git](https://github.com/Kishore0752/CROP-PRODUCTION-PREDICTION.git)
cd CROP-PRODUCTION-PREDICTION
