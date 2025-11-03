import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import os
import plotly.express as px  # For interactive charts

# --- 1. Page Configuration ---
# Set the layout to wide for a more modern feel
st.set_page_config(layout="wide")

# --- 2. Get the absolute path to the CSV ---
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'crop_production.csv')
# --- !! IMPORTANT: Add an image named 'header.jpg' to your folder !! ---
image_path = os.path.join(script_dir, 'header.jpg')


# --- 3. Caching the Model and Preprocessors ---
@st.cache_resource
def load_model_and_data(path_to_csv):
    """
    Loads data, preprocesses it, and trains the RandomForest model.
    Returns the trained model, scaler, encoders, original data for charts, and R2 score.
    """
    try:
        data = pd.read_csv(path_to_csv)
    except FileNotFoundError:
        return None, None, None, None, None, f"Error: 'crop_production.csv' not found. Make sure it's in the same folder as this app."

    # --- Preprocessing ---
    data.dropna(inplace=True)
    data.drop(columns='District_Name', inplace=True)
    
    # Keep a copy of the *original* data for charts and dropdowns
    original_data_for_charts = data.copy()

    # --- Sample the data (from your fixed code) ---
    if len(data) > 50000:
        st.info(f"Original data has {len(data)} rows. Sampling 50,000 for training to conserve memory...")
        training_data = data.sample(n=50000, random_state=42)
    else:
        training_data = data.copy()
        
    # --- Encoders ---
    encoders = {}
    for col in ['State_Name', 'Season', 'Crop']:
        le = LabelEncoder()
        # Fit on the *entire* dataset to know all categories
        data[col] = le.fit_transform(data[col])
        # Transform the *sample* for training
        training_data[col] = le.transform(training_data[col])
        encoders[col] = le

    # --- Define X and y from the SAMPLE ---
    features = ['State_Name', 'Crop_Year', 'Season', 'Crop', 'Area']
    target = 'Production'
    
    X = training_data[features]
    y = training_data[target]

    # --- !! NEW: Perform Train/Test Split to get an R2 Score !! ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Scale Data ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Train the Final Model (Tuned for memory) ---
    st.info(f"Training model on {len(X_train)} rows... Please wait.")
    
    final_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=25,
        random_state=42,
        n_jobs=-1
    )
    final_model.fit(X_train_scaled, y_train)
    
    # --- !! NEW: Calculate R2 Score !! ---
    y_pred = final_model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    
    st.success(f"Model training complete! (Test R²: {r2:.2f})")
    
    return final_model, scaler, encoders, original_data_for_charts, r2, None

# --- 4. Load Model and Data ---
model, scaler, encoders, original_data, r2, error_msg = load_model_and_data(csv_path)

# --- 5. Build the User Interface ---
st.title('🌱 Indian Crop Production Forecaster')

# --- !! NEW: Add a header image !! ---
if os.path.exists(image_path):
    st.image(image_path, use_column_width=True)
else:
    st.info("Place an image named 'header.jpg' in the app folder to display a header.")

if error_msg:
    st.error(error_msg)
else:
    # --- !! NEW: Create Tabs !! ---
    tab1, tab2 = st.tabs(["📊 Prediction Tool", "📈 Data Insights"])

    # --- Tab 1: The Prediction Tool ---
    with tab1:
        st.header('Make a New Prediction')
        
        # Get unique values for dropdowns
        state_options = sorted(original_data['State_Name'].unique())
        season_options = sorted(original_data['Season'].unique())
        crop_options = sorted(original_data['Crop'].unique())
        
        # --- Create sidebar for user inputs ---
        st.sidebar.header('Enter Your Parameters:')
        
        state = st.sidebar.selectbox('State:', options=state_options)
        year = st.sidebar.number_input('Crop Year:', min_value=1997, max_value=2030, value=2025)
        season = st.sidebar.selectbox('Season:', options=season_options)
        crop = st.sidebar.selectbox('Crop:', options=crop_options)
        area = st.sidebar.number_input('Area (in Hectares):', min_value=0.0, value=1000.0, step=100.0)

        # --- Prediction Logic ---
        if st.sidebar.button('Predict Production'):
            try:
                input_data = pd.DataFrame({
                    'State_Name': [state], 'Crop_Year': [year], 'Season': [season],
                    'Crop': [crop], 'Area': [area]
                })
                
                # Transform categorical inputs
                input_data['State_Name'] = encoders['State_Name'].transform(input_data['State_Name'])
                input_data['Season'] = encoders['Season'].transform(input_data['Season'])
                input_data['Crop'] = encoders['Crop'].transform(input_data['Crop'])
                
                # Scale the inputs
                input_data_scaled = scaler.transform(input_data)
                
                # Make the prediction
                prediction = model.predict(input_data_scaled)
                
                # --- !! NEW: Display with st.metric and balloons !! ---
                st.subheader('Prediction Result:')
                st.metric(label="Estimated Crop Production", value=f"{prediction[0]:,.0f} tonnes")
                st.balloons()
            
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.error("This might be because the model did not see this exact combination of inputs during training (e.g., a 'Crop' that never appears in a specific 'State').")

        st.sidebar.info(f"This app uses a RandomForestRegressor with an R² score of {r2:.2f} (trained on a 50k sample).")

        st.markdown("---")
        st.markdown("""
        ### ℹ️ How to Use
        1.  Select the **State**, **Season**, and **Crop** from the dropdowns on the left.
        2.  Enter the **Crop Year** you want to predict for.
        3.  Enter the total **Area** under cultivation (in hectares).
        4.  Click **'Predict Production'** to see the result.
        """)

    # --- Tab 2: The Data Insights Page ---
    with tab2:
        st.header("Insights from the Full Dataset")
        st.markdown("These charts are based on the complete, cleaned dataset.")

        # --- !! NEW: Chart 1 - Top 10 States !! ---
        st.subheader("Top 10 States by Total Production")
        state_prod = original_data.groupby('State_Name')['Production'].sum().nlargest(10).reset_index()
        fig1 = px.bar(state_prod, x='State_Name', y='Production', title='Top 10 States', color='State_Name')
        st.plotly_chart(fig1, use_container_width=True)

        # --- !! NEW: Chart 2 - Top 10 Crops !! ---
        st.subheader("Top 10 Crops by Total Production")
        crop_prod = original_data.groupby('Crop')['Production'].sum().nlargest(10).reset_index()
        fig2 = px.bar(crop_prod, x='Crop', y='Production', title='Top 10 Crops', color='Crop')
        st.plotly_chart(fig2, use_container_width=True)

        # --- !! NEW: Chart 3 - Production Over Time !! ---
        st.subheader("Total Production by Year")
        year_prod = original_data.groupby('Crop_Year')['Production'].sum().reset_index()
        fig3 = px.line(year_prod, x='Crop_Year', y='Production', title='Total Production Over Time', markers=True)
        st.plotly_chart(fig3, use_container_width=True)
        
        # --- !! NEW: Data Table !! ---
        st.subheader("Browse a Sample of the Data")
        st.dataframe(original_data.sample(1000))