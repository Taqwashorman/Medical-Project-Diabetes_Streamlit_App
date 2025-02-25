import streamlit as st
import numpy as np
import joblib

# Load the trained model and scaler
try:
    diabetes_model = joblib.load('diabetes_model.pkl')
    scaler = joblib.load('scaler.pkl')
    model_loaded = True
except Exception as e:
    st.error(f"⚠️ Error loading model: {e}")
    model_loaded = False

st.title("🩺 Diabetes Prediction App")
st.header("Enter Patient Details:")

# Getting the input data from USER
Pregnancies = st.number_input("Number of Pregnancies", min_value=0.0, max_value=17.0, value=0.0, format="%.1f")
Glucose = st.number_input("Level of Glucose", min_value=44.0, max_value=199.0,value=117.0, format="%.1f")
BloodPressure = st.number_input("Blood Pressure", min_value=40.0, max_value=104.0,value=72.0,format="%.1f")
Insulin = st.number_input("Level of Insulin", min_value=14.0, max_value=273.0 ,value=31.0, format="%.1f")
BMI = st.number_input("Body Mass Index", min_value=18.2, max_value=51.0,value=32.0, format="%.2f")
DiabetesPedigreeFunction = st.number_input("Diabetes Ratio", min_value=0.078, max_value=1.500,value=0.620 ,format="%.3f")
Age = st.number_input("Age", min_value=21, max_value=81,value=41)

# Code for prediction
diab_diagnosis = ''

# Prediction
if st.button("🔍 Predict"):
    # Create a NumPy array from the inputs
    input_data = np.array([[Pregnancies, Glucose, BloodPressure, Insulin, BMI, DiabetesPedigreeFunction, Age]])
    
    # Apply the scaler transformation
    input_data_scaled = scaler.transform(input_data)
    
    # Make the prediction
    diab_pred = diabetes_model.predict(input_data_scaled)

    # Display the result    
    if diab_pred[0] == 0:
        diab_diagnosis = "✅ The person is NOT diabetic"
        st.success(diab_diagnosis)
    else:
        diab_diagnosis = "⚠️The person is diabetic.."
        st.error(diab_diagnosis)
else:
        st.warning("⚠️ Model not loaded. Please check the files.")
