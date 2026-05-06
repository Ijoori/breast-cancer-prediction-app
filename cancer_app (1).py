import streamlit as st
import pandas as pd
import joblib

# Load files
model = joblib.load("cancer_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("Breast Cancer Prediction App")
st.write("Enter tumor features to predict whether the tumor is Benign or Malignant.")

# Inputs
radius_mean = st.number_input("Radius Mean", value=0.0)
texture_mean = st.number_input("Texture Mean", value=0.0)
perimeter_mean = st.number_input("Perimeter Mean", value=0.0)
area_mean = st.number_input("Area Mean", value=0.0)
concavity_mean = st.number_input("Concavity Mean", value=0.0)

# Prepare input
input_data = pd.DataFrame([{
    "radius_mean": radius_mean,
    "texture_mean": texture_mean,
    "perimeter_mean": perimeter_mean,
    "area_mean": area_mean,
    "concavity_mean": concavity_mean
}])

# Keep same column order used during training
input_data = input_data[model_columns]

input_scaled = scaler.transform(input_data)

# Predict
if st.button("Predict Diagnosis"):
    prediction = model.predict(input_scaled)
    result = label_encoder.inverse_transform(prediction)

    if result[0] == "B":
        st.success("Prediction: Benign (Non-Cancerous)")
    else:
        st.error("Prediction: Malignant (Cancerous)")
