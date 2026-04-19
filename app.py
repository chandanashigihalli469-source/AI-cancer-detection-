import streamlit as st
import numpy as np
import pandas as pd
import pickle

st.set_page_config(page_title="Cancer Detection", layout="centered")

st.title("🧬 AI-Based Cancer Detection System")

# -------------------------------
# LOAD MODEL
# -------------------------------
model, feature_names = pickle.load(open('model.pkl', 'rb'))

# -------------------------------
# LOAD DATA
# -------------------------------
data = pd.read_csv("Cancer_Data.csv")

# Drop unwanted columns
for col in ['id', 'ID_REF', 'Unnamed: 0']:
    if col in data.columns:
        data = data.drop(col, axis=1)

# -------------------------------
# INPUTS (ONLY 3)
# -------------------------------
st.subheader("Enter Biomarker Values")

radius = st.number_input("radius_mean", value=10.0)
perimeter = st.number_input("perimeter_mean", value=60.0)
area = st.number_input("area_mean", value=300.0)

st.markdown("---")

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("🔍 Predict"):

    # Create base input (mean values)
    input_dict = data.drop(['diagnosis'], axis=1).mean().to_dict()

    # Replace with user inputs
    input_dict['radius_mean'] = radius
    input_dict['perimeter_mean'] = perimeter
    input_dict['area_mean'] = area

    # Convert to dataframe
    input_df = pd.DataFrame([input_dict])

    # 🔥 IMPORTANT FIX: match training feature order
    input_df = input_df[feature_names]

    # Predict
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    # Output
    if prediction == 1:
        st.error(f"⚠️ Tumor Detected (Malignant)\n\nConfidence: {probability[1]:.2f}")
    else:
        st.success(f"✅ Normal (Benign)\n\nConfidence: {probability[0]:.2f}")

    # Debug (optional)
    st.write("Input Data:", input_df)

st.markdown("---")
st.caption("Developed using Machine Learning & Streamlit")