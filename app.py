import streamlit as st
import numpy as np
import pandas as pd
import pickle

st.set_page_config(page_title="Cancer Detection", layout="centered")

st.title("🧬 AI-Based Cancer Detection System")

# 🔹 Load model
model, feature_names = pickle.load(open('model.pkl', 'rb'))

# 🔹 Load dataset (for mean values)
data = pd.read_csv("Cancer_Data.csv")

# 🔹 Remove unwanted column
if 'id' in data.columns:
    data = data.drop('id', axis=1)

# 🔹 Input fields (only 3 for simplicity)
st.subheader("Enter Biomarker Values")

radius = st.number_input("radius_mean", value=10.0)
perimeter = st.number_input("perimeter_mean", value=60.0)
area = st.number_input("area_mean", value=300.0)

st.markdown("---")

# 🔹 Prediction
if st.button("🔍 Predict"):

    # Fill all features with mean values
    input_dict = data.drop(['diagnosis'], axis=1).mean().to_dict()

    # Replace with user inputs
    input_dict['radius_mean'] = radius
    input_dict['perimeter_mean'] = perimeter
    input_dict['area_mean'] = area

    # Convert to dataframe
    input_df = pd.DataFrame([input_dict])

    prediction = model.predict(input_df)

    if prediction[0] == 1:
        st.error("⚠️ Tumor Detected (Malignant)")
    else:
        st.success("✅ Normal (Benign)")

st.markdown("---")
st.caption("Developed using Machine Learning & Streamlit")