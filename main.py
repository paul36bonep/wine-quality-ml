#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Wine Quality Predictor")

# Input fields
st.header("Enter Chemical Attributes:")
features = []
fields = ['Fixed Acidity', 'Volatile Acidity', 'Citric Acid', 'Residual Sugar',
          'Chlorides', 'Free Sulfur Dioxide', 'Total Sulfur Dioxide', 'Density',
          'pH', 'Sulphates', 'Alcohol']

for field in fields:
    val = st.number_input(field, format="%.4f")
    features.append(val)

if st.button("Predict Quality"):
    x_input = scaler.transform([features])
    prediction = model.predict(x_input)[0]
    confidence = model.predict_proba(x_input)[0][prediction]

    result = "Good" if prediction == 1 else "Not Good"
    st.subheader(f"Prediction: {result}")
    st.write(f"Confidence Score: {confidence:.2f}")