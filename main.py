import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Wine Quality Predictor")


st.header("Enter Chemical Attributes:")
fields = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
    'pH', 'sulphates', 'alcohol'
]

user_input = []
for field in fields:
    val = st.number_input(field, format="%.4f")
    user_input.append(val)


if st.button("Predict Quality"):

    input_df = pd.DataFrame([user_input], columns=fields)
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    confidence = model.predict_proba(input_scaled)[0][prediction]

    result = "Good " if prediction == 1 else "Not Good"
    st.subheader(f"Prediction: {result}")
    st.write(f"Confidence Score: **{confidence:.2f}**")
