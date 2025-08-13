
import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.title('Wine Quality Predictor (Regression)')

# load model & scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

st.sidebar.header('Input features')
def user_input_features():
    fixed_acidity = st.sidebar.slider('fixed acidity', 3.0, 16.0, 7.4)
    volatile_acidity = st.sidebar.slider('volatile acidity', 0.0, 1.5, 0.7)
    citric_acid = st.sidebar.slider('citric acid', 0.0, 1.0, 0.0)
    residual_sugar = st.sidebar.slider('residual sugar', 0.6, 65.0, 1.9)
    chlorides = st.sidebar.slider('chlorides', 0.01, 0.2, 0.076)
    free_sulfur_dioxide = st.sidebar.slider('free sulfur dioxide', 1.0, 72.0, 11.0)
    total_sulfur_dioxide = st.sidebar.slider('total sulfur dioxide', 6.0, 289.0, 34.0)
    density = st.sidebar.slider('density', 0.990, 1.005, 0.9978)
    pH = st.sidebar.slider('pH', 2.7, 4.0, 3.51)
    sulphates = st.sidebar.slider('sulphates', 0.2, 2.0, 0.56)
    alcohol = st.sidebar.slider('alcohol', 8.0, 15.0, 9.4)
    data = {'fixed acidity': fixed_acidity,
            'volatile acidity': volatile_acidity,
            'citric acid': citric_acid,
            'residual sugar': residual_sugar,
            'chlorides': chlorides,
            'free sulfur dioxide': free_sulfur_dioxide,
            'total sulfur dioxide': total_sulfur_dioxide,
            'density': density,
            'pH': pH,
            'sulphates': sulphates,
            'alcohol': alcohol}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()
st.subheader('User input features')
st.write(input_df)

# scale and predict
scaled = scaler.transform(input_df)
prediction = model.predict(scaled)
st.subheader('Predicted quality (score)')
st.write(float(prediction[0]))
