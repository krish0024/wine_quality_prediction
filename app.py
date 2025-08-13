import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Page Title ---
st.title('Wine Quality Predictor (Regression)')

# --- Load model, scaler, and dataset ---
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
data = pd.read_csv('winequality-red.csv')  # <-- your dataset file

# =================================================
# 1. DATA EXPLORATION SECTION
# =================================================
st.header("📊 Data Exploration")

# Dataset overview
st.subheader("Dataset Overview")
st.write("Shape of dataset:", data.shape)
st.write("Columns and data types:")
st.write(data.dtypes)

# Show sample data
st.subheader("Sample Data")
st.write(data.head())

# Interactive filtering
st.subheader("Interactive Data Filtering")
selected_quality = st.multiselect("Select Quality Levels:", data['quality'].unique(), default=data['quality'].unique())
filtered_data = data[data['quality'].isin(selected_quality)]
st.write(f"Filtered dataset shape: {filtered_data.shape}")
st.write(filtered_data)

# =================================================
# 2. VISUALISATION SECTION
# =================================================
st.header("📈 Data Visualisation")

# Histogram
st.subheader("Alcohol Distribution")
fig, ax = plt.subplots()
sns.histplot(data['alcohol'], bins=20, kde=True, ax=ax)
st.pyplot(fig)

# Scatter plot
st.subheader("Fixed Acidity vs pH")
fig, ax = plt.subplots()
sns.scatterplot(x='fixed acidity', y='pH', data=data, hue='quality', palette='coolwarm', ax=ax)
st.pyplot(fig)

# Correlation heatmap
st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# =================================================
# 3. MODEL PREDICTION SECTION
# =================================================
st.header("🤖 Model Prediction")

st.sidebar.header('Input Features')
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
    return pd.DataFrame({
        'fixed acidity': [fixed_acidity],
        'volatile acidity': [volatile_acidity],
        'citric acid': [citric_acid],
        'residual sugar': [residual_sugar],
        'chlorides': [chlorides],
        'free sulfur dioxide': [free_sulfur_dioxide],
        'total sulfur dioxide': [total_sulfur_dioxide],
        'density': [density],
        'pH': [pH],
        'sulphates': [sulphates],
        'alcohol': [alcohol]
    })

input_df = user_input_features()
st.subheader('User Input Features')
st.write(input_df)

# Scale and predict
scaled = scaler.transform(input_df)
prediction = model.predict(scaled)
st.subheader('Predicted Quality (Score)')
st.write(float(prediction[0]))

# =================================================
# 4. MODEL PERFORMANCE SECTION
# =================================================
st.header("📉 Model Performance")

# Load test results (adjust as per your saved data)
y_test = np.load('y_test.npy')
y_pred = np.load('y_pred.npy')

from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix
import numpy as np

# Regression metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
st.write(f"**Mean Squared Error:** {mse:.2f}")
st.write(f"**R² Score:** {r2:.2f}")

# Confusion matrix (if classification model)
if len(np.unique(y_test)) < 20:  # assume classification
    cm = confusion_matrix(y_test, np.round(y_pred))
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.subheader("Confusion Matrix")
    st.pyplot(fig)

# Model comparison placeholder
st.subheader("Model Comparison")
st.write("Comparison results between multiple models will be shown here (if available).")
