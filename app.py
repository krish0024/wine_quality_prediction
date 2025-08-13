import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score

# --- Page Title ---
st.title('Wine Quality Predictor (Regression)')

# --- Load dataset from online source ---
@st.cache_data
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    df = pd.read_csv(url, sep=';')
    return df

data = load_data()

# --- Load model and scaler if available ---
try:
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    model_loaded = True
except:
    st.warning("Model or scaler not found. Using a dummy model for demonstration.")
    from sklearn.linear_model import LinearRegression
    X = data.drop('quality', axis=1)
    y = data['quality']
    model = LinearRegression().fit(X, y)
    scaler = None
    model_loaded = False

# =================================================
# 1. DATA EXPLORATION SECTION
# =================================================
st.header("📊 Data Exploration")
st.subheader("Dataset Overview")
st.write("Shape of dataset:", data.shape)
st.write(data.dtypes)
st.subheader("Sample Data")
st.write(data.head())

# Interactive filtering
st.subheader("Interactive Data Filtering")
selected_quality = st.multiselect(
    "Select Quality Levels:", data['quality'].unique(), default=data['quality'].unique()
)
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
sns.scatterplot(
    x='fixed acidity', y='pH', data=data, hue='quality', palette='coolwarm', ax=ax
)
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
    features = {
        'fixed acidity': st.sidebar.slider('fixed acidity', 3.0, 16.0, 7.4),
        'volatile acidity': st.sidebar.slider('volatile acidity', 0.0, 1.5, 0.7),
        'citric acid': st.sidebar.slider('citric acid', 0.0, 1.0, 0.0),
        'residual sugar': st.sidebar.slider('residual sugar', 0.6, 65.0, 1.9),
        'chlorides': st.sidebar.slider('chlorides', 0.01, 0.2, 0.076),
        'free sulfur dioxide': st.sidebar.slider('free sulfur dioxide', 1.0, 72.0, 11.0),
        'total sulfur dioxide': st.sidebar.slider('total sulfur dioxide', 6.0, 289.0, 34.0),
        'density': st.sidebar.slider('density', 0.990, 1.005, 0.9978),
        'pH': st.sidebar.slider('pH', 2.7, 4.0, 3.51),
        'sulphates': st.sidebar.slider('sulphates', 0.2, 2.0, 0.56),
        'alcohol': st.sidebar.slider('alcohol', 8.0, 15.0, 9.4)
    }
    return pd.DataFrame(features, index=[0])

input_df = user_input_features()
st.subheader('User Input Features')
st.write(input_df)

# Scale and predict
if scaler:
    scaled = scaler.transform(input_df)
else:
    scaled = input_df.values
prediction = model.predict(scaled)
st.subheader('Predicted Quality (Score)')
st.write(float(prediction[0]))

# =================================================
# 4. MODEL PERFORMANCE SECTION
# =================================================
st.header("📉 Model Performance")
X_all = data.drop('quality', axis=1)
y_all = data['quality']
y_pred_all = model.predict(X_all)
mse = mean_squared_error(y_all, y_pred_all)
r2 = r2_score(y_all, y_pred_all)
st.write(f"**Mean Squared Error:** {mse:.2f}")
st.write(f"**R² Score:** {r2:.2f}")

st.subheader("Residuals Plot")
fig, ax = plt.subplots()
sns.scatterplot(x=y_all, y=y_all - y_pred_all, ax=ax)
ax.axhline(0, color='red', linestyle='--')
ax.set_xlabel("Actual Quality")
ax.set_ylabel("Residuals")
st.pyplot(fig)

st.subheader("Correlation Heatmap of Features")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

