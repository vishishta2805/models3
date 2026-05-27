import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ---------------- CREATE MODELS FOLDER ----------------

os.makedirs("models", exist_ok=True)

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="AdaBoost Regression",
    page_icon="🚀",
    layout="wide"
)

# ---------------- TITLE ----------------

st.title("🚀 AdaBoost Regression App")
st.write("California Housing Price Prediction")

# ---------------- LOAD DATA ----------------

@st.cache_data
def load_data():
    df = pd.read_csv("housing 2.csv")
    return df

df = load_data()

# ---------------- SHOW DATA ----------------

st.subheader("📊 Dataset")

if st.checkbox("Show Raw Data"):
    st.dataframe(df)

# ---------------- HANDLE MISSING VALUES ----------------

df = df.dropna()

# ---------------- FEATURES & TARGET ----------------

X = df.drop("median_house_value", axis=1)

X = pd.get_dummies(X, drop_first=True)

y = df["median_house_value"]

# ---------------- TRAIN TEST SPLIT ----------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------- SIDEBAR ----------------

st.sidebar.header("⚙ Hyperparameters")

n_estimators = st.sidebar.slider(
    "Number of Estimators",
    10,
    500,
    100,
    step=10
)

learning_rate = st.sidebar.slider(
    "Learning Rate",
    0.01,
    2.0,
    1.0
)

max_depth = st.sidebar.slider(
    "Base Tree Depth",
    1,
    10,
    3
)

# ---------------- BASE MODEL ----------------

base_model = DecisionTreeRegressor(
    max_depth=max_depth
)

# ---------------- ADABOOST MODEL ----------------

model = AdaBoostRegressor(
    estimator=base_model,
    n_estimators=n_estimators,
    learning_rate=learning_rate,
    random_state=42
)

# ---------------- TRAIN MODEL ----------------

model.fit(X_train, y_train)

# ---------------- SAVE MODEL ----------------

joblib.dump(model, "models/adaboost_regressor.pkl")

# ---------------- PREDICTIONS ----------------

y_pred = model.predict(X_test)

# ---------------- METRICS ----------------

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# ---------------- DISPLAY METRICS ----------------

st.subheader("📈 Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric("MAE", f"{mae:.2f}")

with col2:
    st.metric("R² Score", f"{r2:.4f}")

col3, col4 = st.columns(2)

with col3:
    st.metric("MSE", f"{mse:.2f}")

with col4:
    st.metric("RMSE", f"{rmse:.2f}")

# ---------------- ACTUAL VS PREDICTED ----------------

st.subheader("📌 Actual vs Predicted")

fig, ax = plt.subplots(figsize=(8, 5))

ax.scatter(y_test, y_pred)

ax.set_xlabel("Actual Values")
ax.set_ylabel("Predicted Values")
ax.set_title("Actual vs Predicted")

st.pyplot(fig)

# ---------------- FEATURE IMPORTANCE ----------------

st.subheader("⭐ Feature Importance")

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

fig2, ax2 = plt.subplots(figsize=(10, 6))

ax2.barh(
    importance["Feature"],
    importance["Importance"]
)

ax2.set_xlabel("Importance")
ax2.set_ylabel("Features")

st.pyplot(fig2)

# ---------------- SAMPLE PREDICTION ----------------

st.subheader("🏠 Predict House Price")

sample_data = X.iloc[0:1]

if st.button("Predict Sample House Price"):

    prediction = model.predict(sample_data)

    st.success(
        f"Predicted House Price: ${prediction[0]:,.2f}"
    )

# ---------------- SUCCESS MESSAGE ----------------

st.success("AdaBoost Regressor trained and saved successfully!")