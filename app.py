import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("models", exist_ok=True)

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Random Forest Classification",
    page_icon="🌲",
    layout="wide"
)

# ---------------- TITLE ----------------

st.title("🌲 Random Forest Classification App")
st.write("Heart Disease Prediction using Machine Learning")

# ---------------- LOAD DATA ----------------

@st.cache_data
def load_data():
    df = pd.read_csv("heart.csv")
    return df

df = load_data()

# ---------------- SHOW DATA ----------------

st.subheader("📊 Dataset")

if st.checkbox("Show Raw Data"):
    st.dataframe(df)

# ---------------- PREPROCESSING ----------------

X = df.drop("target", axis=1)
y = df["target"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------- SIDEBAR ----------------

st.sidebar.header("⚙ Hyperparameters")

n_estimators = st.sidebar.slider(
    "Number of Trees",
    50,
    500,
    100,
    step=50
)

max_depth = st.sidebar.slider(
    "Max Depth",
    2,
    20,
    10
)

min_samples_split = st.sidebar.slider(
    "Min Samples Split",
    2,
    10,
    2
)

criterion = st.sidebar.selectbox(
    "Criterion",
    ["gini", "entropy"]
)

# ---------------- MODEL ----------------

model = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    criterion=criterion,
    random_state=42
)

model.fit(X_train, y_train)

# ---------------- PREDICTION ----------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

# ---------------- RESULTS ----------------

st.subheader("✅ Model Performance")

st.metric(
    label="Accuracy",
    value=f"{accuracy * 100:.2f}%"
)

# ---------------- CONFUSION MATRIX ----------------

st.subheader("📌 Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(5, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    ax=ax
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

st.pyplot(fig)

# ---------------- CLASSIFICATION REPORT ----------------

st.subheader("📄 Classification Report")

report = classification_report(y_test, y_pred)

st.text(report)

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

fig2, ax2 = plt.subplots(figsize=(10, 5))

sns.barplot(
    x="Importance",
    y="Feature",
    data=importance,
    ax=ax2
)

st.pyplot(fig2)

# ---------------- SAVE MODEL ----------------

joblib.dump(model, "models/random_forest_model.pkl")

st.success("Model trained and saved successfully!")