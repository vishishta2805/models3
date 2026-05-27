import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ---------------- CREATE MODELS FOLDER ----------------

os.makedirs("models", exist_ok=True)

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="AdaBoost Classification",
    page_icon="🚀",
    layout="wide"
)

# ---------------- TITLE ----------------

st.title("🚀 AdaBoost Classification App")
st.write("Heart Disease Prediction using AdaBoost")

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

# ---------------- FEATURES & TARGET ----------------

X = df.drop("target", axis=1)
y = df["target"]

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
    1
)

# ---------------- BASE MODEL ----------------

base_model = DecisionTreeClassifier(
    max_depth=max_depth
)

# ---------------- ADABOOST MODEL ----------------

model = AdaBoostClassifier(
    estimator=base_model,
    n_estimators=n_estimators,
    learning_rate=learning_rate,
    random_state=42
)

# ---------------- TRAIN MODEL ----------------

model.fit(X_train, y_train)

# ---------------- SAVE MODEL ----------------

joblib.dump(model, "models/adaboost_classifier.pkl")

# ---------------- PREDICTIONS ----------------

y_pred = model.predict(X_test)

# ---------------- ACCURACY ----------------

accuracy = accuracy_score(y_test, y_pred)

# ---------------- DISPLAY METRICS ----------------

st.subheader("📈 Model Performance")

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

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

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

fig2, ax2 = plt.subplots(figsize=(10, 6))

ax2.barh(
    importance["Feature"],
    importance["Importance"]
)

ax2.set_xlabel("Importance")
ax2.set_ylabel("Features")

st.pyplot(fig2)

# ---------------- SAMPLE PREDICTION ----------------

st.subheader("Predict Heart Disease")

sample_data = X.iloc[0:1]

if st.button("Predict Sample Patient"):

    prediction = model.predict(sample_data)

    if prediction[0] == 1:
        st.error("High chance of Heart Disease")
    else:
        st.success("Low chance of Heart Disease")

# ---------------- SUCCESS MESSAGE ----------------

st.success("AdaBoost Classifier trained and saved successfully!")