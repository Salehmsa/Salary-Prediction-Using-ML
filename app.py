import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as stats
import os

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# ----------------------------------
# 🎨 UI Style
# ----------------------------------
st.set_page_config(page_title="Salary Predictor Pro", page_icon="💼")

st.title("💼 Salary Prediction App")
st.markdown("### Model Comparison & Best Model Selection")

# ----------------------------------

# 📊 Load Data
# ----------------------------------
df = pd.read_csv("data/salary_data_250.csv")

# ----------------------------------
# 📊 Overview
# ----------------------------------
st.subheader("📊 Dataset Overview")

# ✅ Basic Info
st.markdown("### 📌 Basic Info")

st.write(f"Number of rows: {df.shape[0]}")
st.write(f"Number of columns: {df.shape[1]}")
st.write("Column Names:", df.columns.tolist())

# ✅ Preview
with st.expander("🔍 Show Raw Data"):
    st.dataframe(df.head())

# ✅ Summary
st.subheader("📈 Dataset Summary")
st.dataframe(df.describe())


# ----------------------------------
# Split data
# ----------------------------------
X = df[['experience', 'skill']]
y = df['salary']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------------

# Models
# ----------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100),
    "Decision Tree": DecisionTreeRegressor()
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    results[name] = {
        "model": model,
        "R2": r2_score(y_test, preds),
        "MAE": mean_absolute_error(y_test, preds)
    }

# ----------------------------------
# Model Comparison
# ----------------------------------
st.subheader("🤖 Model Comparison")

for name, metrics in results.items():
    st.write(f"{name} → R²: {metrics['R2']:.3f} | MAE: {metrics['MAE']:.0f}")

# ----------------------------------
# Best Model
# ----------------------------------
best_model_name = max(results, key=lambda x: results[x]["R2"])
st.success(f"🏆 Best Model: {best_model_name}")

# ----------------------------------
# User Input
# ----------------------------------
st.sidebar.header("📥 User Input")

exp = st.sidebar.slider("Experience", 0.0, 15.0, 5.0)
sk = st.sidebar.slider("Skill", 1.0, 10.0, 5.0)

model_choice = st.sidebar.selectbox("Select Model", list(models.keys()))

selected_model = results[model_choice]["model"]
prediction = selected_model.predict([[exp, sk]])[0]

# ----------------------------------
# Prediction
# ----------------------------------
st.subheader("💰 Salary Prediction")
st.success(f"Predicted Salary: {prediction:,.0f} SAR")


# ----------------------------------
# 📊 Dashboard
# ----------------------------------
st.markdown("## 📊 Dashboard")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    ax.scatter(df['experience'], df['salary'], alpha=0.5)
    ax.set_title("Experience vs Salary")
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    ax.scatter(df['skill'], df['salary'], alpha=0.5)
    ax.set_title("Skill vs Salary")
    st.pyplot(fig)

# ----------------------------------
# 🎯 Feature Importance
# ----------------------------------
st.markdown("## 🎯 Feature Importance")

rf_model = results["Random Forest"]["model"]
importance = rf_model.feature_importances_

features = ['Experience', 'Skill']

fig, ax = plt.subplots()
ax.barh(features, importance, color='green')
ax.set_title("Feature Importance")
st.pyplot(fig)

# ----------------------------------
# 📂 Upload CSV
# ----------------------------------
st.markdown("## 📂 Upload Your Data")

uploaded_file = st.file_uploader(
    "Upload CSV with columns: experience, skill",
    type=["csv"]
)

if uploaded_file is not None:
    user_data = pd.read_csv(uploaded_file)

    st.write("📥 Uploaded Data Preview")
    st.dataframe(user_data.head())

    if {'experience','skill'}.issubset(user_data.columns):
        preds = selected_model.predict(user_data[['experience','skill']])
        user_data['Predicted Salary'] = preds

        st.success("✅ Predictions Completed")
        st.dataframe(user_data)
    else:
        st.error("❌ Must contain 'experience' and 'skill' columns")

# ----------------------------------
# 👤 Save Results
# ----------------------------------
st.markdown("## 👤 Save Your Prediction")

name = st.text_input("Enter your name")

if st.button("Save Result"):

    # ✅ Check the name
    if name == "":
        st.warning("⚠️ Please enter your name")
    else:
        # ✅ Create the file if it doesn't exist
        if not os.path.exists("results.csv"):
            with open("results.csv", "w") as f:
                f.write("name,experience,skill,prediction\n")

        # ✅ Save the data
        with open("results.csv", "a") as f:
            f.write(f"{name},{exp},{sk},{prediction}\n")

        # ✅ Success message
        st.success("✅ Saved successfully!")
    
# ----------------------------------
# Footer
# ----------------------------------
st.markdown("---")
st.write("Built by Saleh Mahbub 🚀")