import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os

from model import load_model, train_models

# ----------------------------------
# 🎨 Page Config
# ----------------------------------
st.set_page_config(page_title="Salary Predictor Pro", page_icon="💼")

st.title("💼 Salary Prediction App")
st.markdown("### Production Machine Learning Application")

# ----------------------------------
# 📊 Load Data
# ----------------------------------
df = pd.read_csv("data/salary_data_250.csv")

# ----------------------------------
# 📊 Dataset Overview
# ----------------------------------
st.subheader("📊 Dataset Overview")

st.markdown("### 📌 Basic Info")
st.write(f"Rows: {df.shape[0]}")
st.write(f"Columns: {df.shape[1]}")
st.write("Columns:", df.columns.tolist())

with st.expander("🔍 Show Raw Data"):
    st.dataframe(df.head())

st.subheader("📈 Dataset Summary")
st.dataframe(df.describe())

# ----------------------------------
# ✅ Load or Train Model
# ----------------------------------
@st.cache_resource
def get_model():
    if os.path.exists("model.pkl") and os.path.exists("model_name.pkl"):
        return load_model()
    else:
        st.warning("⚠️ Model not found. Training...")
        results = train_models(df)
        best = max(results, key=lambda x: results[x]["R2"])
        return results[best]["model"]

model = get_model()

# ✅ Load model name
if os.path.exists("model_name.pkl"):
    model_name = joblib.load("model_name.pkl")
else:
    model_name = "Unknown"

# ----------------------------------
# 📥 User Input
# ----------------------------------
st.sidebar.header("📥 User Input")

exp = st.sidebar.slider("Experience", 0.0, 15.0, 5.0)
sk = st.sidebar.slider("Skill", 1.0, 10.0, 5.0)

# ----------------------------------
# 💰 Prediction
# ----------------------------------
prediction = model.predict([[exp, sk]])[0]

st.success(f"🏆 Best Model: {model_name}")

st.subheader("💰 Salary Prediction")
st.success(f"Predicted Salary: {prediction:,.0f} SAR")

st.caption("⚡ Using pretrained model (model.pkl)")

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

    if {'experience', 'skill'}.issubset(user_data.columns):
        preds = model.predict(user_data[['experience', 'skill']])
        user_data["Predicted Salary"] = preds

        st.success("✅ Predictions Completed")
        st.dataframe(user_data)
    else:
        st.error("❌ CSV must contain 'experience' and 'skill' columns")

# ----------------------------------
# 👤 Save Results
# ----------------------------------
st.markdown("## 👤 Save Your Prediction")

name = st.text_input("Enter your name")

if st.button("Save Result"):

    if name == "":
        st.warning("⚠️ Please enter your name")
    else:
        if not os.path.exists("results.csv"):
            with open("results.csv", "w") as f:
                f.write("name,experience,skill,prediction\n")

        with open("results.csv", "a") as f:
            f.write(f"{name},{exp},{sk},{prediction}\n")

        st.success("✅ Saved successfully!")

# ----------------------------------
# Footer
# ----------------------------------
st.markdown("---")
st.write("Built by Saleh Mahbub 🚀")