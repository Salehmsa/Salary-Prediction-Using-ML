import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

st.set_page_config(page_title='Salary Predictor Pro', page_icon='💼')
st.title('💼 Salary Prediction App - Pro')

# Generate data
np.random.seed(42)
n = 250
experience = np.random.uniform(0, 15, n)
skill = np.random.uniform(1, 10, n)
noise = np.random.normal(0, 7000, n)
salary = 25000 + 8000 * experience + 3000 * skill + noise

df = pd.DataFrame({'experience': experience,'skill': skill,'salary': salary})

X = df[['experience','skill']]
y = df['salary']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100),
    'Decision Tree': DecisionTreeRegressor()
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    results[name] = {'model': model,'R2': r2_score(y_test,preds),'MAE': mean_absolute_error(y_test,preds)}

st.sidebar.header('📥 Input')
exp = st.sidebar.slider('Experience',0.0,15.0,5.0)
sk = st.sidebar.slider('Skill',1.0,10.0,5.0)
model_choice = st.sidebar.selectbox('Model', list(models.keys()))

pred = results[model_choice]['model'].predict([[exp,sk]])[0]
st.success(f'💰 Salary: {pred:,.0f} SAR')

st.subheader('📊 Model Comparison')
for name,v in results.items():
    st.write(f"{name} → R²: {v['R2']:.3f} | MAE: {v['MAE']:,.0f}")
