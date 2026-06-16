
import joblib

import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error


# ✅ 1) Evaluation function (here you add it)
def evaluate_model(y_test, y_pred):
    return {
        "R2": r2_score(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred)
    }


# ✅ 2) Main function
def train_models(df):

    X = df[['experience', 'skill']]
    y = df['salary']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

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
            **evaluate_model(y_test, preds)
        }

    best_model_name = max(results, key=lambda x: results[x]["R2"])
    best_model = results[best_model_name]["model"]

    save_model(best_model)

    return results

# ✅ Save the model 
def save_model(model):
    joblib.dump(model, "model.pkl")


# ✅ Download the model
def load_model():
    return joblib.load("model.pkl")

