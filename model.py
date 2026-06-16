import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error


# ----------------------------------
# ✅ Evaluate Model
# ----------------------------------
def evaluate_model(y_test, y_pred):
    """
    Evaluate model performance using R² and MAE
    """
    return {
        "R2": r2_score(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred)
    }


# ----------------------------------
# ✅ Save Model
# ----------------------------------
def save_model(model, model_name):
    """
    Save trained model and its name
    """
    joblib.dump(model, "model.pkl")
    joblib.dump(model_name, "model_name.pkl")


# ----------------------------------
# ✅ Load Model
# ----------------------------------
def load_model():
    """
    Load trained model from file
    """
    return joblib.load("model.pkl")


# ----------------------------------
# ✅ Train Models
# ----------------------------------
def train_models(df):
    """
    Train multiple models, evaluate them,
    choose the best, and save it
    """

    # ✅ Features & Target
    X = df[['experience', 'skill']]
    y = df['salary']

    # ✅ Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ✅ Define Models
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Decision Tree": DecisionTreeRegressor(random_state=42)
    }

    results = {}

    # ✅ Train & Evaluate
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        results[name] = {
            "model": model,
            **evaluate_model(y_test, preds)
        }

    # ✅ Select Best Model
    best_model_name = max(results, key=lambda x: results[x]["R2"])
    best_model = results[best_model_name]["model"]

    # ✅ Save Best Model + Name
    save_model(best_model, best_model_name)

    return results


# ----------------------------------
# ✅ Run Training Manually (Optional)
# ----------------------------------
if __name__ == "__main__":
    df = pd.read_csv("data/salary_data_250.csv")
    train_models(df)
