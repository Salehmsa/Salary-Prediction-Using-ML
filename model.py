import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error


# ----------------------------------
# ✅ Evaluation Function
# ----------------------------------
def evaluate_model(y_test, y_pred):
    return {
        "R2": r2_score(y_test, y_pred),
        "MAE": mean_absolute_error(y_test, y_pred)
    }


# ----------------------------------
# ✅ Save Model
# ----------------------------------
def save_model(model):
    joblib.dump(model, "model.pkl")


# ----------------------------------
# ✅ Load Model
# ----------------------------------
def load_model():
    return joblib.load("model.pkl")


# ----------------------------------
# ✅ Train Models
# ----------------------------------
def train_models(df):

    X = df[['experience', 'skill']]
    y = df['salary']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Decision Tree": DecisionTreeRegressor(random_state=42)
    }

    results = {}

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

    # ✅ Save Model + Name
    save_model(best_model)
    joblib.dump(best_model_name, "model_name.pkl")

    return results


# ----------------------------------
# ✅ Run Training Once (Manual)
# ----------------------------------
if __name__ == "__main__":
    df = pd.read_csv("data/salary_data_250.csv")
    train_models(df)


