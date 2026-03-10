"""
pretrain.py
===========
Run this once locally to train all models and save them to models.pkl.
The Streamlit app will load from this file instead of retraining every session.

    python pretrain.py
"""

import joblib
import pandas as pd
from ml_pipeline import (
    prepare_data,
    train_logistic_regression,
    train_decision_tree,
    train_random_forest,
    train_lightgbm,
    train_mlp,
    get_feature_names,
)

DATA_PATH = "olympics_athletes_dataset.csv"
OUTPUT_PATH = "models.pkl"

print("Loading data...")
raw = pd.read_csv(DATA_PATH)
raw["has_medal"] = raw["medal"] != "No Medal"
raw["medal_points"] = raw["medal"].map({"Gold": 3, "Silver": 2, "Bronze": 1, "No Medal": 0})

print("Preparing features...")
X_train, X_test, y_train, y_test, preprocessor = prepare_data(raw)
feature_names = get_feature_names(preprocessor)

print("Training Logistic Regression...")
lr_model, lr_m = train_logistic_regression(X_train, X_test, y_train, y_test)

print("Training Decision Tree (GridSearchCV)...")
dt_model, dt_m = train_decision_tree(X_train, X_test, y_train, y_test)

print("Training Random Forest (GridSearchCV)...")
rf_model, rf_m = train_random_forest(X_train, X_test, y_train, y_test)

print("Training LightGBM (GridSearchCV)...")
lgbm_model, lgbm_m = train_lightgbm(X_train, X_test, y_train, y_test)

print("Training MLP...")
mlp_model, mlp_m = train_mlp(X_train, X_test, y_train, y_test)

payload = {
    "X_train": X_train, "X_test": X_test,
    "y_train": y_train, "y_test": y_test,
    "preprocessor": preprocessor,
    "feature_names": feature_names,
    "lr":   (lr_model,   lr_m),
    "dt":   (dt_model,   dt_m),
    "rf":   (rf_model,   rf_m),
    "lgbm": (lgbm_model, lgbm_m),
    "mlp":  (mlp_model,  mlp_m),
}

print(f"Saving to {OUTPUT_PATH}...")
joblib.dump(payload, OUTPUT_PATH, compress=3)
print("Done! Commit models.pkl to your repo.")
