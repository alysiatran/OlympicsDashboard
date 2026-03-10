"""
ml_pipeline.py
==============
All ML training logic for the Olympics Dashboard.
Imported by app.py and cached with @st.cache_resource.
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix,
)
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier

RANDOM_STATE = 42

# ──────────────────────────────────────────────────────────────────────
# 2.1  DATA PREPARATION
# ──────────────────────────────────────────────────────────────────────

# Features kept (no target-leaking columns)
NUMERICAL_FEATURES = [
    "age",
    "height_cm",
    "weight_kg",
    "total_olympics_attended",
    "total_medals_won",        # career medal count — historical performance signal
    "country_total_gold",
    "country_total_medals",
    "country_best_rank",
    "country_first_participation",
    "year",
]

CATEGORICAL_FEATURES = [
    "gender",
    "games_type",
    "sport",
    "team_or_individual",
]

# Columns intentionally excluded and why:
# - gold_medals / silver_medals / bronze_medals: direct sub-components of total_medals_won;
#   including all three would cause multi-collinearity.
# - medal_points / medal / has_medal: the target or a direct derivation.
# - result_value / result_unit: post-event measurement → leakage.
# - is_record_holder: post-event designation → leakage.
# - athlete_id / athlete_name / date_of_birth / coach_name / notes: identifiers.
# - nationality / country_name: represented numerically by country_total_* features.
# - host_city / event: very high cardinality with limited added signal.
# Note: total_medals_won is included as a legitimate historical performance indicator.
#   Each athlete appears once; their career medal count is known pre-event context.


def prepare_data(df: pd.DataFrame):
    """Return X_train, X_test, y_train, y_test, preprocessor (fitted on train)."""
    feature_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_cols].copy()
    y = df["has_medal"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    return X_train_proc, X_test_proc, y_train.values, y_test.values, preprocessor


# ──────────────────────────────────────────────────────────────────────
# SHARED HELPERS
# ──────────────────────────────────────────────────────────────────────

def compute_metrics(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    return {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "auc_roc":   round(roc_auc_score(y_test, y_prob), 4),
        "fpr": fpr,
        "tpr": tpr,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "y_prob": y_prob,
    }


CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


# ──────────────────────────────────────────────────────────────────────
# 2.2  LOGISTIC REGRESSION BASELINE
# ──────────────────────────────────────────────────────────────────────

def train_logistic_regression(X_train, X_test, y_train, y_test):
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
        solver="lbfgs",
    )
    model.fit(X_train, y_train)
    metrics = compute_metrics(model, X_test, y_test)
    return model, metrics


# ──────────────────────────────────────────────────────────────────────
# 2.3  DECISION TREE / CART
# ──────────────────────────────────────────────────────────────────────

def train_decision_tree(X_train, X_test, y_train, y_test):
    param_grid = {
        "max_depth":        [3, 5, 7],
        "min_samples_leaf": [5, 10, 20],
    }
    base = DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE)
    gs = GridSearchCV(
        base, param_grid, cv=CV, scoring="f1", n_jobs=-1, refit=True
    )
    gs.fit(X_train, y_train)
    best_model = gs.best_estimator_
    metrics = compute_metrics(best_model, X_test, y_test)
    metrics["best_params"] = gs.best_params_
    metrics["cv_results"] = pd.DataFrame(gs.cv_results_)
    return best_model, metrics


# ──────────────────────────────────────────────────────────────────────
# 2.4  RANDOM FOREST
# ──────────────────────────────────────────────────────────────────────

def train_random_forest(X_train, X_test, y_train, y_test):
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth":    [5, 8],
    }
    base = RandomForestClassifier(
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    )
    gs = GridSearchCV(
        base, param_grid, cv=CV, scoring="f1", n_jobs=-1, refit=True
    )
    gs.fit(X_train, y_train)
    best_model = gs.best_estimator_
    metrics = compute_metrics(best_model, X_test, y_test)
    metrics["best_params"] = gs.best_params_
    metrics["feature_importances"] = best_model.feature_importances_
    return best_model, metrics


# ──────────────────────────────────────────────────────────────────────
# 2.5  LIGHTGBM (BOOSTED TREES)
# ──────────────────────────────────────────────────────────────────────

def train_lightgbm(X_train, X_test, y_train, y_test):
    # scale_pos_weight handles class imbalance for LightGBM
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    spw = neg / pos

    param_grid = {
        "n_estimators":  [100, 200],
        "max_depth":     [4, 6],
        "learning_rate": [0.05, 0.1],
    }
    base = LGBMClassifier(
        scale_pos_weight=spw,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    gs = GridSearchCV(
        base, param_grid, cv=CV, scoring="f1", n_jobs=-1, refit=True
    )
    gs.fit(X_train, y_train)
    best_model = gs.best_estimator_
    metrics = compute_metrics(best_model, X_test, y_test)
    metrics["best_params"] = gs.best_params_
    metrics["feature_importances"] = best_model.feature_importances_
    return best_model, metrics


# ──────────────────────────────────────────────────────────────────────
# 2.6  NEURAL NETWORK — MLP
# ──────────────────────────────────────────────────────────────────────

def train_mlp(X_train, X_test, y_train, y_test):
    """
    Multi-Layer Perceptron using sklearn's MLPClassifier.
    Architecture:
      - Input layer: matches feature dimensions
      - Hidden layer 1: 128 units, ReLU
      - Hidden layer 2: 128 units, ReLU
      - Output layer: 1 unit, sigmoid (logistic)
    Optimizer: Adam | Loss: binary cross-entropy (log_loss)
    Class imbalance handled via SMOTE oversampling on training data.
    """
    # SMOTE to balance classes before training the MLP
    smote = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    model = MLPClassifier(
        hidden_layer_sizes=(128, 128),
        activation="relu",
        solver="adam",
        alpha=1e-4,           # L2 regularisation
        batch_size=64,
        learning_rate_init=0.001,
        max_iter=300,
        random_state=RANDOM_STATE,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        verbose=False,
    )
    model.fit(X_res, y_res)
    metrics = compute_metrics(model, X_test, y_test)
    metrics["loss_curve"] = model.loss_curve_
    metrics["val_scores"] = model.validation_scores_ if hasattr(model, "validation_scores_") else None
    metrics["n_iter"] = model.n_iter_
    return model, metrics


# ──────────────────────────────────────────────────────────────────────
# FEATURE NAMES (for importance plots)
# ──────────────────────────────────────────────────────────────────────

def get_feature_names(preprocessor) -> list[str]:
    num_names = NUMERICAL_FEATURES
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    return num_names + cat_names


# ──────────────────────────────────────────────────────────────────────
# 3.1  SHAP ANALYSIS
# ──────────────────────────────────────────────────────────────────────

def compute_shap(model, X_test: np.ndarray, feature_names: list[str],
                 n_background: int = 100, n_explain: int = 300,
                 random_state: int = RANDOM_STATE):
    """
    Compute SHAP values for a tree-based model.
    Returns a dict with:
      - shap_values: np.ndarray (n_explain × n_features), class-1 SHAP values
      - X_sample:    np.ndarray, the subset of test rows explained
      - feat_df:     pd.DataFrame with feature_name / mean_abs_shap columns
      - waterfall_idx: index of the most-confident correct medal prediction
      - explainer:   the shap.TreeExplainer instance
    """
    import shap

    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(X_test), size=min(n_explain, len(X_test)), replace=False)
    X_sample = X_test[idx]

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_sample)

    # shap_values can be:
    #   list [class0, class1]          — older shap / some RF versions
    #   3-D ndarray (n, features, 2)   — newer shap with RF
    #   2-D ndarray (n, features)      — LGBM binary
    if isinstance(sv, list):
        shap_vals = sv[1]
    elif sv.ndim == 3:
        shap_vals = sv[:, :, 1]    # class-1 slice
    else:
        shap_vals = sv

    # Mean absolute SHAP per feature
    mean_abs = np.abs(shap_vals).mean(axis=0)
    feat_df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    feat_df = feat_df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    # Waterfall: pick test row with highest predicted medal probability that is
    # actually a medal winner (true positive) — interesting "edge case"
    y_sample_pred_prob = model.predict_proba(X_sample)[:, 1]
    y_sample_true      = model.predict(X_sample)          # predicted label
    tp_mask = y_sample_true == 1
    if tp_mask.any():
        waterfall_idx = int(np.argmax(np.where(tp_mask, y_sample_pred_prob, -1)))
    else:
        waterfall_idx = int(np.argmax(y_sample_pred_prob))

    return {
        "shap_values":    shap_vals,
        "X_sample":       X_sample,
        "feat_df":        feat_df,
        "waterfall_idx":  waterfall_idx,
        "explainer":      explainer,
        "expected_value": float(explainer.expected_value[1]) if hasattr(explainer.expected_value, "__len__") else float(explainer.expected_value),
    }
