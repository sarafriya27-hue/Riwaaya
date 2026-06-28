"""
Model training for the Riwaaya dashboard. Everything here is wrapped in
st.cache_resource so the grid searches only run once per session, not on
every interaction.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, roc_curve, confusion_matrix,
                              r2_score, mean_squared_error, mean_absolute_error)

from riwaaya_lib.data_loader import build_feature_matrix

CLASSIFIER_GRIDS = {
    "KNN": (KNeighborsClassifier(), {
        "n_neighbors": [5, 7, 9, 11, 15, 21],
        "weights": ["uniform", "distance"],
        "p": [1, 2],
    }),
    "Decision Tree": (DecisionTreeClassifier(random_state=42), {
        "max_depth": [3, 5, 7, 10, None],
        "min_samples_split": [2, 5, 10],
        "criterion": ["gini", "entropy"],
    }),
    "Random Forest": (RandomForestClassifier(random_state=42), {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, 15, None],
        "min_samples_split": [2, 5],
        "max_features": ["sqrt", "log2"],
    }),
    "Gradient Boosting": (GradientBoostingClassifier(random_state=42), {
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [2, 3, 4],
    }),
}


@st.cache_resource(show_spinner="Training and tuning classification models (runs once per session)...")
def train_classification_models(df: pd.DataFrame):
    X = build_feature_matrix(df)
    y = df["target_membership_interest"]
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = {}
    for name, (estimator, grid) in CLASSIFIER_GRIDS.items():
        gs = GridSearchCV(estimator, grid, cv=cv, scoring="roc_auc", n_jobs=-1)
        gs.fit(X_train_scaled, y_train)
        best_model = gs.best_estimator_

        train_pred = best_model.predict(X_train_scaled)
        test_pred = best_model.predict(X_test_scaled)
        test_proba = best_model.predict_proba(X_test_scaled)[:, 1]

        cm = confusion_matrix(y_test, test_pred)
        tn, fp, fn, tp = cm.ravel()
        total_errors = fp + fn
        fpr, tpr, _ = roc_curve(y_test, test_proba)

        results[name] = {
            "model": best_model,
            "best_params": gs.best_params_,
            "cv_best_score": gs.best_score_,
            "train_accuracy": accuracy_score(y_train, train_pred),
            "test_accuracy": accuracy_score(y_test, test_pred),
            "precision": precision_score(y_test, test_pred),
            "recall": recall_score(y_test, test_pred),
            "f1": f1_score(y_test, test_pred),
            "roc_auc": roc_auc_score(y_test, test_proba),
            "confusion_matrix": cm,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            "fp_pct_of_errors": (fp / total_errors * 100) if total_errors else 0,
            "fn_pct_of_errors": (fn / total_errors * 100) if total_errors else 0,
            "fp_pct_of_test": fp / len(y_test) * 100,
            "fn_pct_of_test": fn / len(y_test) * 100,
            "fpr": fpr, "tpr": tpr,
            "test_proba": test_proba,
        }

    comparison = pd.DataFrame({k: {
        "Train Accuracy": v["train_accuracy"], "Test Accuracy": v["test_accuracy"],
        "Precision": v["precision"], "Recall": v["recall"], "F1 Score": v["f1"],
        "ROC-AUC": v["roc_auc"],
    } for k, v in results.items()}).T.sort_values("ROC-AUC", ascending=False)

    best_name = comparison.index[0]

    # Feature importance from the best tree-based model, falls back to
    # Random Forest if the top model has no feature_importances_ (e.g. KNN)
    importance_source = best_name if hasattr(results[best_name]["model"], "feature_importances_") else "Random Forest"
    importances = pd.Series(
        results[importance_source]["model"].feature_importances_, index=feature_names
    ).sort_values(ascending=False)

    return {
        "results": results,
        "comparison": comparison,
        "best_model_name": best_name,
        "feature_importances": importances,
        "X_test": X_test, "y_test": y_test,
        "feature_names": feature_names,
        "scaler": scaler,
        "all_test_probas": {name: r["test_proba"] for name, r in results.items()},
    }


@st.cache_resource(show_spinner="Training regression models...")
def train_regression_models(df: pd.DataFrame, target_col: str, extra_drop_cols=None):
    extra_drop_cols = extra_drop_cols or []
    drop_cols = set(extra_drop_cols) | {target_col}
    X = build_feature_matrix(df, drop_cols=drop_cols)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.05, max_depth=3, random_state=42),
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        pred = model.predict(X_test_s)
        results[name] = {
            "model": model,
            "r2": r2_score(y_test, pred),
            "rmse": mean_squared_error(y_test, pred) ** 0.5,
            "mae": mean_absolute_error(y_test, pred),
            "y_test": y_test, "pred": pred,
        }
    comparison = pd.DataFrame({k: {"R2": v["r2"], "RMSE": v["rmse"], "MAE": v["mae"]}
                               for k, v in results.items()}).T.sort_values("R2", ascending=False)
    return {"results": results, "comparison": comparison, "best_model_name": comparison.index[0]}
