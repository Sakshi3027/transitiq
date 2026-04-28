import json
import os
import pickle
from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    classification_report,
)
from xgboost import XGBClassifier, XGBRegressor
from loguru import logger

from backend.core.config import settings
from backend.ml.features import build_dataframe, FEATURE_COLUMNS
from backend.pipeline.simulator import generate_transit_event


MODEL_DIR = Path("backend/ml/saved_models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def generate_training_data(n_samples: int = 10000) -> pd.DataFrame:
    """Generate synthetic training data using our simulator."""
    logger.info(f"Generating {n_samples} training samples...")
    events = [generate_transit_event() for _ in range(n_samples)]
    df = build_dataframe(events)
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Delay distribution:\n{df['delay_minutes'].describe()}")
    logger.info(f"Delayed (>5 min) rate: {df['is_delayed'].mean():.2%}")
    return df


def train_classifier(df: pd.DataFrame) -> tuple:
    """Train XGBoost classifier: will this route be delayed > 5 minutes?"""
    X = df[FEATURE_COLUMNS]
    y = df["is_delayed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc": roc_auc_score(y_test, y_prob),
        "classification_report": classification_report(y_test, y_pred),
    }

    logger.info(f"Classifier ROC-AUC: {metrics['roc_auc']:.4f}")
    logger.info(f"\n{metrics['classification_report']}")

    return model, metrics, X_test, y_test


def train_regressor(df: pd.DataFrame) -> tuple:
    """Train XGBoost regressor: how many minutes will the delay be?"""
    X = df[FEATURE_COLUMNS]
    y = df["delay_minutes"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_pred = model.predict(X_test)

    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "r2": r2_score(y_test, y_pred),
    }

    logger.info(f"Regressor MAE: {metrics['mae']:.2f} min")
    logger.info(f"Regressor RMSE: {metrics['rmse']:.2f} min")
    logger.info(f"Regressor R²: {metrics['r2']:.4f}")

    return model, metrics, X_test, y_test


def run_training(n_samples: int = 10000):
    """Full training pipeline with MLflow tracking."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    logger.info("🚀 Starting TransitIQ model training...")

    df = generate_training_data(n_samples)

    with mlflow.start_run(run_name="transitiq-xgboost-v1") as run:
        # Log dataset info
        mlflow.log_param("n_samples", n_samples)
        mlflow.log_param("n_features", len(FEATURE_COLUMNS))
        mlflow.log_param("delayed_rate", round(df["is_delayed"].mean(), 4))

        # Train classifier
        clf_model, clf_metrics, _, _ = train_classifier(df)
        mlflow.log_metric("roc_auc", clf_metrics["roc_auc"])

        # Train regressor
        reg_model, reg_metrics, _, _ = train_regressor(df)
        mlflow.log_metric("mae", reg_metrics["mae"])
        mlflow.log_metric("rmse", reg_metrics["rmse"])
        mlflow.log_metric("r2", reg_metrics["r2"])

        # Save models locally
        clf_path = MODEL_DIR / "classifier.pkl"
        reg_path = MODEL_DIR / "regressor.pkl"

        with open(clf_path, "wb") as f:
            pickle.dump(clf_model, f)
        with open(reg_path, "wb") as f:
            pickle.dump(reg_model, f)

        # Log artifacts to MLflow
        mlflow.log_artifact(str(clf_path))
        mlflow.log_artifact(str(reg_path))

        # Log feature importance
        importance = dict(zip(
            FEATURE_COLUMNS,
            clf_model.feature_importances_.tolist()
        ))
        mlflow.log_dict(importance, "feature_importance.json")

        run_id = run.info.run_id
        logger.info(f"✅ Training complete! MLflow run_id: {run_id}")
        logger.info(f"📁 Models saved to {MODEL_DIR}")

        return {
            "run_id": run_id,
            "clf_model": clf_model,
            "reg_model": reg_model,
            "clf_metrics": clf_metrics,
            "reg_metrics": reg_metrics,
        }


if __name__ == "__main__":
    results = run_training(n_samples=10000)
    logger.info("🎉 Done! Models are ready.")