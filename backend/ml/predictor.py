import pickle
import json
from pathlib import Path
from loguru import logger
from backend.ml.features import extract_features, FEATURE_COLUMNS
import pandas as pd

MODEL_DIR = Path("backend/ml/saved_models")


class TransitPredictor:
    def __init__(self):
        self.classifier = None
        self.regressor = None
        self.model_version = "1.0.0"
        self._load_models()

    def _load_models(self):
        clf_path = MODEL_DIR / "classifier.pkl"
        reg_path = MODEL_DIR / "regressor.pkl"

        if not clf_path.exists() or not reg_path.exists():
            logger.warning("⚠️  Models not found. Run training first.")
            return

        with open(clf_path, "rb") as f:
            self.classifier = pickle.load(f)
        with open(reg_path, "rb") as f:
            self.regressor = pickle.load(f)

        logger.info("✅ Models loaded successfully")

    def predict(self, event: dict) -> dict:
        if not self.classifier or not self.regressor:
            return {"error": "Models not loaded"}

        features = extract_features(event)
        df = pd.DataFrame([features])[FEATURE_COLUMNS]

        delay_prob = float(self.classifier.predict_proba(df)[0][1])
        predicted_delay = float(max(0, self.regressor.predict(df)[0]))
        is_delayed = delay_prob > 0.5
        is_anomaly = predicted_delay > 20 or delay_prob > 0.95

        severity = "normal"
        if predicted_delay > 30:
            severity = "critical"
        elif predicted_delay > 15:
            severity = "high"
        elif predicted_delay > 5:
            severity = "medium"

        return {
            "route_id": event.get("route_id"),
            "stop_id": event.get("stop_id"),
            "predicted_delay_minutes": round(predicted_delay, 2),
            "delay_probability": round(delay_prob, 4),
            "is_delayed": is_delayed,
            "is_anomaly": is_anomaly,
            "severity": severity,
            "model_version": self.model_version,
            "features": features,
        }


# Singleton — loaded once, reused across all requests
predictor = TransitPredictor()