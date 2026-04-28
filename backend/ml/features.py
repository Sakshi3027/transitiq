import pandas as pd
import numpy as np
from datetime import datetime


# Weather mapped to numbers so XGBoost can read it
WEATHER_ENCODING = {
    "clear": 0,
    "cloudy": 1,
    "windy": 2,
    "fog": 3,
    "light_rain": 4,
    "heavy_rain": 5,
    "thunderstorm": 6,
    "snow": 7,
}

OCCUPANCY_ENCODING = {
    "low": 0,
    "medium": 1,
    "high": 2,
    "crush_load": 3,
}

DIRECTION_ENCODING = {
    "northbound": 0,
    "southbound": 1,
    "eastbound": 2,
    "westbound": 3,
}

ROUTE_TYPE_ENCODING = {
    "subway": 0,
    "bus": 1,
}


def extract_features(event: dict) -> dict:
    """
    Turn a raw transit event into ML features.
    This is the most important function in the whole project.
    """
    ts = datetime.fromisoformat(event["timestamp"])

    return {
        # Time features
        "hour": ts.hour,
        "day_of_week": ts.weekday(),       # 0=Monday, 6=Sunday
        "month": ts.month,
        "is_weekend": int(ts.weekday() >= 5),
        "is_peak_hour": int(event.get("is_peak_hour", False)),
        "is_holiday": int(event.get("is_holiday", False)),

        # Weather features
        "weather_code": WEATHER_ENCODING.get(event.get("weather_condition", "clear"), 0),
        "temperature": float(event.get("temperature", 65.0)),
        "precipitation": float(event.get("precipitation", 0.0)),

        # Route features
        "route_type": ROUTE_TYPE_ENCODING.get(event.get("route_type", "bus"), 1),
        "direction": DIRECTION_ENCODING.get(event.get("direction", "northbound"), 0),
        "occupancy": OCCUPANCY_ENCODING.get(event.get("occupancy_level", "low"), 0),

        # Event features
        "special_event_nearby": int(event.get("special_event_nearby", False)),
    }


def build_dataframe(events: list) -> pd.DataFrame:
    """Convert a list of raw events into a feature dataframe."""
    rows = []
    for event in events:
        features = extract_features(event)
        features["delay_minutes"] = float(event.get("delay_minutes", 0))
        features["is_delayed"] = int(features["delay_minutes"] > 5)
        rows.append(features)
    return pd.DataFrame(rows)


FEATURE_COLUMNS = [
    "hour", "day_of_week", "month", "is_weekend",
    "is_peak_hour", "is_holiday", "weather_code",
    "temperature", "precipitation", "route_type",
    "direction", "occupancy", "special_event_nearby",
]