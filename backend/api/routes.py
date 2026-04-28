from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from backend.ml.predictor import predictor
from backend.pipeline.simulator import generate_transit_event, ROUTES

router = APIRouter()


class PredictionRequest(BaseModel):
    route_id: str
    weather_condition: str = "clear"
    temperature: float = 65.0
    precipitation: float = 0.0
    is_peak_hour: bool = False
    is_holiday: bool = False
    special_event_nearby: bool = False
    occupancy_level: str = "medium"
    direction: str = "northbound"
    route_type: str = "bus"
    stop_id: str = "STOP_001"


@router.get("/routes")
async def get_routes():
    """List all transit routes."""
    return {"routes": ROUTES}


@router.post("/predict")
async def predict_delay(request: PredictionRequest):
    """Predict delay for a given route and conditions."""
    from datetime import datetime
    event = {
        **request.model_dump(),
        "timestamp": datetime.utcnow().isoformat(),
        "event_id": "manual-request",
        "vehicle_id": "N/A",
        "scheduled_arrival": datetime.utcnow().isoformat(),
        "actual_arrival": datetime.utcnow().isoformat(),
        "delay_minutes": 0,
    }
    result = predictor.predict(event)
    return result


@router.get("/simulate")
async def simulate_event():
    """Generate and predict a single random transit event."""
    event = generate_transit_event()
    prediction = predictor.predict(event)
    return {"event": event, "prediction": prediction}


@router.get("/simulate/batch")
async def simulate_batch(count: int = 10):
    """Generate multiple predictions at once."""
    results = []
    for _ in range(min(count, 50)):
        event = generate_transit_event()
        prediction = predictor.predict(event)
        results.append({"event": event, "prediction": prediction})
    return {"count": len(results), "results": results}


@router.get("/stats")
async def get_stats():
    """Get live stats across simulated events."""
    events = [generate_transit_event() for _ in range(100)]
    predictions = [predictor.predict(e) for e in events]

    delays = [p["predicted_delay_minutes"] for p in predictions]
    anomalies = sum(1 for p in predictions if p["is_anomaly"])

    by_severity = {}
    for p in predictions:
        sev = p["severity"]
        by_severity[sev] = by_severity.get(sev, 0) + 1

    return {
        "total_events": len(predictions),
        "avg_delay_minutes": round(sum(delays) / len(delays), 2),
        "max_delay_minutes": round(max(delays), 2),
        "anomaly_count": anomalies,
        "anomaly_rate": round(anomalies / len(predictions), 3),
        "severity_breakdown": by_severity,
    }