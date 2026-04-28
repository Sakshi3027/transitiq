import random
import uuid
from datetime import datetime, timedelta
import holidays
from loguru import logger

# All the Chicago transit routes we'll simulate
ROUTES = [
    {"id": "RED", "name": "Red Line", "type": "subway", "stops": 33},
    {"id": "BLUE", "name": "Blue Line", "type": "subway", "stops": 27},
    {"id": "GREEN", "name": "Green Line", "type": "subway", "stops": 28},
    {"id": "BUS_22", "name": "Clark Bus", "type": "bus", "stops": 45},
    {"id": "BUS_36", "name": "Broadway Bus", "type": "bus", "stops": 38},
    {"id": "BUS_77", "name": "Belmont Bus", "type": "bus", "stops": 29},
    {"id": "ORANGE", "name": "Orange Line", "type": "subway", "stops": 13},
    {"id": "BROWN", "name": "Brown Line", "type": "subway", "stops": 18},
    {"id": "BUS_8", "name": "Halsted Bus", "type": "bus", "stops": 52},
    {"id": "PINK", "name": "Pink Line", "type": "subway", "stops": 22},
]

STOPS = [f"STOP_{i:03d}" for i in range(1, 101)]

WEATHER_CONDITIONS = [
    "clear", "cloudy", "light_rain", "heavy_rain",
    "snow", "fog", "thunderstorm", "windy"
]

WEATHER_DELAY_MULTIPLIER = {
    "clear": 1.0,
    "cloudy": 1.1,
    "light_rain": 1.3,
    "heavy_rain": 1.8,
    "snow": 2.5,
    "fog": 1.6,
    "thunderstorm": 2.2,
    "windy": 1.2,
}

US_HOLIDAYS = holidays.US()


def is_peak_hour(dt: datetime) -> bool:
    hour = dt.hour
    is_weekday = dt.weekday() < 5
    morning_peak = 7 <= hour <= 9
    evening_peak = 16 <= hour <= 19
    return is_weekday and (morning_peak or evening_peak)


def generate_delay(route: dict, weather: str, peak: bool, special_event: bool) -> float:
    """Simulate realistic delay in minutes based on conditions."""
    base_delay = random.gauss(mu=2.5, sigma=4.0)  # avg 2.5 min delay, std 4

    # Route type modifier — buses get delayed more than subways
    if route["type"] == "bus":
        base_delay *= 1.4

    # Weather modifier
    base_delay *= WEATHER_DELAY_MULTIPLIER[weather]

    # Peak hour doubles delay probability
    if peak:
        base_delay *= 1.6

    # Special event nearby adds big delay
    if special_event:
        base_delay += random.uniform(5, 20)

    # Inject rare severe delay spike (anomaly)
    if random.random() < 0.03:  # 3% chance of anomaly
        base_delay += random.uniform(25, 60)
        logger.warning(f"⚠️  Anomaly injected for route {route['id']}: {base_delay:.1f} min delay")

    return round(max(0, base_delay), 2)


def generate_weather() -> dict:
    condition = random.choice(WEATHER_CONDITIONS)
    temp_map = {
        "clear": random.uniform(60, 85),
        "cloudy": random.uniform(50, 70),
        "light_rain": random.uniform(45, 65),
        "heavy_rain": random.uniform(40, 60),
        "snow": random.uniform(20, 35),
        "fog": random.uniform(45, 60),
        "thunderstorm": random.uniform(55, 75),
        "windy": random.uniform(40, 65),
    }
    precip_map = {
        "clear": 0.0,
        "cloudy": 0.0,
        "light_rain": random.uniform(0.1, 0.5),
        "heavy_rain": random.uniform(0.5, 2.0),
        "snow": random.uniform(0.2, 1.5),
        "fog": 0.0,
        "thunderstorm": random.uniform(1.0, 3.0),
        "windy": 0.0,
    }
    return {
        "condition": condition,
        "temperature": round(temp_map[condition], 1),
        "precipitation": round(precip_map[condition], 2),
    }


def generate_transit_event() -> dict:
    """Generate one realistic transit event."""
    now = datetime.utcnow()
    route = random.choice(ROUTES)
    stop_id = random.choice(STOPS)
    weather = generate_weather()
    peak = is_peak_hour(now)
    is_holiday = now.date() in US_HOLIDAYS
    special_event = random.random() < 0.05  # 5% chance of nearby event

    delay = generate_delay(route, weather["condition"], peak, special_event)

    scheduled = now - timedelta(minutes=delay)

    return {
        "event_id": str(uuid.uuid4()),
        "route_id": route["id"],
        "route_name": route["name"],
        "route_type": route["type"],
        "stop_id": stop_id,
        "vehicle_id": f"VH_{random.randint(1000, 9999)}",
        "direction": random.choice(["northbound", "southbound", "eastbound", "westbound"]),
        "scheduled_arrival": scheduled.isoformat(),
        "actual_arrival": now.isoformat(),
        "delay_minutes": delay,
        "occupancy_level": random.choice(["low", "medium", "high", "crush_load"]),
        "weather_condition": weather["condition"],
        "temperature": weather["temperature"],
        "precipitation": weather["precipitation"],
        "is_peak_hour": peak,
        "is_holiday": is_holiday,
        "special_event_nearby": special_event,
        "timestamp": now.isoformat(),
    }