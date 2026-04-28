from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio
import json
from loguru import logger
from backend.core.config import settings
from backend.api.routes import router
from backend.api.websocket import ConnectionManager

manager = ConnectionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚇 TransitIQ API starting up...")
    # Start background task that pushes live events to WebSocket clients
    task = asyncio.create_task(live_event_broadcaster(manager))
    yield
    task.cancel()
    logger.info("TransitIQ API shut down.")


app = FastAPI(
    title="TransitIQ API",
    description="Real-time transit delay prediction platform",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/health")
async def health():
    return {"status": "healthy", "version": "1.0.0"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive, wait for client messages
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("Client disconnected from WebSocket")


async def live_event_broadcaster(manager: ConnectionManager):
    """
    Background task: generates live transit events every second
    and broadcasts predictions to all connected WebSocket clients.
    """
    from backend.pipeline.simulator import generate_transit_event
    from backend.ml.predictor import predictor

    logger.info("📡 Live event broadcaster started")

    while True:
        try:
            if manager.active_connections:
                event = generate_transit_event()
                prediction = predictor.predict(event)

                payload = {
                    "type": "prediction",
                    "event": event,
                    "prediction": prediction,
                }

                await manager.broadcast(json.dumps(payload))

            await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"Broadcaster error: {e}")
            await asyncio.sleep(2)