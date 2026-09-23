from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import List
import json

router = APIRouter(tags=["Real-time Alerts"])

class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        self.active.remove(ws)

    async def broadcast(self, message: dict):
        dead = []
        for ws in self.active:
            try:
                await ws.send_text(json.dumps(message))
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.active.remove(ws)


manager = ConnectionManager()


@router.websocket("/ws/alerts")
async def alert_stream(websocket: WebSocket):
    """
    WebSocket endpoint — connect to receive real-time safety alerts.
    Broadcasts whenever a safety alert or anomaly is detected via /telemetry.

    Frontend usage:
        const ws = new WebSocket('ws://localhost:8000/ws/alerts');
        ws.onmessage = (e) => console.log(JSON.parse(e.data));
    """
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
