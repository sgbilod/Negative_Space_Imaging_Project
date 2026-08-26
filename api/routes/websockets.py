"""WebSocket routes for real-time updates"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from .websocket_manager import manager
from .security.auth import get_current_user_ws
from typing import Dict

router = APIRouter()

@router.websocket("/ws/{channel}")
async def websocket_endpoint(
    websocket: WebSocket,
    channel: str,
    user: Dict = Depends(get_current_user_ws)
):
    if channel not in ["security", "jobs", "system"]:
        await websocket.close(code=4004)
        return

    try:
        await manager.connect(websocket, channel, user["sub"])

        try:
            while True:
                # Wait for messages from the client
                data = await websocket.receive_text()
                # Process message if needed

        except WebSocketDisconnect:
            manager.disconnect(websocket, channel, user["sub"])
    except Exception as e:
        import logging
        logging.error(f"WebSocket error: {e}")
        try:
            await websocket.close(code=1011)
        except Exception as close_err:
            logging.error(f"WebSocket close error: {close_err}")
