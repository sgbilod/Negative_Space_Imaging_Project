from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, Set, Optional
import json
import logging
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, Dict[str, WebSocket]] = {
            'security': {},
            'jobs': {},
            'system': {},
        }
        self.user_connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, channel: str, user_id: str):
        await websocket.accept()
        if channel not in self.active_connections:
            self.active_connections[channel] = {}
        self.active_connections[channel][user_id] = websocket

        if user_id not in self.user_connections:
            self.user_connections[user_id] = set()
        self.user_connections[user_id].add(websocket)

        logger.info(f"New connection: {channel} - {user_id}")

    def disconnect(self, websocket: WebSocket, channel: str, user_id: str):
        if channel in self.active_connections:
            self.active_connections[channel].pop(user_id, None)

        if user_id in self.user_connections:
            self.user_connections[user_id].remove(websocket)
            if not self.user_connections[user_id]:
                del self.user_connections[user_id]

        logger.info(f"Disconnected: {channel} - {user_id}")

    async def broadcast_to_channel(self, channel: str, message: dict):
        if channel not in self.active_connections:
            return

        encoded_message = json.dumps({
            'timestamp': datetime.utcnow().isoformat(),
            'channel': channel,
            'data': message
        })

        disconnected = []
        for user_id, websocket in self.active_connections[channel].items():
            try:
                await websocket.send_text(encoded_message)
            except WebSocketDisconnect:
                disconnected.append((user_id, websocket))
            except Exception as e:
                logger.error(f"Error sending message to {user_id}: {e}")
                disconnected.append((user_id, websocket))

        # Clean up disconnected clients
        for user_id, websocket in disconnected:
            self.disconnect(websocket, channel, user_id)

    async def send_to_user(self, user_id: str, message: dict):
        if user_id not in self.user_connections:
            return

        encoded_message = json.dumps({
            'timestamp': datetime.utcnow().isoformat(),
            'type': 'user_message',
            'data': message
        })

        disconnected = set()
        for websocket in self.user_connections[user_id]:
            try:
                await websocket.send_text(encoded_message)
            except WebSocketDisconnect:
                disconnected.add(websocket)
            except Exception as e:
                logger.error(f"Error sending message to user {user_id}: {e}")
                disconnected.add(websocket)

        # Clean up disconnected clients
        for websocket in disconnected:
            for channel in self.active_connections.values():
                for conn_user_id, conn_ws in channel.items():
                    if conn_ws == websocket:
                        self.disconnect(websocket, channel, conn_user_id)

    async def broadcast_system_message(self, message: dict):
        await self.broadcast_to_channel('system', message)

    async def broadcast_security_event(self, event: dict):
        await self.broadcast_to_channel('security', event)

    async def broadcast_job_update(self, job: dict):
        await self.broadcast_to_channel('jobs', job)


# Create a global connection manager instance
manager = ConnectionManager()


class WebSocketNotifier:
    def __init__(self):
        self.connection_manager = manager

    async def notify_security_event(self, event: dict):
        await self.connection_manager.broadcast_security_event(event)

    async def notify_job_update(self, job: dict):
        await self.connection_manager.broadcast_job_update(job)

    async def notify_system_update(self, message: dict):
        await self.connection_manager.broadcast_system_message(message)

    async def notify_user(self, user_id: str, message: dict):
        await self.connection_manager.send_to_user(user_id, message)


# Create a global notifier instance
notifier = WebSocketNotifier()
