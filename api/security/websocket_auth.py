"""WebSocket authentication utilities"""

import os
from fastapi import WebSocket, status
from jose import JWTError, jwt
from typing import Dict

# Use the same JWT settings as in the main API
JWT_SECRET = os.getenv("JWT_SECRET", "change-me-in-production")
JWT_ALGORITHM = "HS256"

async def get_current_user_ws(
    websocket: WebSocket
) -> Dict | None:
    try:
        token = websocket.query_params.get("token")
        if not token:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return None

        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        username = payload.get("sub")
        if username is None:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return None

        return payload
    except JWTError:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return None
    except Exception as e:
        print(f"WebSocket authentication error: {e}")
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        return None
