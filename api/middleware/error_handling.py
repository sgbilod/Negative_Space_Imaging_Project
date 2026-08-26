"""API error handling middleware"""

from fastapi import Request, status
from fastapi.responses import JSONResponse
from typing import Union
from jose import JWTError
from sqlalchemy.exc import SQLAlchemyError
import traceback
import logging

logger = logging.getLogger(__name__)

class APIError(Exception):
    def __init__(
        self,
        status_code: int,
        message: str,
        details: Union[str, dict, None] = None
    ):
        self.status_code = status_code
        self.message = message
        self.details = details
        super().__init__(message)


async def error_handler(request: Request, call_next):
    try:
        return await call_next(request)

    except APIError as e:
        return JSONResponse(
            status_code=e.status_code,
            content={
                "error": e.message,
                "details": e.details,
                "path": request.url.path
            }
        )

    except JWTError:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                "error": "Invalid authentication credentials",
                "path": request.url.path
            }
        )

    except SQLAlchemyError as e:
        logger.error(f"Database error: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Database error occurred",
                "path": request.url.path
            }
        )

    except Exception as e:
        logger.error(f"Unhandled error: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "An unexpected error occurred",
                "path": request.url.path
            }
        )
