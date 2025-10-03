"""Profiling middleware for FastAPI performance analysis"""

import time
from fastapi import Request
import logging

logger = logging.getLogger("profiling")

async def profiling_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"{request.method} {request.url.path} in {process_time:.4f}s")
    response.headers["X-Process-Time"] = str(process_time)
    return response
