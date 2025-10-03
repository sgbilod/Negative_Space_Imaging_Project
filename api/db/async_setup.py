"""Async database and cache setup for performance optimization"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from .production import Config
import aioredis

# Async database engine
engine = create_async_engine(Config.DATABASE_URL, echo=False, future=True)
AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# Redis cache
async def get_redis():
    return await aioredis.from_url(Config.REDIS_URL, encoding="utf-8", decode_responses=True)
