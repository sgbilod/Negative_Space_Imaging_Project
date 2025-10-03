"""
Database Connection Pooling for Negative Space Imaging Project
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.
"""

import logging
from typing import Dict, Any, Optional
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger("ConnectionPool")


class DatabasePool:
    """Database connection pool manager."""

    def __init__(
        self,
        connection_string: str,
        pool_size: int = 5,
        max_overflow: int = 10,
        timeout: int = 30
    ):
        self.connection_string = connection_string

        # Create engine with connection pooling
        self.engine = create_engine(
            connection_string,
            poolclass=QueuePool,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_timeout=timeout,
            pool_pre_ping=True
        )

        # Create session factory
        self.session_factory = scoped_session(
            sessionmaker(
                bind=self.engine,
                expire_on_commit=False
            )
        )

    @contextmanager
    def get_session(self):
        """Get a database session from the pool."""
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise
        finally:
            session.close()

    def get_pool_status(self) -> Dict[str, Any]:
        """Get current pool status."""
        try:
            return {
                'size': self.engine.pool.size(),
                'checkedin': self.engine.pool.checkedin(),
                'checkedout': self.engine.pool.checkedout(),
                'overflow': self.engine.pool.overflow(),
                'timeout': self.engine.pool.timeout()
            }
        except Exception as e:
            logger.error(f"Failed to get pool status: {e}")
            return {}

    def cleanup_pool(self):
        """Clean up idle connections."""
        try:
            self.engine.pool.dispose()
            logger.info("Pool cleaned up")

        except Exception as e:
            logger.error(f"Pool cleanup failed: {e}")

    def validate_connection(self, connection) -> bool:
        """Validate a database connection."""
        try:
            connection.scalar("SELECT 1")
            return True

        except SQLAlchemyError:
            return False

    def warmup_pool(self):
        """Pre-warm the connection pool."""
        try:
            logger.info("Warming up connection pool")

            with self.get_session() as session:
                session.execute("SELECT 1")

            logger.info("Pool warm-up complete")

        except Exception as e:
            logger.error(f"Pool warm-up failed: {e}")

    def reset_pool(self):
        """Reset the connection pool."""
        try:
            logger.info("Resetting connection pool")

            # Dispose of existing pool
            self.cleanup_pool()

            # Create new pool
            self.engine = create_engine(
                self.connection_string,
                poolclass=QueuePool,
                pool_size=self.engine.pool.size(),
                max_overflow=self.engine.pool.overflow(),
                pool_timeout=self.engine.pool.timeout(),
                pool_pre_ping=True
            )

            # Create new session factory
            self.session_factory = scoped_session(
                sessionmaker(
                    bind=self.engine,
                    expire_on_commit=False
                )
            )

            # Warm up new pool
            self.warmup_pool()

            logger.info("Pool reset complete")

        except Exception as e:
            logger.error(f"Pool reset failed: {e}")
            raise

    def health_check(self) -> bool:
        """Check pool health."""
        try:
            with self.get_session() as session:
                # Check basic connectivity
                session.execute("SELECT 1")

                # Get pool status
                status = self.get_pool_status()

                # Check for pool exhaustion
                if status['checkedout'] >= status['size']:
                    logger.warning("Pool near exhaustion")

                return True

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    @contextmanager
    def transaction(self):
        """Execute operations in a transaction."""
        with self.get_session() as session:
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise

    def execute_query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Execute a raw SQL query."""
        with self.get_session() as session:
            try:
                result = session.execute(query, params or {})
                return result.fetchall()

            except Exception as e:
                logger.error(f"Query execution failed: {e}")
                raise


# Example usage
if __name__ == "__main__":
    import os
    import time

    # Get connection string from environment
    connection_string = os.getenv(
        'DATABASE_URL',
        'postgresql://user:pass@localhost/dbname'
    )

    # Create pool
    pool = DatabasePool(
        connection_string,
        pool_size=5,
        max_overflow=10
    )

    # Warm up pool
    pool.warmup_pool()

    # Example transaction
    try:
        with pool.transaction() as session:
            # Your database operations here
            pass

    except Exception as e:
        logger.error(f"Transaction failed: {e}")

    # Monitor pool status
    while True:
        status = pool.get_pool_status()
        logger.info(f"Pool status: {status}")
        time.sleep(5)  # Check every 5 seconds

        # Perform health check
        if not pool.health_check():
            logger.error("Pool health check failed")
            pool.reset_pool()
