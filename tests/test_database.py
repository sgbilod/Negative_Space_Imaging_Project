"""
Database Test Suite for Negative Space Imaging Project
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.
"""

import os
import pytest
import logging
from pathlib import Path
from typing import Generator
from datetime import datetime, timedelta

import sqlalchemy as sa
from sqlalchemy.orm import Session

from database.schema import Base, User, Image, ProcessingJob
from database.connection_pool import DatabasePool
from database.migrations import MigrationManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger("DatabaseTests")

# Test configuration
TEST_DB_URL = os.getenv(
    'TEST_DATABASE_URL',
    'postgresql://test:test@localhost/test_db'
)


@pytest.fixture(scope="session")
def db_pool() -> DatabasePool:
    """Create test database pool."""
    pool = DatabasePool(TEST_DB_URL, pool_size=2)
    pool.warmup_pool()
    return pool


@pytest.fixture(scope="function")
def db_session(db_pool: DatabasePool) -> Generator[Session, None, None]:
    """Create test database session."""
    with db_pool.get_session() as session:
        yield session


@pytest.fixture(scope="session", autouse=True)
def setup_test_db(db_pool: DatabasePool):
    """Set up test database."""
    engine = db_pool.engine

    # Drop all tables
    Base.metadata.drop_all(engine)

    # Create all tables
    Base.metadata.create_all(engine)

    yield

    # Clean up
    Base.metadata.drop_all(engine)


def test_user_crud(db_session: Session):
    """Test user CRUD operations."""
    # Create
    user = User(
        username="testuser",
        email="test@example.com",
        password_hash="hashed_password",
        role="user"
    )
    db_session.add(user)
    db_session.commit()

    # Read
    saved_user = db_session.query(User).filter_by(
        username="testuser"
    ).first()
    assert saved_user is not None
    assert saved_user.email == "test@example.com"

    # Update
    saved_user.role = "admin"
    db_session.commit()

    updated_user = db_session.query(User).filter_by(
        username="testuser"
    ).first()
    assert updated_user.role == "admin"

    # Delete
    db_session.delete(saved_user)
    db_session.commit()

    deleted_user = db_session.query(User).filter_by(
        username="testuser"
    ).first()
    assert deleted_user is None


def test_image_processing_flow(db_session: Session):
    """Test image processing workflow."""
    # Create user
    user = User(
        username="imageuser",
        email="image@example.com",
        password_hash="hashed_password",
        role="user"
    )
    db_session.add(user)

    # Create image
    image = Image(
        owner=user,
        filename="test.jpg",
        file_hash="hash123",
        file_size=1000,
        mime_type="image/jpeg",
        width=1920,
        height=1080
    )
    db_session.add(image)

    # Create processing job
    job = ProcessingJob(
        image=image,
        status="pending",
        algorithm="test_algo",
        parameters='{"param1": "value1"}'
    )
    db_session.add(job)
    db_session.commit()

    # Verify relationships
    saved_image = db_session.query(Image).first()
    assert saved_image.owner.username == "imageuser"
    assert len(saved_image.processing_jobs) == 1

    saved_job = saved_image.processing_jobs[0]
    assert saved_job.status == "pending"
    assert saved_job.algorithm == "test_algo"


def test_connection_pool(db_pool: DatabasePool):
    """Test connection pool functionality."""
    # Get initial status
    initial_status = db_pool.get_pool_status()
    assert isinstance(initial_status, dict)
    assert 'size' in initial_status

    # Test multiple connections
    sessions = []
    for _ in range(3):
        with db_pool.get_session() as session:
            sessions.append(session)
            session.execute("SELECT 1")

    # Check pool status after use
    final_status = db_pool.get_pool_status()
    assert final_status['checkedin'] >= initial_status['checkedin']


def test_migrations(db_pool: DatabasePool):
    """Test database migrations."""
    manager = MigrationManager(TEST_DB_URL)

    # Create test migration
    revision = manager.create_migration("test_migration")
    assert revision is not None

    # Get current revision
    current = manager.get_current_revision()
    assert current is not None

    # Verify database
    assert manager.verify_database()

    # Get history
    history = manager.get_history()
    assert len(history) > 0


def test_transaction_rollback(db_pool: DatabasePool):
    """Test transaction rollback."""
    with pytest.raises(Exception):
        with db_pool.transaction() as session:
            # Create user
            user = User(
                username="rollbackuser",
                email="rollback@example.com",
                password_hash="hashed_password",
                role="user"
            )
            session.add(user)

            # Raise exception to trigger rollback
            raise Exception("Test rollback")

    # Verify user was not created
    with db_pool.get_session() as session:
        user = session.query(User).filter_by(
            username="rollbackuser"
        ).first()
        assert user is None


def test_pool_health_check(db_pool: DatabasePool):
    """Test pool health check."""
    assert db_pool.health_check()

    # Test cleanup and reset
    db_pool.cleanup_pool()
    db_pool.reset_pool()

    # Verify pool is still healthy
    assert db_pool.health_check()


def test_concurrent_transactions(db_pool: DatabasePool):
    """Test concurrent database transactions."""
    import threading
    import queue

    results = queue.Queue()

    def worker(user_id: int):
        try:
            with db_pool.transaction() as session:
                user = User(
                    username=f"user{user_id}",
                    email=f"user{user_id}@example.com",
                    password_hash="hashed_password",
                    role="user"
                )
                session.add(user)
            results.put(("success", user_id))

        except Exception as e:
            results.put(("error", user_id, str(e)))

    # Start multiple threads
    threads = []
    for i in range(5):
        thread = threading.Thread(target=worker, args=(i,))
        thread.start()
        threads.append(thread)

    # Wait for all threads
    for thread in threads:
        thread.join()

    # Check results
    success_count = 0
    while not results.empty():
        result = results.get()
        if result[0] == "success":
            success_count += 1

    assert success_count == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
