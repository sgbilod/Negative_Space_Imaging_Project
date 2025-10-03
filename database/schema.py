"""
Database Schema for Negative Space Imaging Project
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.
"""

from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, DateTime, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from datetime import datetime
import uuid

Base = declarative_base()

class User(Base):
    """User model for authentication and access control."""
    __tablename__ = 'users'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(120), unique=True, nullable=False)
    password_hash = Column(String(256), nullable=False)
    role = Column(String(20), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)

    # Relationships
    sessions = relationship("Session", back_populates="user")
    images = relationship("Image", back_populates="owner")

class Session(Base):
    """Session model for tracking user sessions."""
    __tablename__ = 'sessions'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey('users.id'))
    token = Column(String(256), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    is_active = Column(Boolean, default=True)

    # Relationships
    user = relationship("User", back_populates="sessions")

class Image(Base):
    """Image model for storing image metadata."""
    __tablename__ = 'images'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    owner_id = Column(String(36), ForeignKey('users.id'))
    filename = Column(String(255), nullable=False)
    file_hash = Column(String(64), nullable=False)
    file_size = Column(Integer, nullable=False)
    mime_type = Column(String(50), nullable=False)
    width = Column(Integer)
    height = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    processed = Column(Boolean, default=False)

    # Image specific metadata
    exposure_time = Column(Float)
    iso_speed = Column(Integer)
    focal_length = Column(Float)
    aperture = Column(Float)

    # Relationships
    owner = relationship("User", back_populates="images")
    processing_jobs = relationship("ProcessingJob", back_populates="image")
    signatures = relationship("Signature", back_populates="image")

class ProcessingJob(Base):
    """Model for tracking image processing jobs."""
    __tablename__ = 'processing_jobs'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    image_id = Column(String(36), ForeignKey('images.id'))
    status = Column(String(20), nullable=False)
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    error_message = Column(String(500))

    # Processing parameters
    algorithm = Column(String(50), nullable=False)
    parameters = Column(String(1000))  # JSON string

    # Resource usage
    cpu_time = Column(Float)
    gpu_time = Column(Float)
    memory_used = Column(Float)

    # Relationships
    image = relationship("Image", back_populates="processing_jobs")
    results = relationship("ProcessingResult", back_populates="job")

class ProcessingResult(Base):
    """Model for storing processing results."""
    __tablename__ = 'processing_results'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    job_id = Column(String(36), ForeignKey('processing_jobs.id'))
    result_type = Column(String(50), nullable=False)
    result_data = Column(String(10000))  # JSON string
    created_at = Column(DateTime, default=datetime.utcnow)

    # Verification
    hash = Column(String(64), nullable=False)
    verified = Column(Boolean, default=False)

    # Relationships
    job = relationship("ProcessingJob", back_populates="results")

class Signature(Base):
    """Model for storing cryptographic signatures."""
    __tablename__ = 'signatures'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    image_id = Column(String(36), ForeignKey('images.id'))
    signer_id = Column(String(36), ForeignKey('users.id'))
    signature = Column(String(512), nullable=False)
    signed_at = Column(DateTime, default=datetime.utcnow)
    is_valid = Column(Boolean, default=True)

    # Relationships
    image = relationship("Image", back_populates="signatures")

class AuditLog(Base):
    """Model for security audit logging."""
    __tablename__ = 'audit_logs'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey('users.id'))
    action = Column(String(50), nullable=False)
    resource_type = Column(String(50))
    resource_id = Column(String(36))
    timestamp = Column(DateTime, default=datetime.utcnow)
    ip_address = Column(String(45))
    user_agent = Column(String(200))
    status = Column(String(20))
    details = Column(String(1000))  # JSON string


# Encrypted image storage
from sqlalchemy import LargeBinary

class EncryptedImage(Base):
    """Model for storing encrypted image data."""
    __tablename__ = 'encrypted_images'

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    image_id = Column(String(36), ForeignKey('images.id'))
    encrypted_data = Column(LargeBinary, nullable=False)  # Encrypted image data
    encryption_key_id = Column(String(36), nullable=False)  # Key management reference
    encryption_algorithm = Column(String(50), nullable=False, default='AES-256-GCM')
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    image = relationship("Image")

# Database initialization function
def init_db(connection_string: str):
    """Initialize the database with all tables."""
    engine = create_engine(connection_string)
    Base.metadata.create_all(engine)
    return engine

# Session factory
def create_session(engine):
    """Create a new database session."""
    Session = sessionmaker(bind=engine)
    return Session()
