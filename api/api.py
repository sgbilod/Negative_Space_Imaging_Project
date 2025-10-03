"""
RESTful API Server for Negative Space Imaging
Copyright (c) 2025 Stephen Bilodeau. All rights reserved.
"""

from fastapi import FastAPI, HTTPException, Security, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import jwt
import uvicorn
from pydantic import BaseModel
# ...existing code...

# Import security components
from security.biometric_auth import BiometricVerifier
from security.quantum_encryption import QuantumEncryption
from security.rbac import RBACSystem, Permission
from security.audit_logging import AuditLogger
from security.security_monitor import SecurityMonitor, SecurityEvent, EventSeverity, EventCategory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("API")

# Initialize FastAPI app
app = FastAPI(
    title="Negative Space Imaging API",
    description="Advanced imaging system with quantum-enhanced security",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with actual frontend origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Initialize security components
biometric_verifier = BiometricVerifier()
quantum_encryption = QuantumEncryption()
rbac_system = RBACSystem()
audit_logger = AuditLogger()
security_monitor = SecurityMonitor()

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# JWT configuration
import os
JWT_SECRET = os.getenv("JWT_SECRET", "change-me-in-production")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30


# Data Models
class Token(BaseModel):
    """JWT token response model."""
    access_token: str
    token_type: str


class UserCredentials(BaseModel):
    """User login credentials."""
    username: str
    password: str
    biometric_data: Optional[str] = None


class UserProfile(BaseModel):
    """User profile data."""
    username: str
    email: str
    roles: List[str]
    metadata: Optional[Dict] = None


class ImageData(BaseModel):
    """Image processing request data."""
    image_id: str
    processing_type: str
    parameters: Optional[Dict] = None


class ProcessingJob(BaseModel):
    """Image processing job details."""
    job_id: str
    status: str
    progress: float
    results: Optional[Dict] = None


# Authentication and Authorization
def get_current_user(token: str = Security(oauth2_scheme)) -> UserProfile:
    """Validate JWT token and return user."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token"
            )

        # Get user profile
        # Replace with actual user lookup
        user = UserProfile(
            username=username,
            email=f"{username}@example.com",
            roles=["user"]
        )

        return user

    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )


def check_permission(
    user: UserProfile,
    required_permission: Permission
) -> bool:
    """Check if user has required permission."""
    return rbac_system.check_permission(user.username, required_permission)


# API Routes
@app.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """User login endpoint."""
    try:
        # Verify credentials (replace with actual verification)
        if not verify_credentials(form_data.username, form_data.password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )

        # Create access token
        access_token = create_access_token(form_data.username)

        # Log successful login
        audit_logger.log_event(
            "USER_LOGIN",
            "INFO",
            {"user": form_data.username},
            "auth_service"
        )

        return Token(
            access_token=access_token,
            token_type="bearer"
        )

    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@app.post("/verify-biometric")
async def verify_biometric(
    biometric_data: str,
    user: UserProfile = Depends(get_current_user)
):
    """Verify user's biometric data."""
    try:
        result = biometric_verifier.verify_biometric(
            user.username,
            biometric_data.encode()
        )

        if not result:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Biometric verification failed"
            )

        return {"status": "success"}

    except Exception as e:
        logger.error(f"Biometric verification error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Verification failed"
        )


@app.post("/images/process")
async def process_image(
    data: ImageData,
    user: UserProfile = Depends(get_current_user)
):
    """Process image with specified parameters."""
    try:
        # Check permissions
        if not check_permission(user, Permission.EXECUTE):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )

        # Create processing job
        job = create_processing_job(data, user)

        # Log job creation
        audit_logger.log_event(
            "IMAGE_PROCESSING",
            "INFO",
            {
                "user": user.username,
                "image_id": data.image_id,
                "job_id": job.job_id
            },
            "processing_service"
        )

        return job

    except Exception as e:
        logger.error(f"Image processing error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Processing failed"
        )


@app.get("/jobs/{job_id}")
async def get_job_status(
    job_id: str,
    user: UserProfile = Depends(get_current_user)
):
    """Get status of processing job."""
    try:
        # Check permissions
        if not check_permission(user, Permission.READ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )

        # Get job status
        job = get_processing_job(job_id)
        if job is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found"
            )

        return job

    except Exception as e:
        logger.error(f"Job status error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Status check failed"
        )


@app.post("/users")
async def create_user(
    user: UserProfile,
    current_user: UserProfile = Depends(get_current_user)
):
    """Create new user."""
    try:
        # Check admin permission
        if not check_permission(current_user, Permission.ADMIN):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )

        # Create user in RBAC system
        success = rbac_system.create_user(user)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User creation failed"
            )

        # Log user creation
        audit_logger.log_event(
            "USER_CREATED",
            "INFO",
            {"user": user.username},
            "user_service"
        )

        return {"status": "success"}

    except Exception as e:
        logger.error(f"User creation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User creation failed"
        )


@app.get("/audit-logs")
async def get_audit_logs(
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    user: UserProfile = Depends(get_current_user)
):
    """Get audit logs."""
    try:
        # Check audit permission
        if not check_permission(user, Permission.AUDIT):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )

        # Get logs
        logs = audit_logger.search_logs(
            {"user": user.username},
            start_time,
            end_time
        )

        return {"logs": logs}

    except Exception as e:
        logger.error(f"Audit log error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Log retrieval failed"
        )


@app.get("/security-events")
async def get_security_events(
    severity: Optional[str] = None,
    category: Optional[str] = None,
    user: UserProfile = Depends(get_current_user)
):
    """Get security events."""
    try:
        # Check admin permission
        if not check_permission(user, Permission.ADMIN):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )

        # Build query
        query = {}
        if severity:
            query["severity"] = severity
        if category:
            query["category"] = category

        # Get events
        events = security_monitor.search_logs(query)

        return {"events": events}

    except Exception as e:
        logger.error(f"Security event error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Event retrieval failed"
        )


# Helper Functions
def verify_credentials(username: str, password: str) -> bool:
    """Verify user credentials."""
    # Replace with actual verification logic
    return True


def create_access_token(username: str) -> str:
    """Create JWT access token."""
    expires = datetime.utcnow() + timedelta(
        minutes=ACCESS_TOKEN_EXPIRE_MINUTES
    )

    payload = {
        "sub": username,
        "exp": expires
    }

    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def create_processing_job(
    data: ImageData,
    user: UserProfile
) -> ProcessingJob:
    """Create new image processing job."""
    # Replace with actual job creation logic
    return ProcessingJob(
        job_id="123",
        status="pending",
        progress=0.0
    )


def get_processing_job(job_id: str) -> Optional[ProcessingJob]:
    """Get processing job status."""
    # Replace with actual job retrieval logic
    return ProcessingJob(
        job_id=job_id,
        status="running",
        progress=0.5
    )


# Main entry point
if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
