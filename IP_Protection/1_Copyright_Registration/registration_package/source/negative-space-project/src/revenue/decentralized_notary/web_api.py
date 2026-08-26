"""
Web API for the Decentralized Notary Network

This module provides a RESTful API for interacting with the
Decentralized Time Notary Network using FastAPI.
"""

from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Optional, Union, Any
import uvicorn
import json
import uuid
from datetime import datetime

from .notary_network import NotaryAPI, NotaryNetwork
from .blockchain_connector import BlockchainConnector, Blockchain


# Define API models
class LandmarkRegistration(BaseModel):
    name: str
    description: str
    location: Dict[str, float]
    spatial_signature: Union[str, List[List[float]]]
    metadata: Optional[Dict[str, Any]] = None


class NodeRegistration(BaseModel):
    owner_id: str
    owner_data: Optional[Dict[str, Any]] = None


class ProofOfViewSubmission(BaseModel):
    node_id: str
    landmark_id: str
    proof_signature: Union[str, List[List[float]]]


class NotarizationRequest(BaseModel):
    document_hash: str
    metadata: Optional[Dict[str, Any]] = None
    min_nodes: Optional[int] = 3


class VerificationRequest(BaseModel):
    notarization_id: str


# Initialize the API
app = FastAPI(
    title="Decentralized Notary Network API",
    description="API for the Negative Space Imaging Project's Decentralized Notary Network",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize shared resources
blockchain = Blockchain()
blockchain_connector = BlockchainConnector(blockchain)
notary_api = NotaryAPI()


# Error handling middleware
@app.middleware("http")
async def errors_handling(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as exc:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": str(exc)}
        )


# API endpoints
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Decentralized Notary Network API",
        "version": "1.0.0",
        "description": "API for the Negative Space Imaging Project's Decentralized Notary Network"
    }


@app.post("/landmarks", status_code=status.HTTP_201_CREATED)
async def register_landmark(landmark: LandmarkRegistration):
    """Register a new landmark for Proof-of-View validation."""
    result = notary_api.register_landmark(landmark.dict())
    
    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["error"]
        )
        
    return result


@app.get("/landmarks/{landmark_id}")
async def get_landmark(landmark_id: str):
    """Get details of a registered landmark."""
    landmark = notary_api.network.landmark_registry.get_landmark(landmark_id)
    
    if not landmark:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Landmark not found: {landmark_id}"
        )
        
    return landmark


@app.get("/landmarks/near")
async def get_landmarks_near(latitude: float, longitude: float, radius_km: float = 10.0):
    """Get landmarks near a specific location."""
    landmarks = notary_api.network.landmark_registry.get_landmarks_near(
        latitude=latitude,
        longitude=longitude,
        radius_km=radius_km
    )
    
    return {
        "count": len(landmarks),
        "landmarks": landmarks
    }


@app.post("/nodes", status_code=status.HTTP_201_CREATED)
async def register_node(registration: NodeRegistration):
    """Register a new notary node."""
    result = notary_api.register_node(registration.dict())
    
    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["error"]
        )
        
    return result


@app.post("/nodes/proof-of-view")
async def submit_proof_of_view(submission: ProofOfViewSubmission):
    """Submit a Proof-of-View for a node."""
    result = notary_api.submit_proof_of_view(submission.dict())
    
    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["error"]
        )
        
    if not result.get("success", False):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.get("reason", "Proof submission failed")
        )
    
    # Record on blockchain
    blockchain_result = blockchain_connector.record_proof_of_view(result)
    
    # Combine results
    result["blockchain"] = blockchain_result
    
    return result


@app.post("/notarize", status_code=status.HTTP_201_CREATED)
async def notarize_document(request: NotarizationRequest):
    """Notarize a document using the notary network."""
    result = notary_api.notarize_document(request.dict())
    
    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result["error"]
        )
        
    if not result.get("success", False):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.get("reason", "Notarization failed")
        )
    
    # Record on blockchain
    blockchain_result = blockchain_connector.record_notarization(result["notarization"])
    
    # Combine results
    result["blockchain"] = blockchain_result
    
    return result


@app.post("/verify")
async def verify_notarization(request: VerificationRequest):
    """Verify a document notarization."""
    # Check in the notary network
    network_result = notary_api.verify_notarization(request.dict())
    
    if "error" in network_result:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=network_result["error"]
        )
        
    if not network_result.get("verified", False):
        return {
            "verified": False,
            "network_verification": network_result
        }
        
    # The blockchain verification requires a transaction ID, which would be stored
    # in the notarization metadata in a real implementation
    # For Phase 1, we'll just return the network verification
    
    return {
        "verified": True,
        "network_verification": network_result
    }


@app.get("/blockchain/status")
async def get_blockchain_status():
    """Get the current status of the blockchain."""
    return blockchain_connector.get_blockchain_status()


def start_server():
    """Start the FastAPI server."""
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    start_server()
