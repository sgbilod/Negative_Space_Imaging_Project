
"""API endpoint tests (fully corrected)"""

import pytest
from fastapi.testclient import TestClient
from api import app, create_access_token

client = TestClient(app)

@pytest.fixture(scope="session", autouse=True)
def setup_test_user():
    # Create test user with admin role if not exists
    token = create_access_token("testuser")
    user_payload = {
        "username": "testuser",
        "email": "testuser@example.com",
        "roles": ["admin"]
    }
    client.post(
        "/users",
        headers={"Authorization": f"Bearer {token}"},
        json=user_payload
    )

@pytest.fixture
def test_token():
    return create_access_token("testuser")

def test_login():
    response = client.post(
        "/token",
        data={
            "username": "testuser",
            "password": "testpass"
        }
    )
    assert response.status_code == 200
    assert "access_token" in response.json()
    assert response.json()["token_type"] == "bearer"

def test_create_user(test_token):
    response = client.post(
        "/users",
        headers={"Authorization": f"Bearer {test_token}"},
        json={
            "username": "newuser",
            "email": "newuser@example.com",
            "roles": ["viewer"]
        }
    )
    assert response.status_code == 200

def test_process_image(test_token):
    data = {
        "image_id": "testimg1",
        "processing_type": "enhance",
        "parameters": {"contrast": 10, "brightness": 5}
    }
    response = client.post(
        "/images/process",
        headers={"Authorization": f"Bearer {test_token}"},
        json=data
    )
    assert response.status_code == 200
    assert "job_id" in response.json()

def test_get_job_status(test_token):
    data = {
        "image_id": "testimg2",
        "processing_type": "enhance",
        "parameters": {"contrast": 10, "brightness": 5}
    }
    create_response = client.post(
        "/images/process",
        headers={"Authorization": f"Bearer {test_token}"},
        json=data
    )
    job_id = create_response.json()["job_id"]
    status_response = client.get(
        f"/jobs/{job_id}",
        headers={"Authorization": f"Bearer {test_token}"}
    )
    assert status_response.status_code == 200
    assert "status" in status_response.json()

def test_get_audit_logs(test_token):
    response = client.get(
        "/audit-logs",
        headers={"Authorization": f"Bearer {test_token}"}
    )
    assert response.status_code == 200
    assert "logs" in response.json()

def test_get_security_events(test_token):
    response = client.get(
        "/security-events",
        headers={"Authorization": f"Bearer {test_token}"}
    )
    assert response.status_code == 200
    assert "events" in response.json()
