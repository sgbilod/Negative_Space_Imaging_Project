
import time
import random
from locust import HttpUser, task, between

class NegativeSpaceUser(HttpUser):
    wait_time = between(1, 3)

    @task(20)
    def health_check(self):
        """Health check endpoint - 20% of requests"""
        self.client.get("/api/health")

    @task(30)
    def get_images(self):
        """Get images endpoint - 30% of requests"""
        self.client.get("/api/images", params={"limit": 10, "offset": random.randint(0, 100)})

    @task(25)
    def process_image(self):
        """Process image endpoint - 25% of requests"""
        payload = {
            "image_url": f"https://example.com/image_{random.randint(1, 1000)}.jpg",
            "processing_options": {
                "enhancement": True,
                "negative_space_detection": True,
                "quality": "high"
            }
        }
        self.client.post("/api/process", json=payload)

    @task(15)
    def get_analytics(self):
        """Analytics endpoint - 15% of requests"""
        self.client.get("/api/analytics", params={"period": "24h"})

    @task(10)
    def auth_login(self):
        """Authentication endpoint - 10% of requests"""
        payload = {
            "username": f"user_{random.randint(1, 100)}",
            "password": "test_password"
        }
        self.client.post("/api/auth/login", json=payload)

    def on_start(self):
        """Setup method called when a user starts"""
        self.client.headers.update({
            "Authorization": "Bearer test-token",
            "Content-Type": "application/json",
            "User-Agent": "NegativeSpaceLoadTest/1.0"
        })
