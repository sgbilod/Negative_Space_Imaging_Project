"""Backend load test script using locust"""
from locust import HttpUser, TaskSet, task, between
import os

class UserBehavior(TaskSet):
    @task(2)
    def login(self):
        self.client.post("/token", data={"username": "testuser", "password": "testpass"})

    @task(1)
    def get_dashboard(self):
        self.client.get("/jobs/123")

    @task(1)
    def process_image(self):
        files = {'image': ('test.jpg', b'dummy', 'image/jpeg')}
        self.client.post("/images/process", files=files, data={"processing_type": "enhance"})

class WebsiteUser(HttpUser):
    tasks = [UserBehavior]
    wait_time = between(1, 3)
    host = os.getenv("API_HOST", "http://localhost:8000")
