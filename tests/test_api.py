import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health") # Assuming /health exists, or I should add it
    # If /health doesn't exist, this will fail as expected
    assert response.status_code == 200

def test_predict_no_file():
    response = client.post("/predict")
    assert response.status_code == 422 # FastAPI validation error

# You need a sample image in the repository to test /predict effectively
