from fastapi.testclient import TestClient
import app


def test_version():
    client = TestClient(app.app)
    r = client.get("/version")
    assert r.status_code == 200

    data = r.json()
    assert "model" in data
    assert "api" in data
    assert isinstance(data["model"], str)
    assert isinstance(data["api"], str)
