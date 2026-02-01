from fastapi.testclient import TestClient
import app

def test_healthz():
    client = TestClient(app.app)
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
