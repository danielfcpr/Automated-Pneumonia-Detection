from fastapi.testclient import TestClient
import app


def test_predict_missing_file_returns_422():
    client = TestClient(app.app)
    r = client.post("/predict")
    assert r.status_code == 422


def test_predict_rejects_unsupported_content_type():
    client = TestClient(app.app)
    files = {"file": ("x.txt", b"hello", "text/plain")}
    r = client.post("/predict", files=files)
    assert r.status_code == 415


def test_predict_rejects_empty_file():
    client = TestClient(app.app)
    files = {"file": ("xray.png", b"", "image/png")}
    r = client.post("/predict", files=files)
    assert r.status_code == 400
