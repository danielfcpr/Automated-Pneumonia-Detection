from pathlib import Path
from fastapi.testclient import TestClient
import app


def _first_image(folder: Path):
    for ext in (".png", ".jpg", ".jpeg"):
        imgs = list(folder.rglob(f"*{ext}"))
        if imgs:
            return imgs[0]
    return None

def test_predict_smoke_one_image_per_class():

    client = TestClient(app.app)

    repo_root = Path(__file__).resolve().parents[2]
    normal_dir = repo_root / "data" / "normal"
    pneu_dir = repo_root / "data" / "pneumonia"

    assert normal_dir.exists(), f"Missing folder: {normal_dir}"
    assert pneu_dir.exists(), f"Missing folder: {pneu_dir}"

    normal_img = _first_image(normal_dir)
    pneu_img = _first_image(pneu_dir)

    assert normal_img is not None, "No image found in data/normal"
    assert pneu_img is not None, "No image found in data/pneumonia"

    for img_path in (normal_img, pneu_img):
        with open(img_path, "rb") as f:
            files = {"file": (img_path.name, f, "image/jpeg")}
            r = client.post("/predict", files=files)

        assert r.status_code == 200, f"Failed on {img_path}: {r.text}"
        data = r.json()

        assert set(data.keys()) == {"label", "prob", "model_version"}
        assert data["label"] in ("NORMAL", "PNEUMONIA")
        assert 0.0 <= float(data["prob"]) <= 1.0
        assert isinstance(data["model_version"], str) and data["model_version"]
