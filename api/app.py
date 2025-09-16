# api/app.py
import os
from typing import Literal
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
import numpy as np
import keras
from utils.preprocessing import decode_and_preprocess

# ----- Config -----
MODEL_PATH = os.getenv("MODEL_PATH", "models/CNN_model.keras")
IMAGE_CONTENT_TYPES = {"image/png", "image/jpeg"}
APP_TITLE = "Pneumonia Inference API"
APP_VERSION = "0.1.0"

# ----- App -----
app = FastAPI(title=APP_TITLE, version=APP_VERSION)

# Allow your Gradio UI to call the API (tighten origins later if you want)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

class Prediction(BaseModel):
    label: Literal["NORMAL", "PNEUMONIA"]
    prob: float
    model_version: str

def _load_model():
    try:
        return keras.saving.load_model(MODEL_PATH)
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {e!s}")

_model = None
def get_model():
    global _model
    if _model is None:
        _model = _load_model()
    return _model

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.get("/version")
def version():
    return {"model": os.path.basename(MODEL_PATH), "api": APP_VERSION}

@app.post("/predict", response_model=Prediction)
async def predict(file: UploadFile = File(...)):
    if file.content_type not in IMAGE_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Only PNG/JPEG supported.")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file.")

    try:
        # (1, 128, 128, 3) float32 in [0,1], RGB — matches training
        x: np.ndarray = decode_and_preprocess(raw)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing failed: {e!s}")

    model = get_model()

    # Single-neuron sigmoid -> pneumonia probability in [0,1]
    try:
        prob_pneu = float(model.predict(x, verbose=0)[0][0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e!s}")

    label = "PNEUMONIA" if prob_pneu >= 0.5 else "NORMAL"
    prob = prob_pneu if label == "PNEUMONIA" else (1.0 - prob_pneu)

    return {
        "label": label,
        "prob": prob,
        "model_version": os.path.basename(MODEL_PATH),
    }
