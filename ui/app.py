import os
import time
import io
import cv2
import requests
import numpy as np
import gradio as gr
from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv("API_URL")
if API_URL is None:
    raise RuntimeError("API_URL env var is not set. Set it to your FastAPI /predict endpoint.")

def infer(image_np: np.ndarray):
    """
    image_np: HxWxC RGB uint8 (Gradio provides this)
    Sends JPEG to FastAPI and returns label + confidence + latency.
    """
    if image_np is None:
        return "No image received."

    # Encode to JPEG
    ok, buf = cv2.imencode(".jpg", image_np[:, :, ::-1])  # Gradio gives RGB; OpenCV expects BGR
    if not ok:
        return "Failed to encode image."

    files = {"file": ("upload.jpg", io.BytesIO(buf.tobytes()), "image/jpeg")}

    t0 = time.time()
    try:
        r = requests.post(API_URL, files=files, timeout=15)
        r.raise_for_status()
        data = r.json()
    except requests.exceptions.RequestException as e:
        return f"Request error: {e}"
    except ValueError:
        return "Invalid JSON from server."

    latency_ms = (time.time() - t0) * 1000.0

    # Expected keys: label, prob, model_version
    label = data.get("label", "UNKNOWN")
    prob = data.get("prob", None)
    mv = data.get("model_version", "n/a")

    if prob is None:
        return f"Model output missing probability. Raw: {data}"

    return (
        f"Prediction: **{label}**\n\n"
        f"Confidence: **{prob*100:.0f}**%\n\n"
        f"Model: `{mv}`\n"
        f"Latency: ~{latency_ms:.0f} ms"
    )

demo = gr.Interface(
    fn=infer,
    inputs=gr.Image(type="numpy", label="Chest X-ray (PNG/JPG)"),
    outputs=gr.Markdown(),
    title="Pneumonia Detector (MVP)",
    description="Upload a chest X-ray image to get a prediction. "
                "This is an MVP demo — not for clinical use.",
    allow_flagging="never",
)

if __name__ == "__main__":
    # Gradio default port is
    demo.launch(server_name="0.0.0.0", server_port=8080, show_api=False, share=False)
