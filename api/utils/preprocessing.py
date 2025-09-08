import numpy as np
import cv2 as cv

TARGET_SIZE = (128, 128)   # (W, H)
MODEL_INPUT_CHANNELS = 3   # RGB

def decode_and_preprocess(b: bytes) -> np.ndarray:
    """
    Bytes -> BGR (cv2) -> RGB -> resize(128x128) -> [0,1] float32 -> (1,H,W,3)
    """
    arr = np.frombuffer(b, np.uint8)
    img = cv.imdecode(arr, cv.IMREAD_COLOR)       # HxWx3 (BGR) uint8
    if img is None:
        raise ValueError("Could not decode image.")
    # BGR -> RGB to match color_mode='rgb' used in training
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # Resize
    img = cv.resize(img, TARGET_SIZE, interpolation=cv.INTER_LINEAR)
    # Normalize & add batch dim
    x = img.astype(np.float32) / 255.0              # (H,W,3) in [0,1]
    x = np.expand_dims(x, axis=0)                   # (1,H,W,3)
    return x
