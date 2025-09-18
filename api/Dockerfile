# --- Base image (Python 3.12, CPU) ---
FROM python:3.12-slim-bookworm

# --- Environment variables ---
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# --- System libs required by TensorFlow/OpenCV (headless) ---
# libgomp1       : OpenMP runtime (TensorFlow / some numpy ops)
# libglib2.0-0   : GLib used by OpenCV
# libgl1         : minimal OpenGL lib used by OpenCV headless codecs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    libgl1 \
 && rm -rf /var/lib/apt/lists/*

# --- Workdir ---
WORKDIR /app

# --- Python dependencies (pinned) ---
# Keep this step separate for better Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- App code & model ---
# Copies: app.py, utils/, models/CNN_model.keras, etc.
COPY . .

# --- Security: run as non-root ---
RUN useradd -m -u 10001 appuser && chown -R appuser:appuser /app
USER appuser

# --- Runtime configuration ---
ENV MODEL_PATH="models/CNN_model.keras"
EXPOSE 8080

# --- Start FastAPI (Uvicorn) ---
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
