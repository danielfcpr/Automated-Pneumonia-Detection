# --- Base image (Python 3.12, slim) ---
FROM python:3.12-slim-bookworm

# Hygiene
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Minimal OS deps for OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install deps first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Non-root for security
RUN useradd -m -u 10002 appuser && chown -R appuser:appuser /app
USER appuser

# Configure runtime
ENV API_URL=""
EXPOSE 8080

# Start Gradio UI
CMD ["python", "app.py"]
