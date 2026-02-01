# Pneumonia Detection — Machine Learning Demo

## Overview

This project is a **machine learning inference service** for detecting pneumonia from chest X-ray images.
It was built as a **portfolio project** to demonstrate end-to-end ML engineering practices, including
model serving, containerization, automated testing, and CI/CD.

⚠️ **Important**  
This application is provided **for educational and demonstration purposes only**.
It is **NOT intended for clinical diagnosis or medical decision-making**.

---

## Key Features

- CNN-based pneumonia detection model (TensorFlow / Keras)
- FastAPI inference service with:
  - `/healthz` health endpoint
  - `/version` model/API version endpoint
  - `/predict` image inference endpoint
- Gradio web UI for interactive testing
- Fully containerized with Docker
- Automated testing:
  - API validation tests
  - End-to-end inference smoke test using real images
- Continuous Integration with GitHub Actions
  - Tests executed on every push and pull request
  - Docker image build validation in CI

---

## Tech Stack

**Machine Learning**
- TensorFlow / Keras
- NumPy
- OpenCV (headless)

**Backend / API**
- FastAPI
- Pydantic
- Uvicorn

**UI**
- Gradio

**DevOps / MLOps**
- Docker
- GitHub Actions (CI)
- Pytest

---

## Project Structure

```text
Automated-Pneumonia-Detection/
├── api/
│   ├── app.py              # FastAPI inference service
│   ├── models/             # Trained CNN model (.keras)
│   ├── utils/              # Image preprocessing
│   ├── tests/              # Unit, API, and E2E tests
│   ├── Dockerfile
│   ├── requirements.txt
│   └── requirements-dev.txt
├── data/
│   ├── normal/             # Sample NORMAL X-ray images
│   └── pneumonia/          # Sample PNEUMONIA X-ray images
├── notebooks/              # Model training / experimentation
├── .github/workflows/
│   └── ci.yml              # GitHub Actions CI pipeline
├── pytest.ini
└── README.md
```

---

## Run Locally
1️. Clone the repository
```bash
git clone https://https://github.com/danielfcpr/Automated-Pneumonia-Detection.git
cd Automated-Pneumonia-Detection
```
2. Run with Docker (recommended)
```bash
docker build -t pneumonia-detection-api ./api
docker run --rm -p 8000:8080 pneumonia-api:local
```
3️. Run tests locally
```bash
pip install -r api/requirements.txt
pip install -r api/requirements-dev.txt
pytest -q
```
---

## CI / CD
This project uses GitHub Actions for Continuous Integration:

- All tests are executed on every push and pull request

- Docker image build is validated in CI

- CI must pass before changes are merged

This ensures reliability and reproducibility across environments.

---

## Model Notes

- Task: Binary image classification — normal vs pneumonia

- Training Data size: 5,856 images — 1,583 normal (label=0), 4,273 pneumonia (label=1)

- Split: 80% train / 10% val / 10% test (batch size 32)

- Model: CNN (Keras / TensorFlow)

- Key metric: Recall on pneumonia ≈ 98.5% (priority to reduce false negatives)

- Validation approach: Repeated training on different splits to check robustness (principles of cross-validation; not full recorded k-fold)
- The model is a CNN trained on chest X-ray images for pneumonia classification

- Inference returns:

    - predicted label (NORMAL / PNEUMONIA)

    - confidence score

    - model version (v1)

The model and predictions are not clinically validated

---
## Deployment

The application is deployed on Azure Container Apps.
Docker images are built using Azure Container Registry (ACR) remote builds and
then rolled out to the running services.
---
## Author

**Daniel Calvo Pérez**  
Machine Learning / Data Engineer    
This project is part of my professional portfolio.