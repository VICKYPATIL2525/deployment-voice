# =============================================================================
# Mindspace — Voice API Client Script (call_api.py)
# =============================================================================
# This script demonstrates how to call every endpoint of the Mindspace Voice
# Mental Health Classifier API from Python.
#
# It reads the API key from the .env file in the project root (same key used
# by the FastAPI server) and sends authenticated requests to all 4 endpoints:
#
#   1. GET  /           — Service metadata (model name, classes, accuracy)
#   2. GET  /health     — Readiness check (confirms artifacts are loaded)
#   3. GET  /model/info — Full model metadata (hyperparams, metrics, features)
#   4. POST /predict    — Prediction from 1,351 acoustic voice features
#
# Prerequisites:
#   - The API server must be running on http://127.0.0.1:9100
#     Start it with: uvicorn api_voice_to_sentiment:app --port 9100 --reload
#   - A valid .env file must exist with: API_KEY=your-secret-key
#   - The `requests` and `python-dotenv` packages must be installed
#
# Usage:
#   python call_api.py
# =============================================================================

import json
import os
import requests
from dotenv import load_dotenv

# ─── Configuration ────────────────────────────────────────────────────────────
# load_dotenv() reads the .env file from the current directory and injects
# its key-value pairs into os.environ so we can access API_KEY securely
# without hardcoding it in source code.
load_dotenv()

# Base URL where the FastAPI server is running (default local development port)
BASE_URL = "http://127.0.0.1:9100"

# Read the API key from environment — will raise KeyError if .env is missing
API_KEY = os.environ["API_KEY"]

# Common headers sent with every request:
#   - X-API-Key: authenticates the client (required by all endpoints)
#   - Content-Type: tells the server we are sending JSON
HEADERS = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json",
}


# ─── Endpoint 1: GET / ────────────────────────────────────────────────────────
def call_root():
    """
    GET / — Service Info

    Returns basic runtime metadata about the deployed model:
      - service name
      - model type (XGBoost)
      - test accuracy (0.6033)
      - list of 6 output classes
      - number of input features (1,351)
    """
    print("=" * 60)
    print("1. GET /  (Service Info)")
    print("=" * 60)

    # Send GET request with API key header
    resp = requests.get(f"{BASE_URL}/", headers=HEADERS)
    print(f"Status: {resp.status_code}")

    # Pretty-print the full JSON response
    print(json.dumps(resp.json(), indent=2))


# ─── Endpoint 2: GET /health ──────────────────────────────────────────────────
def call_health():
    """
    GET /health — Health Check

    Confirms the server is up and all ML artifacts (model, scaler,
    label encoder, outlier transformers, feature names, metadata)
    were loaded successfully at startup.

    Returns:
      - status: "ok"
      - artifacts_loaded: true/false
    """
    print("\n" + "=" * 60)
    print("2. GET /health  (Health Check)")
    print("=" * 60)

    resp = requests.get(f"{BASE_URL}/health", headers=HEADERS)
    print(f"Status: {resp.status_code}")
    print(json.dumps(resp.json(), indent=2))


# ─── Endpoint 3: GET /model/info ──────────────────────────────────────────────
def call_model_info():
    """
    GET /model/info — Full Model Metadata

    Returns the complete model_metadata.json saved during training, including:
      - best_model_name   ("XGBoost")
      - tuned hyperparameters
      - cross-validation score
      - test_metrics       (accuracy, classification report)
      - class_names        (the 6 mental health profiles)
      - feature_names      (all 1,351 acoustic feature names)
      - n_features         (1351)

    This endpoint is useful for inspecting exactly which features the
    model expects and what performance was measured during training.
    """
    print("\n" + "=" * 60)
    print("3. GET /model/info  (Model Metadata)")
    print("=" * 60)

    resp = requests.get(f"{BASE_URL}/model/info", headers=HEADERS)
    print(f"Status: {resp.status_code}")

    data = resp.json()
    # Print a concise summary instead of the full response
    # (the full response includes all 1,351 feature names)
    print(f"  Model        : {data.get('best_model_name')}")
    print(f"  Accuracy     : {data.get('test_metrics', {}).get('accuracy')}")
    print(f"  Classes      : {data.get('class_names')}")
    print(f"  Num Features : {data.get('n_features')}")


# ─── Endpoint 4: POST /predict ────────────────────────────────────────────────
def call_predict():
    """
    POST /predict — Run Prediction

    Sends 1,351 pre-extracted OpenSMILE acoustic features to the API
    and receives the predicted mental health profile.

    Input format (JSON body):
    {
        "features": {
            "pcm_fftMag_fband250-650_sma_de_pctlrange0-1": 6.524812,
            "mfcc_sma_de[11]_meanPeakDist": 0.229548,
            ...  (all 1,351 feature names with float values)
        }
    }

    The features are extracted from raw audio using OpenSMILE and include:
      - MFCC coefficients     (voice timbre and tone shape)
      - F0 / pitch features   (how high or low the voice is)
      - Harmonics-to-noise    (voice clarity vs breathiness)
      - Spectral features     (energy distribution across frequencies)
      - Shimmer / jitter      (voice tremor and irregularity)

    Output:
      - prediction    : predicted class (e.g. "normal", "depression")
      - confidence    : probability of the predicted class (0.0–1.0)
      - probabilities : full distribution across all 6 classes
      - model         : model name ("XGBoost")
      - accuracy      : model's test-set accuracy (0.6033)
    """
    print("\n" + "=" * 60)
    print("4. POST /predict  (Prediction)")
    print("=" * 60)

    # Load the demo sample payload from the sample data folder.
    # This file contains 1,351 pre-extracted acoustic features from
    # a "normal" voice sample, ready to send to the API.
    with open("demo-api-input-data-sample/voice_normal_sample_1.json") as f:
        payload = json.load(f)

    # Send POST request with the feature payload as JSON body
    resp = requests.post(f"{BASE_URL}/predict", headers=HEADERS, json=payload)
    print(f"Status: {resp.status_code}")

    # Parse and display the prediction results
    result = resp.json()
    print(f"  Prediction   : {result.get('prediction')}")
    print(f"  Confidence   : {result.get('confidence')}")
    print(f"  Model        : {result.get('model')}")
    print(f"  Accuracy     : {result.get('accuracy')}")
    print(f"  Probabilities:")
    for cls, prob in result.get("probabilities", {}).items():
        print(f"    {cls:12s} : {prob}")


# ─── Main ─────────────────────────────────────────────────────────────────────
# When run directly, call all 4 endpoints in sequence to verify
# the API is working end-to-end.
if __name__ == "__main__":
    call_root()
    call_health()
    call_model_info()
    call_predict()
    print("\n" + "=" * 60)
    print("All endpoints called successfully.")
    print("=" * 60)
