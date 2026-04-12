# =============================================================================
# Mindspace Mental Health Classifier — Voice Features API (FastAPI Inference Server)
# =============================================================================
# This file is the backend API for voice/acoustic-based mental health prediction.
# It loads the trained XGBoost model and all preprocessing artifacts at startup,
# then serves predictions via HTTP endpoints.
#
# The API expects clients to send 1,351 pre-extracted acoustic features (from
# audio/speech using a tool like OpenSMILE) and returns the predicted mental
# health profile along with confidence and probabilities for all classes.
#
# Flow for every prediction request:
#   1. Client sends 1,351 float acoustic features as a flat JSON object
#   2. API validates all required feature names are present
#   3. API applies outlier smoothing (same transforms used during training)
#   4. API scales the features with RobustScaler (same scaler from training)
#   5. XGBoost model predicts one of 6 mental health profiles
#   6. API returns the label + confidence + all 6 class probabilities
#
# Compared to api_text_to_sentiment.py (LightGBM, 43 linguistic features, 7 classes):
#   - This API uses acoustic voice features (MFCC, pitch, shimmer, jitter, etc.)
#   - 1,351 input features (OpenSMILE feature set)
#   - 6 output classes: anxiety, bipolar, depression, normal, stress, suicidal
#   - XGBoost model with 60.33% test accuracy
# =============================================================================

import json
import os
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, model_validator
from typing import Dict

# ─── Load environment variables ───────────────────────────────────────────────
# Reads API_KEY from deployment/.env at startup.
# The voice API shares the same .env as the text API.
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# ─── API Key auth ─────────────────────────────────────────────────────────────
_API_KEY = os.environ.get("API_KEY", "")
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(key: str = Security(_api_key_header)) -> None:
    """FastAPI dependency — guards any route it is added to."""
    if not _API_KEY:
        raise HTTPException(status_code=500, detail="Server misconfiguration: API_KEY not set.")
    if key != _API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key. Pass it as X-API-Key header.")


# ─── Artifact paths ──────────────────────────────────────────────────────────
# Points to the XGBoost model artifacts folder.
# __file__ is deployment-voice/api_voice_to_sentiment.py → .parent is this folder.
ARTIFACTS_DIR = Path(__file__).parent / "pipeline_output" / "XGBoost_27032026_152209"

# ─── Global state (loaded once at startup) ────────────────────────────────────
artifacts: dict = {}


def load_artifacts() -> None:
    """
    Load all ML artifacts from disk into the global `artifacts` dict.
    Called once when the server starts up (see lifespan below).

    Artifacts loaded:
      - best_model.joblib           → trained XGBoost classifier
      - scaler.joblib               → RobustScaler (fit on training rows)
      - label_encoder.joblib        → maps integer predictions → class name strings
      - encoding_artifacts.joblib   → categorical encoding maps
      - outlier_transformers.joblib → per-column smoothing params
      - feature_names.json          → ordered list of 1,351 acoustic feature names
      - model_metadata.json         → hyperparams, CV score, test metrics, class names
    """
    artifacts["model"]                = joblib.load(ARTIFACTS_DIR / "best_model.joblib")
    artifacts["scaler"]               = joblib.load(ARTIFACTS_DIR / "scaler.joblib")
    artifacts["label_encoder"]        = joblib.load(ARTIFACTS_DIR / "label_encoder.joblib")
    artifacts["encoding"]             = joblib.load(ARTIFACTS_DIR / "encoding_artifacts.joblib")
    artifacts["outlier_transformers"] = joblib.load(ARTIFACTS_DIR / "outlier_transformers.joblib")
    artifacts["feature_names"]        = json.loads((ARTIFACTS_DIR / "feature_names.json").read_text())
    artifacts["metadata"]             = json.loads((ARTIFACTS_DIR / "model_metadata.json").read_text())


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager — runs artifact loading at startup.
    After loading artifacts, injects a real-data example into the PredictRequest
    schema so the Swagger UI /docs "Try it out" shows all 1,351 feature names
    with realistic values instead of an empty dict.
    """
    load_artifacts()

    # Build a Swagger example from the first row of feature_names with zeros,
    # then override with a real sample if the demo file exists.
    feature_names = artifacts.get("feature_names", [])
    example_features = {f: 0.0 for f in feature_names}

    # Try to load a real sample from the demo data folder
    demo_sample_path = Path(__file__).parent / "demo-api-input-data-sample" / "voice_normal_sample_1.json"
    if demo_sample_path.exists():
        demo_data = json.loads(demo_sample_path.read_text())
        if "features" in demo_data:
            example_features = demo_data["features"]

    # Inject into OpenAPI schema so Swagger shows real values
    PredictRequest.model_config["json_schema_extra"] = {
        "examples": [{"features": example_features}]
    }
    # Force Pydantic to rebuild the JSON schema with the new example
    PredictRequest.model_rebuild(force=True)

    yield


# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Mindspace Mental Health Classifier — Voice",
    description=(
        "Predicts mental health profile from acoustic voice features (1,351 OpenSMILE features). "
        "Output classes: anxiety, bipolar, depression, normal, stress, suicidal."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ─── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Tighten to specific frontend domains in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request / Response schemas ───────────────────────────────────────────────

class PredictRequest(BaseModel):
    """
    Input schema for the voice-based prediction endpoint.

    Send a flat JSON object where every key is one of the 1,351 acoustic feature
    names (from an OpenSMILE feature extractor) and every value is a float.

    Example (abbreviated):
    {
        "pcm_fftMag_fband250-650_sma_de_pctlrange0-1": 0.0042,
        "mfcc_sma_de[11]_meanPeakDist": 12.34,
        ...  (all 1,351 features required)
    }

    The full list of required feature names is available at GET /model/info
    under the "feature_names" key.

    Validation:
      - All 1,351 feature names must be present (no extras required, but missing names
        will cause a 422 error).
      - All values must be numeric (int or float). NaN and Inf are rejected.
    """
    model_config = {}  # Populated at startup with a real-data Swagger example

    features: Dict[str, float]

    @model_validator(mode="after")
    def validate_features(self) -> "PredictRequest":
        """
        Runtime validation against the authoritative feature list loaded from disk.
        Runs after the model is constructed so `artifacts` is already populated.

        Checks:
          1. No NaN or Inf values in any feature.
          2. All required feature names are present in the input.
        """
        feature_names = artifacts.get("feature_names", [])
        raw = self.features

        # Reject NaN / Inf
        bad_values = [k for k, v in raw.items() if not np.isfinite(v)]
        if bad_values:
            raise ValueError(
                f"Non-finite values (NaN/Inf) found in features: {bad_values[:5]}"
                + ("..." if len(bad_values) > 5 else "")
            )

        # Check all required features are present
        if feature_names:
            missing = [f for f in feature_names if f not in raw]
            if missing:
                raise ValueError(
                    f"{len(missing)} required feature(s) missing. "
                    f"First 5 missing: {missing[:5]}"
                )

        return self


class PredictResponse(BaseModel):
    """
    What the API returns after a successful prediction.

      prediction    — the predicted mental health profile (e.g. "depression")
      confidence    — probability assigned to the predicted class (0.0 to 1.0)
      probabilities — full probability distribution across all 6 classes
      model         — name of the model that made the prediction ("XGBoost")
      accuracy      — the model's test-set accuracy from training (0.6033)
    """
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    model: str
    accuracy: float


# ─── Preprocessing ────────────────────────────────────────────────────────────

def apply_outlier_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply per-column outlier smoothing using the transformers saved during training.

    During training, each column was tested with 4 strategies:
      - winsorize    → clamp to [lower, upper] percentile bounds
      - sqrt         → square-root compress large values (shift non-negative first)
      - yeo-johnson  → power transform for both positive and negative values
      - log1p        → log(1+x) compression

    The best strategy per column is stored in outlier_transformers.joblib.
    """
    transformers = artifacts["outlier_transformers"]
    df = df.copy()

    for col, info in transformers.items():
        if col not in df.columns:
            continue
        strategy = info["strategy"]

        if strategy == "yeo-johnson":
            pt = info["fitted_pt"]
            df[col] = pt.transform(df[[col]].values).ravel()

        elif strategy == "sqrt":
            # Clip negatives to 0 then sqrt — matches training pipeline exactly
            df[col] = np.sqrt(df[col].clip(lower=0))

        elif strategy == "log1p":
            # Clip negatives to 0 then log1p — matches training pipeline exactly
            df[col] = np.log1p(df[col].clip(lower=0))

        elif strategy == "winsorize":
            lower = info["lower"]
            upper = info["upper"]
            df[col] = df[col].clip(lower=lower, upper=upper)

    return df


def preprocess(raw_features: dict) -> np.ndarray:
    """
    Full preprocessing pipeline for a single inference sample.
    Mirrors training pipeline steps exactly.

    Steps:
      1. Wrap the feature dict in a single-row DataFrame
      2. Apply per-column outlier smoothing
      3. Scale with RobustScaler (fit on training data)
      4. Select and reorder to exactly 1,351 features the model expects

    Returns a (1, 1351) numpy array ready for model.predict_proba().
    """
    feature_names = artifacts["feature_names"]
    df = pd.DataFrame([raw_features])

    # Apply outlier smoothing
    df = apply_outlier_transforms(df)

    # Scale and reorder features to match training order exactly
    scaler = artifacts["scaler"]
    return scaler.transform(df[feature_names].values)


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", dependencies=[Security(verify_api_key)])
def root():
    """
    Service info endpoint — returns a summary of the running model.
    No authentication required.
    """
    meta = artifacts.get("metadata", {})
    return {
        "service": "Mindspace Mental Health Classifier — Voice",
        "model": meta.get("best_model_name"),
        "accuracy": meta.get("test_metrics", {}).get("accuracy"),
        "classes": meta.get("class_names"),
        "n_features": meta.get("n_features"),
    }


@app.get("/health", dependencies=[Security(verify_api_key)])
def health():
    """
    Health check endpoint — confirms all artifacts are loaded and server is ready.
    Returns { "status": "ok", "artifacts_loaded": true } when healthy.
    """
    expected_keys = {"model", "scaler", "label_encoder", "encoding", "outlier_transformers", "feature_names", "metadata"}
    return {"status": "ok", "artifacts_loaded": expected_keys.issubset(artifacts.keys())}


@app.post("/predict", response_model=PredictResponse, dependencies=[Security(verify_api_key)])
def predict(request: PredictRequest):
    """
    Main prediction endpoint.

    Accepts 1,351 pre-extracted OpenSMILE acoustic features as a flat JSON object
    under the key "features" and returns:
      - prediction    → the most likely mental health profile
      - confidence    → probability of the predicted class (0–1)
      - probabilities → full distribution across all 6 classes
      - model / accuracy → metadata about the model

    Request body format:
    {
        "features": {
            "pcm_fftMag_fband250-650_sma_de_pctlrange0-1": 0.0042,
            "mfcc_sma_de[11]_meanPeakDist": 12.34,
            ... (all 1,351 features)
        }
    }
    """
    # ── Preprocessing ────────────────────────────────────────────────────────
    try:
        X = preprocess(request.features)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Preprocessing failed: {e}")

    # ── Inference ────────────────────────────────────────────────────────────
    try:
        model = artifacts["model"]
        le    = artifacts["label_encoder"]
        meta  = artifacts["metadata"]

        proba      = model.predict_proba(X)[0]
        pred_idx   = int(np.argmax(proba))
        pred_label = le.inverse_transform([pred_idx])[0]
        confidence = float(proba[pred_idx])

        class_names   = le.classes_.tolist()
        probabilities = {cls: round(float(p), 4) for cls, p in zip(class_names, proba)}

        return PredictResponse(
            prediction=pred_label,
            confidence=round(confidence, 4),
            probabilities=probabilities,
            model=meta.get("best_model_name", "XGBoost"),
            accuracy=meta.get("test_metrics", {}).get("accuracy", 0.0),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.get("/model/info", dependencies=[Security(verify_api_key)])
def model_info():
    """
    Full model metadata endpoint — returns everything saved about the trained model.
    Includes hyperparameters, CV score, test-set metrics, and the full list of
    1,351 required feature names.
    """
    return artifacts.get("metadata", {})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_voice_to_sentiment:app", host="0.0.0.0", port=9100, reload=True)

# To run locally (from inside deployment-voice/):
# uvicorn api_voice_to_sentiment:app --reload --port 9100
# Then open http://localhost:9100/docs for the interactive Swagger UI.