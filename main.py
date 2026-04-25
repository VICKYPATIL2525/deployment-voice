import json
import os
import sys
import uuid
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, model_validator

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# ─── API Key auth ─────────────────────────────────────────────────────────────
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _get_api_key() -> str:
    return os.environ.get("MINDSPACE_VOICE_API_KEY", "")


def verify_api_key(key: str = Security(_api_key_header)) -> None:
    api_key = _get_api_key()
    if not api_key:
        raise HTTPException(status_code=500, detail={"error": "server_misconfiguration", "message": "MINDSPACE_VOICE_API_KEY is not set."})
    if key != api_key:
        raise HTTPException(status_code=403, detail={"error": "invalid_api_key", "message": "Invalid or missing API key. Pass it as X-API-Key header."})


# ─── Artifact paths ───────────────────────────────────────────────────────────
ARTIFACTS_DIR = Path(__file__).parent / "pipeline_output" / "XGBoost_27032026_152209"

artifacts: dict = {}


def load_artifacts() -> None:
    artifacts["model"]                = joblib.load(ARTIFACTS_DIR / "best_model.joblib")
    artifacts["scaler"]               = joblib.load(ARTIFACTS_DIR / "scaler.joblib")
    artifacts["label_encoder"]        = joblib.load(ARTIFACTS_DIR / "label_encoder.joblib")
    artifacts["encoding"]             = joblib.load(ARTIFACTS_DIR / "encoding_artifacts.joblib")
    artifacts["outlier_transformers"] = joblib.load(ARTIFACTS_DIR / "outlier_transformers.joblib")
    artifacts["feature_names"]        = json.loads((ARTIFACTS_DIR / "feature_names.json").read_text())
    artifacts["metadata"]             = json.loads((ARTIFACTS_DIR / "model_metadata.json").read_text())


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fail hard at startup — container is marked unhealthy immediately if artifacts missing.
    try:
        load_artifacts()
    except Exception as e:
        print(f"FATAL: Failed to load model artifacts — {e}", file=sys.stderr)
        sys.exit(1)

    if not _get_api_key():
        print("FATAL: MINDSPACE_VOICE_API_KEY is not set in the environment.", file=sys.stderr)
        sys.exit(1)

    # Inject a real Swagger example from the demo sample so /docs shows actual feature names.
    feature_names = artifacts.get("feature_names", [])
    example_features = {f: 0.0 for f in feature_names}
    demo_path = Path(__file__).parent / "demo-api-input-data-sample" / "voice_normal_sample_1.json"
    if demo_path.exists():
        demo_data = json.loads(demo_path.read_text())
        if "features" in demo_data:
            example_features = demo_data["features"]

    PredictRequest.model_config["json_schema_extra"] = {
        "examples": [{"features": example_features}]
    }
    PredictRequest.model_rebuild(force=True)

    yield


# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Mindspace Mental Health Classifier — Voice",
    description=(
        "Predicts mental health profile from 1,351 acoustic voice features (OpenSMILE). "
        "Output classes: anxiety, bipolar, depression, normal, stress, suicidal."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request / Response schemas ───────────────────────────────────────────────

class PredictRequest(BaseModel):
    """
    Send a flat JSON object under "features" where every key is one of the
    1,351 OpenSMILE acoustic feature names and every value is a float.

    The full list of required feature names is available at GET /model/info
    under the "feature_names" key.
    """
    model_config = {}  # Populated at startup with a real-data Swagger example

    features: Dict[str, float]

    @model_validator(mode="after")
    def validate_features(self) -> "PredictRequest":
        feature_names = artifacts.get("feature_names", [])
        raw = self.features

        bad_values = [k for k, v in raw.items() if not np.isfinite(v)]
        if bad_values:
            raise ValueError(
                f"Non-finite values (NaN/Inf) in features: {bad_values[:5]}"
                + ("..." if len(bad_values) > 5 else "")
            )

        if feature_names:
            missing = [f for f in feature_names if f not in raw]
            if missing:
                raise ValueError(
                    f"{len(missing)} required feature(s) missing. "
                    f"First 5: {missing[:5]}"
                )

        return self


class PredictResponse(BaseModel):
    prediction_id: str
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    model_name: str


# ─── Preprocessing ────────────────────────────────────────────────────────────

def apply_outlier_transforms(df: pd.DataFrame) -> pd.DataFrame:
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
            df[col] = np.sqrt(df[col].clip(lower=0))
        elif strategy == "log1p":
            df[col] = np.log1p(df[col].clip(lower=0))
        elif strategy == "winsorize":
            df[col] = df[col].clip(lower=info["lower"], upper=info["upper"])

    return df


def preprocess(raw_features: dict) -> np.ndarray:
    feature_names = artifacts["feature_names"]
    df = pd.DataFrame([raw_features])
    df = apply_outlier_transforms(df)
    scaler = artifacts["scaler"]
    return scaler.transform(df[feature_names].values)


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def root(_: None = Security(verify_api_key)):
    meta = artifacts.get("metadata", {})
    return {
        "service": "Mindspace Mental Health Classifier — Voice",
        "status": "running",
        "classes": meta.get("class_names"),
        "n_features": meta.get("n_features"),
    }


@app.get("/health")
def health():
    expected_keys = {"model", "scaler", "label_encoder", "encoding", "outlier_transformers", "feature_names", "metadata"}
    ready = expected_keys.issubset(artifacts.keys())
    if not ready:
        return JSONResponse(status_code=503, content={"status": "unavailable", "artifacts_loaded": len(artifacts)})
    return {
        "status": "ok",
        "model": artifacts.get("metadata", {}).get("best_model_name"),
        "artifacts_loaded": len(artifacts),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest, _: None = Security(verify_api_key)):
    try:
        X = preprocess(request.features)
    except Exception as e:
        raise HTTPException(status_code=422, detail={"error": "preprocessing_failed", "message": str(e)})

    try:
        model = artifacts["model"]
        le    = artifacts["label_encoder"]

        proba      = model.predict_proba(X)[0]
        pred_idx   = int(np.argmax(proba))
        pred_label = le.inverse_transform([pred_idx])[0]
        confidence = float(proba[pred_idx])

        class_names   = le.classes_.tolist()
        probabilities = {cls: round(float(p), 4) for cls, p in zip(class_names, proba)}

        return PredictResponse(
            prediction_id=str(uuid.uuid4()),
            prediction=pred_label,
            confidence=round(confidence, 4),
            probabilities=probabilities,
            model_name=artifacts.get("metadata", {}).get("best_model_name", "unknown"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": "prediction_failed", "message": str(e)})


@app.get("/model/info")
def model_info(_: None = Security(verify_api_key)):
    meta = artifacts.get("metadata", {})
    return {
        "model": meta.get("best_model_name"),
        "n_features": meta.get("n_features"),
        "feature_names": meta.get("feature_names"),
        "classes": meta.get("class_names"),
        "scaler": meta.get("scaler"),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=9100)
