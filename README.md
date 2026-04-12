# Deployment Voice API

FastAPI inference service for the Mindspace voice-based mental health classifier. The service loads a trained XGBoost pipeline at startup and exposes HTTP endpoints for health checks, model metadata, and predictions from pre-extracted acoustic features.

## What This Repo Contains

- `api_voice_to_sentiment.py`: FastAPI app and inference pipeline.
- `pipeline_output/XGBoost_27032026_152209/`: trained model and preprocessing artifacts.
- `demo-api-input-data-sample/voice_normal_sample_1.json`: sample request payload.
- `Dockerfile`: container image definition.
- `docker-compose.yml`: local container orchestration.

## Model Summary

- Model: XGBoost
- Task: 6-class classification
- Classes: `anxiety`, `bipolar`, `depression`, `normal`, `stress`, `suicidal`
- Input size: 1,351 acoustic features
- Reported test accuracy: `0.6033`

The API expects features that have already been extracted upstream, for example from an OpenSMILE-based pipeline. This repo does not perform raw audio feature extraction.

## Requirements

- Python 3.11 if running locally
- A `.env` file in the project root containing:

```env
API_KEY=your-secret-key
```

You can copy `.env.example` to `.env` and then set your key.

## Run Locally

Install dependencies and start the API:

```powershell
pip install -r requirements.txt
uvicorn api_voice_to_sentiment:app --host 0.0.0.0 --port 9100 --reload
```

Docs will be available at `http://localhost:9100/docs`.

## Run With Docker

Build and run with Compose:

```powershell
docker compose up --build
```

Stop the service:

```powershell
docker compose down
```

The API is exposed on port `9100`.

## API Surface

- `GET /`: basic service metadata, requires `X-API-Key`
- `GET /health`: readiness check, requires `X-API-Key`
- `POST /predict`: prediction endpoint, requires `X-API-Key`
- `GET /model/info`: full model metadata, requires `X-API-Key`

## Endpoint Details

### `GET /`

Requires header:

```http
X-API-Key: your-secret-key
```

Returns basic runtime metadata for the deployed service.

Example response:

```json
{
  "service": "Mindspace Mental Health Classifier — Voice",
  "model": "XGBoost",
  "accuracy": 0.6033,
  "classes": ["anxiety", "bipolar", "depression", "normal", "stress", "suicidal"],
  "n_features": 1351
}
```

### `GET /health`

Requires header:

```http
X-API-Key: your-secret-key
```

Returns service readiness and whether model artifacts were loaded successfully.

Example response:

```json
{
  "status": "ok",
  "artifacts_loaded": true
}
```

### `POST /predict`

Requires header:

```http
X-API-Key: your-secret-key
```

Request body:

```json
{
  "features": {
    "feature_name_1": 0.123,
    "feature_name_2": -0.456
  }
}
```

Response body:

```json
{
  "prediction": "normal",
  "confidence": 0.8123,
  "probabilities": {
    "anxiety": 0.0312,
    "bipolar": 0.0441,
    "depression": 0.0527,
    "normal": 0.8123,
    "stress": 0.0419,
    "suicidal": 0.0178
  },
  "model": "XGBoost",
  "accuracy": 0.6033
}
```

### `GET /model/info`

Requires the same `X-API-Key` header. Returns model metadata loaded from the saved training artifacts, including:

- model name
- tuned hyperparameters
- cross-validation score
- test metrics
- class names
- required feature names

## Request Format

`POST /predict` expects a JSON body shaped like:

```json
{
  "features": {
    "feature_name_1": 0.123,
    "feature_name_2": -0.456
  }
}
```

All 1,351 required feature names must be present. The easiest way to test the API is to reuse the sample payload in `demo-api-input-data-sample/voice_normal_sample_1.json`.

Example request:

```powershell
curl -X POST http://localhost:9100/predict ^
  -H "Content-Type: application/json" ^
  -H "X-API-Key: your-secret-key" ^
  --data @demo-api-input-data-sample/voice_normal_sample_1.json
```

## Notes For Developers

- Artifacts are loaded once at startup through the FastAPI lifespan hook.
- Preprocessing includes saved outlier transforms and a saved scaler before inference.
- CORS is currently open to all origins and should be tightened for production deployment.
- The sample payload is also injected into the Swagger schema when available.