# Mindspace — Voice Sentiment API

A FastAPI inference server that predicts a mental health profile from 1,351 pre-extracted acoustic voice features using a trained XGBoost model.

## What it does

Accepts 1,351 OpenSMILE acoustic features extracted from a speech/audio sample and returns the most likely mental health profile out of 6 classes: `anxiety`, `bipolar`, `depression`, `normal`, `stress`, `suicidal`.

## Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/` | No | Service info — name, supported classes, feature count |
| GET | `/health` | No | Health check — returns `{"status": "ok"}` when ready |
| POST | `/predict` | Yes | Run prediction on 1,351 acoustic features |
| GET | `/model/info` | Yes | Model structure info (feature names, classes, scaler) |

## Input format

The `/predict` endpoint expects a JSON object with a single `"features"` key containing all 1,351 acoustic feature names as a flat dict:

```json
{
  "features": {
    "pcm_fftMag_fband250-650_sma_de_pctlrange0-1": 6.524812,
    "mfcc_sma_de[11]_meanPeakDist": 0.229548,
    "...": "... (all 1,351 features required)"
  }
}
```

The full list of feature names is available at `GET /model/info` under `"feature_names"`. See `payload.json` for a complete sample input.

## How to use

### Prerequisites

```
pip install requests python-dotenv
```

### Call with Python `requests`

```python
import requests

url = "http://<host>:9100/predict"

payload = {
    "features": {
        "pcm_fftMag_fband250-650_sma_de_pctlrange0-1": 6.524812,
        "mfcc_sma_de[11]_meanPeakDist": 0.229548
        # ... all 1,351 features
    }
}

headers = {
    "X-API-Key": "your_api_key"
}

response = requests.post(url, json=payload, headers=headers)
print(response.json())
```

### Ready-to-run script

```bash
python test_predict_api.py
```

`test_predict_api.py` reads `payload.json`, calls `/predict`, and prints the result. Requires `.env` with `MINDSPACE_VOICE_API_KEY` set.

## Setup

### 1. Configure environment

```bash
cp example.env .env
# Edit .env and set MINDSPACE_VOICE_API_KEY to your key
```

### 2. Run with Docker (recommended)

```bash
docker compose up --build
```

### 3. Run locally

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 9100
```

Interactive docs available at `http://localhost:9100/docs`.

## File structure

```
deployment-voice/
├── main.py                    # FastAPI application
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker image definition
├── docker-compose.yml         # Single-service compose file
├── payload.json               # Sample input for /predict (1,351 features)
├── output.json                # Sample output from /predict
├── test_predict_api.py        # Ready-to-run test script
├── example.env                # Environment variable template
├── .env                       # Your actual keys (never commit this)
├── demo-api-input-data-sample/ # Sample input files per class
└── pipeline_output/           # Trained model artifacts
```
