# =============================================================================
# Mindspace — Voice API (XGBoost, port 9100)
# =============================================================================
# Build context: deployment-voice/
# docker build -t mindspace-voice-api .
# docker run --env-file .env -p 9100:9100 mindspace-voice-api
# =============================================================================

FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer-cached unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the API source and all required artifacts
COPY api_voice_to_sentiment.py .
COPY pipeline_output/ pipeline_output/
COPY demo-api-input-data-sample/ demo-api-input-data-sample/

# API_KEY is injected at runtime via --env-file or -e, not baked into the image
EXPOSE 9100

CMD ["uvicorn", "api_voice_to_sentiment:app", "--host", "0.0.0.0", "--port", "9100"]
