FROM python:3.11

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY api_voice_to_sentiment.py .
COPY pipeline_output/ pipeline_output/
COPY demo-api-input-data-sample/ demo-api-input-data-sample/

EXPOSE 9100

CMD ["uvicorn", "api_voice_to_sentiment:app", "--host", "0.0.0.0", "--port", "9100"]
