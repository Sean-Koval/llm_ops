FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Install the package
RUN pip install -e .

# Create directories for data and models
RUN mkdir -p /app/data/raw /app/data/processed /app/models /app/outputs

# Create a dummy model directory structure for testing
RUN mkdir -p /app/models/dummy-model
# Create a minimal config.json in the dummy model directory
RUN echo '{"model_type": "bert", "architectures": ["BertForSequenceClassification"]}' > /app/models/dummy-model/config.json

# Set environment variables
ENV PYTHONPATH=/app
ENV MODEL_PATH="/app/models/dummy-model"
ENV LOG_LEVEL="INFO"
ENV SKIP_MODEL_LOAD="true"

# Expose port for API
EXPOSE 8000

# Set entrypoint (using minimal server for testing)
ENTRYPOINT ["python", "scripts/minimal_server.py"]

# Default arguments (for start_server.py, not used with minimal_server.py)
# CMD ["--model_path", "/app/models/dummy-model", "--host", "0.0.0.0", "--port", "8000", "--skip_model_load", "true"]