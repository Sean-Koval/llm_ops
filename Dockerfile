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

# Set environment variables
ENV PYTHONPATH=/app
ENV MODEL_PATH="/app/models"
ENV LOG_LEVEL="INFO"

# Expose port for API
EXPOSE 8000

# Set entrypoint
ENTRYPOINT ["python", "scripts/start_server.py"]

# Default arguments
CMD ["--model_path", "/app/models", "--host", "0.0.0.0", "--port", "8000"]