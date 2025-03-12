# Compliance Detection Model Deployment Guide

This guide provides detailed instructions for deploying the compliance detection model to production environments. It covers both local deployment for testing and cloud deployment for production use.

## Table of Contents
1. [Prerequisites](#1-prerequisites)
2. [Local Deployment](#2-local-deployment)
3. [Docker Deployment](#3-docker-deployment)
4. [Kubernetes Deployment](#4-kubernetes-deployment)
5. [Cloud Deployment (GCP)](#5-cloud-deployment-gcp)
6. [Monitoring Setup](#6-monitoring-setup)
7. [Scaling Considerations](#7-scaling-considerations)
8. [Troubleshooting](#8-troubleshooting)
9. [Security Considerations](#9-security-considerations)

## 1. Prerequisites

Before deploying the compliance detection model, ensure you have:

- Access to Google Cloud Project with Vertex AI enabled
- Service account with appropriate permissions
- Docker installed for container deployments
- kubectl and gcloud CLI for Kubernetes deployments
- Python 3.9+ for local development

**Required Environment Variables:**
```bash
# Set environment variables for deployment
export PROJECT_ID="your-gcp-project-id"
export REGION="us-central1"
export SERVICE_ACCOUNT="compliance-detector@${PROJECT_ID}.iam.gserviceaccount.com"
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
```

## 2. Local Deployment

For development and testing, you can run the API server locally:

```bash
# Install dependencies
pip install -e .

# Run the API server locally
python workflows/compliance_detection/scripts/deploy_api.py \
  --config workflows/compliance_detection/configs/compliance_config.yaml \
  --host 0.0.0.0 \
  --port 8000 \
  --metrics-port 8001

# Test the API
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Here is the quarterly report that was published this morning.", "metadata": {"source": "email"}}'
```

For local development with all services:

```bash
# Start MLflow, Prometheus, and Grafana locally
docker-compose -f docker-compose-dev.yml up -d

# Access services:
# - API: http://localhost:8000/docs
# - MLflow: http://localhost:5000
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000
```

## 3. Docker Deployment

For containerized deployment:

```bash
# Build the Docker image
docker build -t gcr.io/${PROJECT_ID}/compliance-detector:latest -f Dockerfile .

# Run locally with Docker
docker run -p 8000:8000 -p 8001:8001 \
  -v ${GOOGLE_APPLICATION_CREDENTIALS}:/tmp/credentials.json \
  -e GOOGLE_APPLICATION_CREDENTIALS=/tmp/credentials.json \
  gcr.io/${PROJECT_ID}/compliance-detector:latest

# Push to Google Container Registry
docker push gcr.io/${PROJECT_ID}/compliance-detector:latest
```

Using Docker Compose for multi-service deployment:

```bash
# Deploy all services with Docker Compose
docker-compose up -d

# Scale API servers if needed
docker-compose up -d --scale compliance-api=3
```

## 4. Kubernetes Deployment

For orchestrated deployment with Kubernetes:

```bash
# Update Kubernetes manifests with your project information
sed -i "s/PROJECT_ID/${PROJECT_ID}/g" k8s/api-deployment.yaml

# Deploy to Kubernetes
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/volumes.yaml
kubectl apply -f k8s/mlflow-deployment.yaml
kubectl apply -f k8s/monitoring-deployment.yaml
kubectl apply -f k8s/api-deployment.yaml

# Check deployment status
kubectl get pods -n compliance-detection

# Port-forward for local access
kubectl port-forward -n compliance-detection svc/compliance-api 8000:8000
```

For Minikube local testing:

```bash
# Start Minikube
minikube start --memory 4096 --cpus 2

# Deploy to Minikube
kubectl apply -f k8s/minikube/volumes.yaml
kubectl apply -f k8s/minikube/mlflow-deployment.yaml
kubectl apply -f k8s/minikube/monitoring-deployment.yaml
kubectl apply -f k8s/minikube/api-deployment.yaml

# Access services through Minikube
minikube service compliance-api -n compliance-detection
```

## 5. Cloud Deployment (GCP)

### Cloud Run Deployment

For serverless deployment with Cloud Run:

```bash
# Build and push with Cloud Build
gcloud builds submit --tag gcr.io/${PROJECT_ID}/compliance-detector

# Deploy to Cloud Run
gcloud run deploy compliance-detector \
  --image gcr.io/${PROJECT_ID}/compliance-detector \
  --platform managed \
  --region ${REGION} \
  --service-account ${SERVICE_ACCOUNT} \
  --memory 2Gi \
  --timeout 300s \
  --concurrency 80 \
  --set-env-vars="PROJECT_ID=${PROJECT_ID},MODEL_LOCATION=${REGION}"

# Get the service URL
gcloud run services describe compliance-detector \
  --platform managed \
  --region ${REGION} \
  --format 'value(status.url)'
```

### GKE Deployment

For production deployment with Google Kubernetes Engine:

```bash
# Create GKE cluster
gcloud container clusters create compliance-cluster \
  --zone ${REGION}-a \
  --num-nodes 3 \
  --machine-type e2-standard-4

# Configure kubectl to use GKE cluster
gcloud container clusters get-credentials compliance-cluster \
  --zone ${REGION}-a

# Deploy to GKE
kubectl apply -f k8s/gke/namespace.yaml
kubectl apply -f k8s/gke/volumes.yaml
kubectl apply -f k8s/gke/mlflow-deployment.yaml
kubectl apply -f k8s/gke/monitoring-deployment.yaml
kubectl apply -f k8s/gke/api-deployment.yaml

# Set up Ingress
kubectl apply -f k8s/gke/ingress.yaml
```

## 6. Monitoring Setup

### Prometheus Configuration

Configure Prometheus to scrape metrics from the API:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'compliance-api'
    scrape_interval: 15s
    static_configs:
      - targets: ['compliance-api:8001']
    metrics_path: /metrics
```

### Grafana Dashboards

Import dashboards for monitoring:

1. Navigate to Grafana (default: http://localhost:3000)
2. Go to Dashboards > Import
3. Either upload JSON from `monitoring/grafana/compliance_dashboard.json` or paste its contents
4. Select the Prometheus data source
5. Click Import

### Alerting

Configure alerts for critical metrics:

```yaml
# alert_rules.yml
groups:
- name: compliance-detector
  rules:
  - alert: HighErrorRate
    expr: rate(compliance_detection_errors_total[5m]) / rate(compliance_detection_requests_total[5m]) > 0.05
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High error rate detected"
      description: "Error rate is above 5% for 5 minutes"
      
  - alert: HighLatency
    expr: histogram_quantile(0.95, sum(rate(compliance_detection_latency_ms_bucket[5m])) by (le)) > 500
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High API latency detected"
      description: "95th percentile latency is above 500ms for 5 minutes"

  - alert: PredictionDrift
    expr: compliance_detection_drift_score > 0.1
    for: 30m
    labels:
      severity: warning
    annotations:
      summary: "Prediction distribution drift detected"
      description: "Distribution drift score above threshold for 30 minutes"
```

## 7. Scaling Considerations

### Vertical Scaling

Adjust resource allocation based on load:

```yaml
# For Kubernetes deployment (in api-deployment.yaml)
resources:
  requests:
    memory: "1Gi"
    cpu: "500m"
  limits:
    memory: "2Gi"
    cpu: "1000m"
```

### Horizontal Scaling

Configure autoscaling for variable load:

```yaml
# For Kubernetes deployment (in api-deployment.yaml)
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: compliance-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: compliance-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Caching Strategy

Implement caching for frequent messages:

```python
# In deploy_api.py

from functools import lru_cache
import hashlib

# Cache function for predictions (using LRU cache)
@lru_cache(maxsize=1000)
def cached_predict(text_hash):
    # Retrieve original text from hash map (implementation-dependent)
    text = text_hash_map.get(text_hash)
    if not text:
        return None
    
    # Make actual prediction
    return predictor.predict(text)

# In prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(message: Message, request: Request):
    # Create a hash of the message text
    text_hash = hashlib.md5(message.text.encode()).hexdigest()
    
    # Store in hash map if needed
    text_hash_map[text_hash] = message.text
    
    # Check cache
    cached_result = cached_predict(text_hash)
    if cached_result:
        # Log cache hit
        logger.info(f"Cache hit for request: {request_id}")
    else:
        # Make prediction
        result = predictor.predict(message.text)
        # Rest of the function...
```

## 8. Troubleshooting

### Common Issues and Solutions

**Issue: API container fails to start**
```bash
# Check logs
kubectl logs deployment/compliance-api -n compliance-detection

# Common fix: Check credentials
kubectl create secret generic vertex-credentials \
  --from-file=key.json=${GOOGLE_APPLICATION_CREDENTIALS} \
  -n compliance-detection
```

**Issue: Model loading errors**
```bash
# Check if model registry is accessible
kubectl exec -it $(kubectl get pods -l app=compliance-api -n compliance-detection -o jsonpath='{.items[0].metadata.name}') -n compliance-detection -- ls -la /app/workflows/compliance_detection/model_registry

# Fix: Update model registry path in config
kubectl create configmap compliance-config \
  --from-file=compliance_config.yaml=workflows/compliance_detection/configs/compliance_config.yaml \
  -n compliance-detection
```

**Issue: High latency or timeouts**
```bash
# Check resource usage
kubectl top pods -n compliance-detection

# Fix: Increase resource limits
kubectl edit deployment compliance-api -n compliance-detection
# Update resources section
```

### Diagnostic Commands

```bash
# Test API connectivity
kubectl run -it --rm --restart=Never curl-test --image=curlimages/curl -- curl compliance-api:8000/health

# Check Prometheus metrics
kubectl port-forward svc/prometheus 9090:9090 -n compliance-detection
# Then visit http://localhost:9090

# View detailed API logs
kubectl logs -f -l app=compliance-api -n compliance-detection

# List all routes in the API
curl http://localhost:8000/openapi.json | jq '.paths | keys'
```

## 9. Security Considerations

### Authentication

Add authentication to the API:

```python
# In deploy_api.py
from fastapi.security import APIKeyHeader
from fastapi import Security, HTTPException, status

# Define API key header
API_KEY = os.environ.get("API_KEY", "your-secret-key")
api_key_header = APIKeyHeader(name="X-API-Key")

# Dependency for API key validation
def get_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key"
        )
    return api_key

# Add to endpoints
@app.post("/predict", response_model=PredictionResponse)
async def predict(message: Message, request: Request, api_key: str = Security(get_api_key)):
    # Function implementation...
```

### Rate Limiting

Add rate limiting to prevent abuse:

```python
# In deploy_api.py
from fastapi import Request, HTTPException
import time
from datetime import datetime, timedelta
from collections import defaultdict

# Simple rate limiter
class RateLimiter:
    def __init__(self, requests_per_minute=60):
        self.requests_per_minute = requests_per_minute
        self.request_history = defaultdict(list)
    
    def check_limit(self, client_ip: str) -> bool:
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        
        # Remove old requests
        self.request_history[client_ip] = [
            ts for ts in self.request_history[client_ip] 
            if ts > minute_ago
        ]
        
        # Check current rate
        if len(self.request_history[client_ip]) >= self.requests_per_minute:
            return False
        
        # Add current request
        self.request_history[client_ip].append(now)
        return True

# Create limiter
rate_limiter = RateLimiter(requests_per_minute=60)

# Add middleware for rate limiting
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host
    
    if not rate_limiter.check_limit(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )
    
    response = await call_next(request)
    return response
```

### Data Protection

Best practices for protecting sensitive data:

1. **Log Sanitization**: Don't log full message contents
2. **Data Encryption**: Encrypt data at rest and in transit
3. **Minimal Information**: Return only necessary information in API responses
4. **PII Protection**: Implement PII detection and masking

Example implementation:

```python
# In deploy_api.py

def sanitize_for_logging(text: str) -> str:
    """Sanitize text for logging to remove sensitive information"""
    # Truncate long messages
    if len(text) > 50:
        return text[:50] + "..."
    
    return text

# In predict function
logger.info(f"Received prediction request: {request_id}, text: {sanitize_for_logging(message.text)}")
```

---

## Appendix: Production Checklist

Before going to production, ensure:

- [ ] All secrets are stored securely (not in code)
- [ ] Authentication is implemented
- [ ] Rate limiting is in place
- [ ] Monitoring and alerting are configured
- [ ] Error handling is robust
- [ ] Logging is appropriate (not excessive, not too minimal)
- [ ] Proper resource limits are set
- [ ] Scaling strategy is defined
- [ ] Rollback plan is documented
- [ ] Security review has been conducted

This checklist ensures a smooth deployment and operation of the compliance detection system in production environments.