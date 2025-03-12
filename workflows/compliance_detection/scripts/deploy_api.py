#!/usr/bin/env python
"""
Deploy the compliance detection API as a FastAPI application.
"""

import os
import json
import yaml
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import google.cloud.aiplatform as vertexai
import mlflow
from prometheus_client import Counter, Histogram, start_http_server

from llm_ops_pipeline.utils.logging import setup_logging
from llm_ops_pipeline.config.config import load_config
from workflows.compliance_detection.scripts.compliance_predictor import CompliancePredictor

# Setup argument parser
parser = argparse.ArgumentParser(description="Deploy compliance detection API")
parser.add_argument("--config", type=str, default="workflows/compliance_detection/configs/compliance_config.yaml", 
                    help="Path to configuration file")
parser.add_argument("--host", type=str, default="0.0.0.0",
                    help="Host to run the server on")
parser.add_argument("--port", type=int, default=8000,
                    help="Port to run the server on")
parser.add_argument("--metrics-port", type=int, default=8001,
                    help="Port for Prometheus metrics")
args = parser.parse_args()

# Load configuration
config = load_config(args.config)
logger = setup_logging(name="compliance_api", level=config.get("monitoring", {}).get("log_level", "INFO"))

# Initialize Prometheus metrics
prediction_counter = Counter(
    f"{config['monitoring']['metrics_prefix']}_predictions_total",
    "Total number of predictions",
    ["result"]
)
latency_histogram = Histogram(
    f"{config['monitoring']['metrics_prefix']}_latency_ms",
    "Prediction latency in milliseconds",
    buckets=[10, 50, 100, 200, 500, 1000, 2000, 5000]
)

# Initialize FastAPI
app = FastAPI(
    title="Compliance Detection API",
    description="API for detecting compliance breaches in text messages",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor
predictor = None

# Pydantic models
class Message(BaseModel):
    text: str = Field(..., description="Text message to classify")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Optional metadata")

class BatchRequest(BaseModel):
    messages: List[Message] = Field(..., description="List of messages to classify")

class PredictionResponse(BaseModel):
    text: str = Field(..., description="Original text")
    prediction: str = Field(..., description="Predicted category")
    confidence: float = Field(..., description="Confidence score")
    timestamp: str = Field(..., description="Timestamp of prediction")
    request_id: str = Field(..., description="Unique request ID")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Optional metadata")

class BatchResponse(BaseModel):
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    request_id: str = Field(..., description="Unique request ID")
    count: int = Field(..., description="Number of predictions")
    timestamp: str = Field(..., description="Timestamp of request")

# Initialization and shutdown events
@app.on_event("startup")
def startup_event():
    global predictor
    logger.info("Starting Compliance Detection API")
    
    # Start Prometheus metrics server
    start_http_server(args.metrics_port)
    logger.info(f"Prometheus metrics available at port {args.metrics_port}")
    
    # Initialize predictor
    predictor = CompliancePredictor(args.config)
    logger.info("Predictor initialized")

@app.on_event("shutdown")
def shutdown_event():
    logger.info("Shutting down Compliance Detection API")

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Single prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(message: Message, request: Request, background_tasks: BackgroundTasks):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    
    request_id = request.headers.get("X-Request-ID", datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    
    # Log request
    logger.info(f"Received prediction request: {request_id}")
    
    # Make prediction with timing
    with latency_histogram.time():
        result = predictor.predict(message.text, log_prediction=False)
    
    # Update Prometheus metrics
    prediction_counter.labels(result=result["prediction"]).inc()
    
    # Log prediction to MLflow in background
    def log_to_mlflow():
        try:
            with mlflow.start_run(run_name=f"api-{request_id}"):
                mlflow.log_param("request_id", request_id)
                mlflow.log_param("text_length", len(message.text))
                mlflow.log_metric("latency_ms", result["latency_ms"])
                mlflow.set_tag("prediction", result["prediction"])
                mlflow.set_tag("source", "api")
                
                # Log request and result
                with open("request.json", "w") as f:
                    json.dump({"text": message.text, "metadata": message.metadata}, f)
                with open("result.json", "w") as f:
                    json.dump(result, f)
                
                mlflow.log_artifact("request.json")
                mlflow.log_artifact("result.json")
        except Exception as e:
            logger.error(f"Error logging to MLflow: {str(e)}")
    
    # Run MLflow logging in background
    background_tasks.add_task(log_to_mlflow)
    
    # Prepare response
    response = PredictionResponse(
        text=message.text,
        prediction=result["prediction"],
        confidence=result["confidence"],
        timestamp=datetime.now().isoformat(),
        request_id=request_id,
        metadata=message.metadata
    )
    
    return response

# Batch prediction endpoint
@app.post("/predict/batch", response_model=BatchResponse)
async def predict_batch(batch: BatchRequest, request: Request, background_tasks: BackgroundTasks):
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    
    request_id = request.headers.get("X-Request-ID", datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    
    # Log request
    logger.info(f"Received batch prediction request: {request_id} with {len(batch.messages)} messages")
    
    # Extract texts
    texts = [msg.text for msg in batch.messages]
    
    # Make predictions with timing
    with latency_histogram.time():
        results = predictor.predict_batch(texts, log_predictions=False)
    
    # Update Prometheus metrics
    for result in results:
        prediction_counter.labels(result=result["prediction"]).inc()
    
    # Log batch to MLflow in background
    def log_batch_to_mlflow():
        try:
            with mlflow.start_run(run_name=f"api-batch-{request_id}"):
                mlflow.log_param("request_id", request_id)
                mlflow.log_param("batch_size", len(batch.messages))
                
                avg_latency = sum(r["latency_ms"] for r in results) / len(results)
                mlflow.log_metric("avg_latency_ms", avg_latency)
                
                # Log category counts
                category_counts = {cat: 0 for cat in config["data"]["categories"]}
                for result in results:
                    category_counts[result["prediction"]] += 1
                
                for cat, count in category_counts.items():
                    mlflow.log_metric(f"count_{cat}", count)
                
                # Log request and results
                with open("batch_request.json", "w") as f:
                    json.dump(batch.dict(), f)
                with open("batch_results.json", "w") as f:
                    json.dump(results, f)
                
                mlflow.log_artifact("batch_request.json")
                mlflow.log_artifact("batch_results.json")
                mlflow.set_tag("source", "api")
        except Exception as e:
            logger.error(f"Error logging batch to MLflow: {str(e)}")
    
    # Run MLflow logging in background
    background_tasks.add_task(log_batch_to_mlflow)
    
    # Prepare responses
    predictions = []
    for i, result in enumerate(results):
        prediction = PredictionResponse(
            text=batch.messages[i].text,
            prediction=result["prediction"],
            confidence=result["confidence"],
            timestamp=result["timestamp"],
            request_id=f"{request_id}-{i}",
            metadata=batch.messages[i].metadata
        )
        predictions.append(prediction)
    
    # Prepare batch response
    response = BatchResponse(
        predictions=predictions,
        request_id=request_id,
        count=len(predictions),
        timestamp=datetime.now().isoformat()
    )
    
    return response

# Model info endpoint
@app.get("/model/info")
def model_info():
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    
    return {
        "model_id": config["model"]["model_id"],
        "categories": config["data"]["categories"],
        "provider": config["model"]["provider"],
        "version": "1.0.0"
    }

# Run the server
if __name__ == "__main__":
    logger.info(f"Starting server at {args.host}:{args.port}")
    uvicorn.run("deploy_api:app", host=args.host, port=args.port, reload=False)