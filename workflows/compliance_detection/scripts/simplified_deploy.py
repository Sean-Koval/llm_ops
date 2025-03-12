#!/usr/bin/env python
"""
Simplified deployment script for demonstration purposes.
This simulates deploying a FastAPI service for compliance detection.
"""

import json
import random
import time
from datetime import datetime
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

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

def load_model():
    """Load model metadata and prompt from the registry"""
    print("Loading model from registry...")
    
    # Load model metadata
    with open("workflows/compliance_detection/model_registry/model_metadata.json", 'r') as f:
        model_metadata = json.load(f)
    
    print(f"Loaded model: {model_metadata['name']} (v{model_metadata['version']})")
    print(f"F1 Score: {model_metadata['metrics']['f1_macro']:.4f}")
    
    return model_metadata

def simulate_deployment():
    """Simulate deploying the API service"""
    print("\nSimulating deployment of compliance detection API...")
    
    # Load model
    model_metadata = load_model()
    
    # Create FastAPI app (for demonstration only)
    app = FastAPI(
        title="Compliance Detection API",
        description="API for detecting compliance breaches in text messages",
        version=model_metadata["version"]
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Define prediction function
    def predict_compliance(text: str) -> Dict[str, Any]:
        """Simulate making a prediction using the loaded model"""
        # In a real implementation, this would call the Vertex AI API with the prompt
        
        # Simulate latency
        time.sleep(random.uniform(0.1, 0.3))
        
        # For demonstration, we'll generate a random prediction with high accuracy
        categories = ["ETHICAL_BREACH", "ILLEGAL_ACTIVITY", "REGULATORY_VIOLATION", 
                     "CONFIDENTIAL_INFO", "HARASSMENT", "COMPLIANT"]
        
        # Use some simple rules to make plausible predictions
        likely_category = "COMPLIANT"
        
        if "guarantee" in text.lower() or "hidden fees" in text.lower():
            likely_category = "ETHICAL_BREACH"
        elif "offshore" in text.lower() or "launder" in text.lower() or "manipulate" in text.lower():
            likely_category = "ILLEGAL_ACTIVITY"
        elif "earnings report" in text.lower() or "before it's released" in text.lower():
            likely_category = "REGULATORY_VIOLATION"
        elif "client list" in text.lower() or "account balance" in text.lower() or "SSN" in text.lower():
            likely_category = "CONFIDENTIAL_INFO"
        elif "attractive" in text.lower() or "she's only" in text.lower():
            likely_category = "HARASSMENT"
        
        # 95% of the time, return the likely category, otherwise random
        if random.random() < 0.95:
            prediction = likely_category
            confidence = random.uniform(0.85, 0.98)
        else:
            categories.remove(likely_category)
            prediction = random.choice(categories)
            confidence = random.uniform(0.6, 0.85)
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "latency_ms": random.uniform(150, 250)
        }
    
    # Define endpoints
    @app.get("/health")
    def health_check():
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}
    
    @app.post("/predict", response_model=PredictionResponse)
    async def predict(message: Message, request: Request):
        request_id = f"req-{int(time.time())}-{random.randint(1000, 9999)}"
        
        # Make prediction
        result = predict_compliance(message.text)
        
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
    
    @app.post("/predict/batch", response_model=BatchResponse)
    async def predict_batch(batch: BatchRequest, request: Request):
        request_id = f"batch-{int(time.time())}-{random.randint(1000, 9999)}"
        
        # Process each message
        predictions = []
        for i, message in enumerate(batch.messages):
            # Make prediction
            result = predict_compliance(message.text)
            
            # Create prediction response
            prediction = PredictionResponse(
                text=message.text,
                prediction=result["prediction"],
                confidence=result["confidence"],
                timestamp=datetime.now().isoformat(),
                request_id=f"{request_id}-{i}",
                metadata=message.metadata
            )
            
            predictions.append(prediction)
        
        # Create batch response
        response = BatchResponse(
            predictions=predictions,
            request_id=request_id,
            count=len(predictions),
            timestamp=datetime.now().isoformat()
        )
        
        return response
    
    @app.get("/model/info")
    def model_info():
        return {
            "name": model_metadata["name"],
            "version": model_metadata["version"],
            "metrics": model_metadata["metrics"],
            "created_at": model_metadata["created_at"],
            "categories": ["ETHICAL_BREACH", "ILLEGAL_ACTIVITY", "REGULATORY_VIOLATION", 
                           "CONFIDENTIAL_INFO", "HARASSMENT", "COMPLIANT"]
        }
    
    # Print deployment details
    print("\nAPI documentation would be available at http://localhost:8000/docs")
    print("\nEndpoints:")
    print("  GET  /health - Health check")
    print("  POST /predict - Predict a single message")
    print("  POST /predict/batch - Predict multiple messages")
    print("  GET  /model/info - Get model information")
    
    print("\nDeployment complete! Service is now running.")
    
    # In a real implementation, we would run the FastAPI app with uvicorn
    # uvicorn.run(app, host="0.0.0.0", port=8000)
    
    # For demonstration, show a sample prediction
    sample_text = "Here's the complete client list with their account balances and SSNs."
    prediction = predict_compliance(sample_text)
    
    print("\nSample prediction:")
    print(f"  Text: {sample_text}")
    print(f"  Prediction: {prediction['prediction']}")
    print(f"  Confidence: {prediction['confidence']:.4f}")
    print(f"  Latency: {prediction['latency_ms']:.2f} ms")
    
    print("\nModel monitoring metrics would be available in Prometheus/Grafana.")
    print("Request logs would be available in Cloud Logging.")
    
    return app

def simulate_monitoring_setup():
    """Simulate setting up monitoring for the API"""
    print("\nSetting up monitoring for compliance detection API...")
    print("  1. Configuring Prometheus metrics for request rate, latency, and error rate")
    print("  2. Setting up Grafana dashboards for real-time monitoring")
    print("  3. Configuring alerting for error spikes and latency increases")
    print("  4. Setting up prediction drift detection")
    
    # In a real implementation, this would configure Prometheus and Grafana

def main():
    print("Starting deployment process for compliance detector model...")
    
    # Simulate deployment
    app = simulate_deployment()
    
    # Simulate monitoring setup
    simulate_monitoring_setup()
    
    print("\nDeployment workflow complete!")

if __name__ == "__main__":
    main()