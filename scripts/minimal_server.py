#!/usr/bin/env python
"""Minimal API server for testing."""

import uvicorn
from fastapi import FastAPI
from typing import Dict, Any
from pydantic import BaseModel

app = FastAPI(title="Minimal Test API")

class PredictionRequest(BaseModel):
    """Request model for prediction API."""
    text: str
    options: Dict[str, Any] = {}

class PredictionResponse(BaseModel):
    """Response model for prediction API."""
    prediction: str
    processing_time: float

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Hello from the LLM Ops Pipeline API"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Prediction endpoint."""
    return {
        "prediction": f"Dummy prediction for text: {request.text}",
        "processing_time": 0.01
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)