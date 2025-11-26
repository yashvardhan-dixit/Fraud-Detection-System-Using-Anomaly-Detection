"""FastAPI application for fraud detection predictions."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from datetime import datetime
import joblib
import os

from ..utils.logger import get_logger, setup_logger
from ..utils.config import get_config

# Setup logger
setup_logger(log_file="logs/api.log")
logger = get_logger()

# Initialize FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud detection using anomaly detection models",
    version="1.0.0"
)

# Global variables for models
isolation_forest_model = None
autoencoder_model = None
feature_engineer = None


class Transaction(BaseModel):
    """Transaction data model."""
    transaction_id: str = Field(..., description="Unique transaction ID")
    timestamp: str = Field(..., description="Transaction timestamp (ISO format)")
    user_id: str = Field(..., description="User ID")
    amount: float = Field(..., gt=0, description="Transaction amount")
    transaction_type: str = Field(..., description="Transaction type (purchase, withdrawal, transfer, payment)")
    location: str = Field(..., description="Transaction location (domestic, international)")
    merchant_risk_score: float = Field(..., ge=0, le=100, description="Merchant risk score (0-100)")
    
    class Config:
        schema_extra = {
            "example": {
                "transaction_id": "TXN_00001234",
                "timestamp": "2024-01-15T14:30:00",
                "user_id": "USER_000123",
                "amount": 150.50,
                "transaction_type": "purchase",
                "location": "domestic",
                "merchant_risk_score": 35.0
            }
        }


class PredictionResponse(BaseModel):
    """Prediction response model."""
    transaction_id: str
    is_fraud: bool
    fraud_probability: float
    model: str
    risk_level: str
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    models_loaded: Dict[str, bool]
    timestamp: str


@app.on_event("startup")
async def load_models():
    """Load models on startup."""
    global isolation_forest_model, autoencoder_model, feature_engineer
    
    logger.info("Loading models...")
    
    try:
        from ..models.isolation_forest import IsolationForestDetector
        
        # Load models (paths should be configured)
        models_dir = "models/saved"
        
        # Check if models exist
        if os.path.exists(f"{models_dir}/isolation_forest.joblib"):
            isolation_forest_model = IsolationForestDetector.load(f"{models_dir}/isolation_forest.joblib")
            logger.info("Isolation Forest model loaded")
        
        if os.path.exists(f"{models_dir}/feature_engineer.joblib"):
            feature_engineer = joblib.load(f"{models_dir}/feature_engineer.joblib")
            logger.info("Feature engineer loaded")
        
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Error loading models: {e}")


@app.get("/", tags=["General"])
async def root():
    """Root endpoint."""
    return {
        "message": "Fraud Detection API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "predict_batch": "/predict/batch"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        models_loaded={
            "isolation_forest": isolation_forest_model is not None,
            "autoencoder": autoencoder_model is not None,
            "feature_engineer": feature_engineer is not None
        },
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_fraud(transaction: Transaction):
    """Predict fraud for a single transaction.
    
    Args:
        transaction: Transaction data
        
    Returns:
        Prediction response with fraud probability and risk level
    """
    if isolation_forest_model is None or feature_engineer is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please train and save models first."
        )
    
    try:
        # Convert transaction to DataFrame
        transaction_dict = transaction.dict()
        df = pd.DataFrame([transaction_dict])
        
        # Engineer features
        X = feature_engineer.transform(df)
        
        # Make prediction using Isolation Forest
        prediction = isolation_forest_model.predict(X)[0]
        probability = isolation_forest_model.predict_proba(X)[0]
        
        # Determine risk level
        if probability < 0.3:
            risk_level = "low"
        elif probability < 0.6:
            risk_level = "medium"
        elif probability < 0.8:
            risk_level = "high"
        else:
            risk_level = "critical"
        
        return PredictionResponse(
            transaction_id=transaction.transaction_id,
            is_fraud=bool(prediction),
            fraud_probability=float(probability),
            model="isolation_forest",
            risk_level=risk_level,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", tags=["Prediction"])
async def predict_fraud_batch(transactions: List[Transaction]):
    """Predict fraud for multiple transactions.
    
    Args:
        transactions: List of transaction data
        
    Returns:
        List of prediction responses
    """
    if isolation_forest_model is None or feature_engineer is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please train and save models first."
        )
    
    try:
        # Convert transactions to DataFrame
        transactions_dict = [t.dict() for t in transactions]
        df = pd.DataFrame(transactions_dict)
        
        # Engineer features
        X = feature_engineer.transform(df)
        
        # Make predictions
        predictions = isolation_forest_model.predict(X)
        probabilities = isolation_forest_model.predict_proba(X)
        
        # Create responses
        responses = []
        for i, transaction in enumerate(transactions):
            probability = float(probabilities[i])
            
            # Determine risk level
            if probability < 0.3:
                risk_level = "low"
            elif probability < 0.6:
                risk_level = "medium"
            elif probability < 0.8:
                risk_level = "high"
            else:
                risk_level = "critical"
            
            responses.append(PredictionResponse(
                transaction_id=transaction.transaction_id,
                is_fraud=bool(predictions[i]),
                fraud_probability=probability,
                model="isolation_forest",
                risk_level=risk_level,
                timestamp=datetime.now().isoformat()
            ))
        
        return responses
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/models/info", tags=["Models"])
async def get_models_info():
    """Get information about loaded models."""
    if isolation_forest_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    return {
        "isolation_forest": {
            "loaded": isolation_forest_model is not None,
            "params": isolation_forest_model.get_params() if isolation_forest_model else None
        },
        "feature_engineer": {
            "loaded": feature_engineer is not None,
            "features": feature_engineer.get_feature_names() if feature_engineer else None
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    config = get_config()
    host = config.get("api.host", "0.0.0.0")
    port = config.get("api.port", 8000)
    
    uvicorn.run(app, host=host, port=port)
