"""Isolation Forest model for anomaly detection."""

import numpy as np
from sklearn.ensemble import IsolationForest
from typing import Dict, Any, Optional
import joblib
from ..utils.logger import get_logger

logger = get_logger()


class IsolationForestDetector:
    """Isolation Forest anomaly detector for fraud detection."""
    
    def __init__(
        self,
        contamination: float = 0.02,
        n_estimators: int = 100,
        max_samples: int = 256,
        random_state: int = 42,
        n_jobs: int = -1
    ):
        """Initialize Isolation Forest detector.
        
        Args:
            contamination: Expected proportion of outliers in the dataset
            n_estimators: Number of base estimators
            max_samples: Number of samples to draw for each estimator
            random_state: Random seed
            n_jobs: Number of parallel jobs
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=random_state,
            n_jobs=n_jobs
        )
        
        logger.info(f"Initialized IsolationForestDetector with contamination={contamination}")
    
    def fit(self, X: np.ndarray) -> 'IsolationForestDetector':
        """Fit the Isolation Forest model.
        
        Args:
            X: Training features
            
        Returns:
            Self
        """
        logger.info(f"Training Isolation Forest on {X.shape[0]} samples with {X.shape[1]} features")
        self.model.fit(X)
        logger.info("Isolation Forest training completed")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies.
        
        Args:
            X: Features to predict
            
        Returns:
            Binary predictions (1 for fraud, 0 for normal)
        """
        # IsolationForest returns -1 for outliers, 1 for inliers
        predictions = self.model.predict(X)
        # Convert to 0/1 (1 for fraud)
        return (predictions == -1).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly scores (probability-like).
        
        Args:
            X: Features to predict
            
        Returns:
            Anomaly scores (higher = more anomalous)
        """
        # Get anomaly scores (negative values for outliers)
        scores = self.model.score_samples(X)
        
        # Convert to probability-like scores (0 to 1, higher = more anomalous)
        # Normalize scores to [0, 1] range
        scores_normalized = 1 / (1 + np.exp(scores))  # Sigmoid transformation
        
        return scores_normalized
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Get decision function scores.
        
        Args:
            X: Features
            
        Returns:
            Decision scores
        """
        return self.model.decision_function(X)
    
    def save(self, filepath: str) -> None:
        """Save model to disk.
        
        Args:
            filepath: Path to save the model
        """
        # Save the entire detector object (not just self.model)
        joblib.dump(self, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load model from disk.
        
        Args:
            filepath: Path to load the model from
        """
        self.model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters.
        
        Returns:
            Dictionary of model parameters
        """
        return {
            'contamination': self.contamination,
            'n_estimators': self.n_estimators,
            'max_samples': self.max_samples,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs
        }
