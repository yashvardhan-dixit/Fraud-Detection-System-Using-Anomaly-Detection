"""Example usage of the Fraud Detection System."""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# Add project root to path for examples directory
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.generator import FraudDataGenerator
from src.data.preprocessing import FeatureEngineer
from src.models.isolation_forest import IsolationForestDetector
from src.models.autoencoder import AutoencoderDetector
from src.models.evaluation import ModelEvaluator
from src.utils.logger import setup_logger, get_logger

# Setup logging
setup_logger(level='INFO')
logger = get_logger()


def example_data_generation():
    """Example: Generate synthetic fraud data."""
    logger.info("Example 1: Generating synthetic fraud data")
    
    generator = FraudDataGenerator(n_samples=1000, fraud_ratio=0.05, random_state=42)
    df = generator.generate()
    
    logger.info(f"Generated {len(df)} transactions")
    logger.info(f"Fraud cases: {df['is_fraud'].sum()}")
    logger.info(f"Fraud ratio: {df['is_fraud'].mean()*100:.2f}%")
    logger.info(f"\nSample data:\n{df.head()}")
    
    return df


def example_feature_engineering(df):
    """Example: Feature engineering."""
    logger.info("\nExample 2: Feature engineering")
    
    engineer = FeatureEngineer()
    X, y = engineer.fit_transform(df)
    
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Number of features: {len(engineer.get_feature_names())}")
    logger.info(f"Features: {engineer.get_feature_names()}")
    
    return X, y, engineer


def example_train_isolation_forest(X, y):
    """Example: Train Isolation Forest."""
    logger.info("\nExample 3: Training Isolation Forest")
    
    # Train on normal transactions only
    normal_mask = y == 0
    X_normal = X[normal_mask]
    
    model = IsolationForestDetector(contamination=0.05, random_state=42)
    model.fit(X_normal)
    
    # Make predictions
    predictions = model.predict(X)
    scores = model.predict_proba(X)
    
    logger.info(f"Detected {predictions.sum()} anomalies")
    logger.info(f"Detection rate: {predictions.sum() / len(predictions) * 100:.2f}%")
    
    # Evaluate
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(y, predictions, scores, "Isolation Forest")
    
    logger.info(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1-Score: {metrics['f1_score']:.4f}")
    
    return model, metrics


def example_train_autoencoder(X, y):
    """Example: Train Autoencoder."""
    logger.info("\nExample 4: Training Autoencoder")
    
    # Train on normal transactions only
    normal_mask = y == 0
    X_normal = X[normal_mask]
    
    model = AutoencoderDetector(
        input_dim=X.shape[1],
        encoding_dim=8,
        hidden_dims=[32, 16],
        dropout_rate=0.2
    )
    
    model.fit(X_normal, epochs=10, batch_size=64, verbose=1)
    
    # Make predictions
    predictions = model.predict(X)
    scores = model.predict_proba(X)
    
    logger.info(f"Detected {predictions.sum()} anomalies")
    
    # Evaluate
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(y, predictions, scores, "Autoencoder")
    
    logger.info(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1-Score: {metrics['f1_score']:.4f}")
    
    return model, metrics


def example_threshold_tuning(y, scores):
    """Example: Threshold tuning."""
    logger.info("\nExample 5: Threshold tuning")
    
    evaluator = ModelEvaluator()
    optimal_threshold, metrics = evaluator.find_optimal_threshold(
        y, scores,
        metric='f1',
        threshold_range=(0.3, 0.9),
        steps=50
    )
    
    logger.info(f"Optimal threshold: {optimal_threshold:.4f}")
    logger.info(f"F1-Score at optimal: {metrics['f1_score']:.4f}")
    logger.info(f"Precision at optimal: {metrics['precision']:.4f}")
    logger.info(f"Recall at optimal: {metrics['recall']:.4f}")
    
    return optimal_threshold


def example_single_prediction(model, engineer):
    """Example: Make prediction for single transaction."""
    logger.info("\nExample 6: Single transaction prediction")
    
    # Create a sample transaction
    transaction = pd.DataFrame([{
        'transaction_id': 'TXN_EXAMPLE',
        'timestamp': datetime.now(),
        'user_id': 'USER_001',
        'amount': 2500.0,  # High amount (suspicious)
        'transaction_type': 'purchase',
        'location': 'international',  # International (more risky)
        'merchant_risk_score': 75.0,  # High risk score
        'is_fraud': 0  # Unknown
    }])
    
    # Engineer features
    X = engineer.transform(transaction)
    
    # Make prediction
    prediction = model.predict(X)[0]
    score = model.predict_proba(X)[0]
    
    logger.info("Transaction details:")
    logger.info(f"  Amount: ${transaction['amount'].iloc[0]:.2f}")
    logger.info(f"  Location: {transaction['location'].iloc[0]}")
    logger.info(f"  Risk score: {transaction['merchant_risk_score'].iloc[0]:.1f}")
    logger.info("\nPrediction:")
    logger.info(f"  Is fraud: {'Yes' if prediction == 1 else 'No'}")
    logger.info(f"  Fraud probability: {score:.4f}")
    logger.info(f"  Risk level: {'High' if score > 0.7 else 'Medium' if score > 0.4 else 'Low'}")


def main():
    """Run all examples."""
    logger.info("="*60)
    logger.info("Fraud Detection System - Examples")
    logger.info("="*60)
    
    # 1. Generate data
    df = example_data_generation()
    
    # 2. Feature engineering
    X, y, engineer = example_feature_engineering(df)
    
    # 3. Train Isolation Forest
    iso_model, iso_metrics = example_train_isolation_forest(X, y)
    
    # 4. Train Autoencoder (optional, takes longer)
    # ae_model, ae_metrics = example_train_autoencoder(X, y)
    
    # 5. Threshold tuning
    iso_scores = iso_model.predict_proba(X)
    optimal_threshold = example_threshold_tuning(y, iso_scores)
    
    # 6. Single prediction
    example_single_prediction(iso_model, engineer)
    
    logger.info("\n" + "="*60)
    logger.info("Examples completed!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
