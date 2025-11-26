"""Tests for fraud detection models."""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.isolation_forest import IsolationForestDetector
from src.models.autoencoder import AutoencoderDetector
from src.data.preprocessing import FeatureEngineer
from src.data.generator import FraudDataGenerator


class TestIsolationForest:
    """Test Isolation Forest model."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = IsolationForestDetector(contamination=0.1)
        assert model.contamination == 0.1
        assert model.model is not None
    
    def test_fit_predict(self):
        """Test fitting and prediction."""
        # Generate sample data
        X = np.random.randn(100, 10)
        
        # Initialize and fit model
        model = IsolationForestDetector(contamination=0.1)
        model.fit(X)
        
        # Make predictions
        predictions = model.predict(X)
        
        assert predictions.shape == (100,)
        assert set(predictions).issubset({0, 1})
        assert predictions.sum() > 0  # Should detect some anomalies
    
    def test_predict_proba(self):
        """Test probability predictions."""
        X = np.random.randn(100, 10)
        
        model = IsolationForestDetector(contamination=0.1)
        model.fit(X)
        
        scores = model.predict_proba(X)
        
        assert scores.shape == (100,)
        assert np.all(scores >= 0) and np.all(scores <= 1)


class TestAutoencoder:
    """Test Autoencoder model."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = AutoencoderDetector(input_dim=10, encoding_dim=5)
        assert model.input_dim == 10
        assert model.encoding_dim == 5
        assert model.model is not None
    
    def test_fit_predict(self):
        """Test fitting and prediction."""
        # Generate sample data
        X = np.random.randn(200, 10)
        
        # Initialize and fit model
        model = AutoencoderDetector(input_dim=10, encoding_dim=5)
        model.fit(X, epochs=2, batch_size=32, verbose=0)
        
        # Make predictions
        predictions = model.predict(X)
        
        assert predictions.shape == (200,)
        assert set(predictions).issubset({0, 1})
        assert model.threshold is not None
    
    def test_reconstruction_error(self):
        """Test reconstruction error calculation."""
        X = np.random.randn(100, 10)
        
        model = AutoencoderDetector(input_dim=10, encoding_dim=5)
        model.fit(X, epochs=2, verbose=0)
        
        errors = model.get_reconstruction_error(X)
        
        assert errors.shape == (100,)
        assert np.all(errors >= 0)


class TestFeatureEngineer:
    """Test feature engineering."""
    
    def test_time_features(self):
        """Test time feature creation."""
        import pandas as pd
        from datetime import datetime, timedelta
        
        # Create sample data
        df = pd.DataFrame({
            'timestamp': [datetime.now() - timedelta(hours=i) for i in range(10)],
            'user_id': ['USER_1'] * 10,
            'amount': np.random.rand(10) * 100
        })
        
        engineer = FeatureEngineer()
        result_df = engineer.create_time_features(df)
        
        assert 'hour' in result_df.columns
        assert 'day_of_week' in result_df.columns
        assert 'time_diff' in result_df.columns
    
    def test_fit_transform(self):
        """Test full feature engineering pipeline."""
        import pandas as pd
        from datetime import datetime, timedelta
        
        # Generate sample transaction data
        df = pd.DataFrame({
            'timestamp': [datetime.now() - timedelta(hours=i) for i in range(100)],
            'user_id': [f'USER_{i % 10}' for i in range(100)],
            'amount': np.random.rand(100) * 1000,
            'transaction_type': np.random.choice(['purchase', 'withdrawal'], 100),
            'location': np.random.choice(['domestic', 'international'], 100),
            'merchant_risk_score': np.random.rand(100) * 100,
            'is_fraud': np.random.choice([0, 1], 100, p=[0.9, 0.1])
        })
        
        engineer = FeatureEngineer()
        X, y = engineer.fit_transform(df)
        
        assert X.shape[0] == 100
        assert y.shape[0] == 100
        assert len(engineer.get_feature_names()) > 0


class TestDataGenerator:
    """Test data generation."""
    
    def test_generate_data(self):
        """Test synthetic data generation."""
        generator = FraudDataGenerator(n_samples=100, fraud_ratio=0.1)
        df = generator.generate()
        
        assert len(df) == 100
        assert 'is_fraud' in df.columns
        assert 'amount' in df.columns
        assert 'timestamp' in df.columns
        
        # Check fraud ratio
        fraud_count = df['is_fraud'].sum()
        assert 5 <= fraud_count <= 15  # Allow some variance
    
    def test_fraud_characteristics(self):
        """Test that fraudulent transactions have expected characteristics."""
        generator = FraudDataGenerator(n_samples=1000, fraud_ratio=0.1, random_state=42)
        df = generator.generate()
        
        fraud_df = df[df['is_fraud'] == 1]
        normal_df = df[df['is_fraud'] == 0]
        
        # Fraudulent transactions should have higher average amounts
        assert fraud_df['amount'].mean() > normal_df['amount'].mean()
        
        # Fraudulent transactions should have higher risk scores
        assert fraud_df['merchant_risk_score'].mean() > normal_df['merchant_risk_score'].mean()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
