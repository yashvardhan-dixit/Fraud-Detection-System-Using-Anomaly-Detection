"""Data generator for synthetic fraud detection dataset."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple
from ..utils.logger import get_logger

logger = get_logger()


class FraudDataGenerator:
    """Generate synthetic fraud detection dataset."""
    
    def __init__(self, n_samples: int = 10000, fraud_ratio: float = 0.02, random_state: int = 42):
        """Initialize data generator.
        
        Args:
            n_samples: Number of transactions to generate
            fraud_ratio: Ratio of fraudulent transactions
            random_state: Random seed for reproducibility
        """
        self.n_samples = n_samples
        self.fraud_ratio = fraud_ratio
        self.random_state = random_state
        np.random.seed(random_state)
        
        logger.info(f"Initialized FraudDataGenerator with {n_samples} samples, {fraud_ratio*100}% fraud rate")
    
    def generate(self) -> pd.DataFrame:
        """Generate synthetic transaction data.
        
        Returns:
            DataFrame with synthetic transaction data
        """
        logger.info("Generating synthetic transaction data...")
        
        # Generate base features
        n_fraud = int(self.n_samples * self.fraud_ratio)
        n_normal = self.n_samples - n_fraud
        
        # Transaction IDs
        transaction_ids = [f"TXN_{i:08d}" for i in range(self.n_samples)]
        
        # Timestamps (last 30 days)
        start_date = datetime.now() - timedelta(days=30)
        timestamps = [
            start_date + timedelta(seconds=np.random.randint(0, 30*24*60*60))
            for _ in range(self.n_samples)
        ]
        timestamps = sorted(timestamps)
        
        # Normal transactions
        normal_amounts = np.random.lognormal(mean=3.5, sigma=1.2, size=n_normal)
        normal_amounts = np.clip(normal_amounts, 1, 1000)
        
        # Fraudulent transactions (typically higher amounts)
        fraud_amounts = np.random.lognormal(mean=5.0, sigma=1.5, size=n_fraud)
        fraud_amounts = np.clip(fraud_amounts, 100, 10000)
        
        # Combine and shuffle
        amounts = np.concatenate([normal_amounts, fraud_amounts])
        is_fraud = np.concatenate([np.zeros(n_normal), np.ones(n_fraud)])
        
        # Shuffle
        shuffle_idx = np.random.permutation(self.n_samples)
        amounts = amounts[shuffle_idx]
        is_fraud = is_fraud[shuffle_idx]
        
        # Transaction types
        transaction_types = np.random.choice(
            ['purchase', 'withdrawal', 'transfer', 'payment'],
            size=self.n_samples,
            p=[0.5, 0.2, 0.15, 0.15]
        )
        
        # Locations
        locations = np.random.choice(
            ['domestic', 'international'],
            size=self.n_samples,
            p=[0.85, 0.15]
        )
        # Fraud is more likely international
        locations[is_fraud == 1] = np.random.choice(
            ['domestic', 'international'],
            size=n_fraud,
            p=[0.3, 0.7]
        )
        
        # User IDs (some users have multiple transactions)
        n_users = self.n_samples // 5
        user_ids = [f"USER_{np.random.randint(0, n_users):06d}" for _ in range(self.n_samples)]
        
        # Merchant risk score (0-100, higher for fraud)
        merchant_risk = np.random.normal(30, 15, self.n_samples)
        merchant_risk[is_fraud == 1] = np.random.normal(70, 15, n_fraud)
        merchant_risk = np.clip(merchant_risk, 0, 100)
        
        # Create DataFrame
        df = pd.DataFrame({
            'transaction_id': transaction_ids,
            'timestamp': timestamps,
            'user_id': user_ids,
            'amount': amounts,
            'transaction_type': transaction_types,
            'location': locations,
            'merchant_risk_score': merchant_risk,
            'is_fraud': is_fraud.astype(int)
        })
        
        logger.info(f"Generated {len(df)} transactions with {df['is_fraud'].sum()} fraudulent cases")
        
        return df
    
    def save_data(self, df: pd.DataFrame, filepath: str) -> None:
        """Save generated data to CSV file.
        
        Args:
            df: DataFrame to save
            filepath: Path to save the data
        """
        df.to_csv(filepath, index=False)
        logger.info(f"Data saved to {filepath}")


def load_or_generate_data(filepath: str = None, n_samples: int = 10000, 
                          fraud_ratio: float = 0.02, random_state: int = 42) -> pd.DataFrame:
    """Load existing data or generate new data.
    
    Args:
        filepath: Path to existing data file. If None or doesn't exist, generates new data.
        n_samples: Number of samples to generate if creating new data
        fraud_ratio: Ratio of fraudulent transactions
        random_state: Random seed
        
    Returns:
        DataFrame with transaction data
    """
    import os
    
    if filepath and os.path.exists(filepath):
        logger.info(f"Loading existing data from {filepath}")
        return pd.read_csv(filepath, parse_dates=['timestamp'])
    else:
        logger.info("Generating new synthetic data")
        generator = FraudDataGenerator(n_samples, fraud_ratio, random_state)
        return generator.generate()
