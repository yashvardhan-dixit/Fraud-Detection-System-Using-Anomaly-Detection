"""Data preprocessing and feature engineering."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, List, Dict
from ..utils.logger import get_logger

logger = get_logger()


class FeatureEngineer:
    """Feature engineering for fraud detection."""
    
    def __init__(self):
        """Initialize feature engineer."""
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        logger.info("Initialized FeatureEngineer")
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features.
        
        Args:
            df: Input DataFrame with 'timestamp' column
            
        Returns:
            DataFrame with additional time features
        """
        df = df.copy()
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Extract time features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
        
        # Sort by timestamp for sequential features
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Time difference from previous transaction (in seconds)
        df['time_diff'] = df.groupby('user_id')['timestamp'].diff().dt.total_seconds()
        df['time_diff'] = df['time_diff'].fillna(df['time_diff'].median())
        
        logger.info("Created time-based features")
        return df
    
    def create_aggregation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create aggregation features per user.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with additional aggregation features
        """
        df = df.copy()
        
        # User-level aggregations
        user_stats = df.groupby('user_id').agg({
            'amount': ['mean', 'std', 'min', 'max', 'count'],
            'transaction_type': lambda x: x.mode()[0] if len(x) > 0 and len(x.mode()) > 0 else 'unknown'
        }).reset_index()
        
        user_stats.columns = [
            'user_id', 'avg_amount_user', 'std_amount_user', 
            'min_amount_user', 'max_amount_user', 'transaction_count_user',
            'most_common_type'
        ]
        
        df = df.merge(user_stats, on='user_id', how='left')
        
        # Fill NaN values
        df['std_amount_user'] = df['std_amount_user'].fillna(0)
        
        logger.info("Created aggregation features")
        return df
    
    def create_velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create velocity features (transactions per time window).
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with velocity features
        """
        df = df.copy()
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Initialize columns
        df['velocity_1h'] = 0
        df['velocity_24h'] = 0
        
        # Efficient vectorized approach using groupby and rolling
        for user in df['user_id'].unique():
            user_mask = df['user_id'] == user
            user_indices = df[user_mask].index
            user_times = df.loc[user_mask, 'timestamp']
            
            # For each transaction, count previous transactions in time windows
            for idx in user_indices:
                current_time = df.loc[idx, 'timestamp']
                
                # Vectorized comparison for this user's transactions
                time_diffs = (current_time - user_times).dt.total_seconds()
                
                # Count transactions in windows (excluding current)
                df.loc[idx, 'velocity_1h'] = ((time_diffs > 0) & (time_diffs <= 3600)).sum()
                df.loc[idx, 'velocity_24h'] = ((time_diffs > 0) & (time_diffs <= 86400)).sum()
        
        logger.info("Created velocity features")
        return df
    
    def create_amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create amount-based features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with amount features
        """
        df = df.copy()
        
        # Z-score of amount (overall)
        df['amount_zscore'] = (df['amount'] - df['amount'].mean()) / df['amount'].std()
        
        # Z-score per user (deviation from user's typical spending)
        df['amount_zscore_user'] = (df['amount'] - df['avg_amount_user']) / (df['std_amount_user'] + 1e-6)
        
        # Amount ratio to user average
        df['amount_ratio_user_avg'] = df['amount'] / (df['avg_amount_user'] + 1e-6)
        
        logger.info("Created amount-based features")
        return df
    
    def fit_transform(self, df: pd.DataFrame, target_col: str = 'is_fraud') -> Tuple[np.ndarray, np.ndarray]:
        """Fit feature engineering pipeline and transform data.
        
        Args:
            df: Input DataFrame
            target_col: Name of target column
            
        Returns:
            Tuple of (features, target)
        """
        logger.info("Fitting and transforming features...")
        
        # Create all features
        df = self.create_time_features(df)
        df = self.create_aggregation_features(df)
        df = self.create_velocity_features(df)
        df = self.create_amount_features(df)
        
        # Select feature columns
        feature_cols = [
            'amount', 'hour', 'day_of_week', 'is_weekend', 'is_night',
            'time_diff', 'merchant_risk_score',
            'avg_amount_user', 'std_amount_user', 'transaction_count_user',
            'velocity_1h', 'velocity_24h',
            'amount_zscore', 'amount_zscore_user', 'amount_ratio_user_avg'
        ]
        
        # Encode categorical features
        categorical_cols = ['transaction_type', 'location']
        for col in categorical_cols:
            if col in df.columns:
                self.encoders[col] = LabelEncoder()
                df[f'{col}_encoded'] = self.encoders[col].fit_transform(df[col])
                feature_cols.append(f'{col}_encoded')
        
        # Extract features
        X = df[feature_cols].values
        y = df[target_col].values if target_col in df.columns else None
        
        # Scale features
        self.scalers['standard'] = StandardScaler()
        X = self.scalers['standard'].fit_transform(X)
        
        self.feature_names = feature_cols
        logger.info(f"Created {len(feature_cols)} features")
        
        return X, y
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform data using fitted pipeline.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Transformed features
        """
        logger.info("Transforming features...")
        
        # Create all features
        df = self.create_time_features(df)
        df = self.create_aggregation_features(df)
        df = self.create_velocity_features(df)
        df = self.create_amount_features(df)
        
        # Encode categorical features
        for col, encoder in self.encoders.items():
            if col in df.columns:
                # Handle unseen categories
                df[f'{col}_encoded'] = df[col].apply(
                    lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
                )
        
        # Extract features
        X = df[self.feature_names].values
        
        # Scale features
        X = self.scalers['standard'].transform(X)
        
        return X
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names.
        
        Returns:
            List of feature names
        """
        return self.feature_names
