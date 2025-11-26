"""Autoencoder model for anomaly detection."""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from typing import Tuple, List, Optional
import joblib
from ..utils.logger import get_logger

logger = get_logger()


class AutoencoderDetector:
    """Autoencoder-based anomaly detector for fraud detection."""
    
    def __init__(
        self,
        input_dim: int,
        encoding_dim: int = 8,
        hidden_dims: List[int] = [32, 16],
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001
    ):
        """Initialize Autoencoder detector.
        
        Args:
            input_dim: Number of input features
            encoding_dim: Dimension of encoded representation
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
        """
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        self.model = self._build_model()
        self.threshold = None
        
        logger.info(f"Initialized AutoencoderDetector with input_dim={input_dim}, encoding_dim={encoding_dim}")
    
    def _build_model(self) -> Model:
        """Build autoencoder model architecture.
        
        Returns:
            Keras Model
        """
        # Input layer
        input_layer = layers.Input(shape=(self.input_dim,))
        
        # Encoder
        encoded = input_layer
        for dim in self.hidden_dims:
            encoded = layers.Dense(dim, activation='relu')(encoded)
            encoded = layers.Dropout(self.dropout_rate)(encoded)
        
        # Bottleneck
        encoded = layers.Dense(self.encoding_dim, activation='relu', name='encoding')(encoded)
        
        # Decoder
        decoded = encoded
        for dim in reversed(self.hidden_dims):
            decoded = layers.Dense(dim, activation='relu')(decoded)
            decoded = layers.Dropout(self.dropout_rate)(decoded)
        
        # Output layer
        decoded = layers.Dense(self.input_dim, activation='linear')(decoded)
        
        # Create model
        autoencoder = Model(input_layer, decoded)
        
        # Compile model
        autoencoder.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        logger.info("Autoencoder model built successfully")
        return autoencoder
    
    def fit(
        self,
        X: np.ndarray,
        epochs: int = 50,
        batch_size: int = 256,
        validation_split: float = 0.2,
        verbose: int = 1
    ) -> 'AutoencoderDetector':
        """Fit the autoencoder model.
        
        Args:
            X: Training features (normal transactions only for best results)
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
            verbose: Verbosity mode
            
        Returns:
            Self
        """
        logger.info(f"Training Autoencoder on {X.shape[0]} samples with {X.shape[1]} features")
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            X, X,  # Autoencoder reconstructs input
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose
        )
        
        # Calculate threshold based on training data reconstruction error
        train_predictions = self.model.predict(X, verbose=0)
        train_mse = np.mean(np.square(X - train_predictions), axis=1)
        
        # Set threshold as mean + 2*std of reconstruction error
        self.threshold = np.mean(train_mse) + 2 * np.std(train_mse)
        
        logger.info(f"Autoencoder training completed. Threshold set to {self.threshold:.4f}")
        return self
    
    def predict(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """Predict anomalies.
        
        Args:
            X: Features to predict
            threshold: Custom threshold for anomaly detection. If None, uses fitted threshold.
            
        Returns:
            Binary predictions (1 for fraud, 0 for normal)
        """
        reconstruction_errors = self.get_reconstruction_error(X)
        
        if threshold is None:
            threshold = self.threshold
        
        if threshold is None:
            raise ValueError("Threshold not set. Either fit the model or provide a threshold.")
        
        return (reconstruction_errors > threshold).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly scores based on reconstruction error.
        
        Args:
            X: Features to predict
            
        Returns:
            Anomaly scores (normalized reconstruction errors)
        """
        reconstruction_errors = self.get_reconstruction_error(X)
        
        # Normalize scores to [0, 1] range using sigmoid
        scores = 1 / (1 + np.exp(-reconstruction_errors))
        
        return scores
    
    def get_reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Calculate reconstruction error for input data.
        
        Args:
            X: Input features
            
        Returns:
            Reconstruction errors (MSE per sample)
        """
        reconstructions = self.model.predict(X, verbose=0)
        mse = np.mean(np.square(X - reconstructions), axis=1)
        return mse
    
    def save(self, model_path: str, threshold_path: str) -> None:
        """Save model and threshold to disk.
        
        Args:
            model_path: Path to save the Keras model
            threshold_path: Path to save the threshold
        """
        self.model.save(model_path)
        joblib.dump({'threshold': self.threshold}, threshold_path)
        logger.info(f"Model saved to {model_path}, threshold saved to {threshold_path}")
    
    def load(self, model_path: str, threshold_path: str) -> None:
        """Load model and threshold from disk.
        
        Args:
            model_path: Path to load the Keras model from
            threshold_path: Path to load the threshold from
        """
        self.model = keras.models.load_model(model_path)
        threshold_data = joblib.load(threshold_path)
        self.threshold = threshold_data['threshold']
        logger.info(f"Model loaded from {model_path}, threshold loaded from {threshold_path}")
    
    def get_params(self) -> dict:
        """Get model parameters.
        
        Returns:
            Dictionary of model parameters
        """
        return {
            'input_dim': self.input_dim,
            'encoding_dim': self.encoding_dim,
            'hidden_dims': self.hidden_dims,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'threshold': self.threshold
        }
