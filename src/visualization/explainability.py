"""Model explainability using SHAP."""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from typing import Optional, List
from ..utils.logger import get_logger

logger = get_logger()


class ModelExplainer:
    """Explain fraud detection model predictions using SHAP."""
    
    def __init__(self, model, feature_names: List[str]):
        """Initialize model explainer.
        
        Args:
            model: Trained model with predict method
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        
        logger.info("Initialized ModelExplainer")
    
    def fit_explainer(self, X: np.ndarray, background_samples: int = 100):
        """Fit SHAP explainer on background data.
        
        Args:
            X: Background data for SHAP
            background_samples: Number of background samples to use
        """
        logger.info(f"Fitting SHAP explainer with {background_samples} background samples...")
        
        # Sample background data if needed
        if len(X) > background_samples:
            background_idx = np.random.choice(len(X), background_samples, replace=False)
            background_data = X[background_idx]
        else:
            background_data = X
        
        # Create SHAP explainer
        try:
            # Try KernelExplainer (works with any model)
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.model.predict,
                background_data
            )
            logger.info("SHAP KernelExplainer fitted successfully")
        except Exception as e:
            logger.warning(f"Failed to create KernelExplainer: {e}")
            logger.info("Attempting TreeExplainer...")
            
            try:
                # Try TreeExplainer for tree-based models
                self.explainer = shap.TreeExplainer(self.model.model)
                logger.info("SHAP TreeExplainer fitted successfully")
            except Exception as e2:
                logger.error(f"Failed to create explainer: {e2}")
                raise
    
    def explain_predictions(self, X: np.ndarray, max_samples: int = 100):
        """Calculate SHAP values for predictions.
        
        Args:
            X: Data to explain
            max_samples: Maximum number of samples to explain
        """
        if self.explainer is None:
            raise ValueError("Explainer not fitted. Call fit_explainer first.")
        
        logger.info(f"Calculating SHAP values for {min(len(X), max_samples)} samples...")
        
        # Limit samples for computational efficiency
        if len(X) > max_samples:
            explain_idx = np.random.choice(len(X), max_samples, replace=False)
            X_explain = X[explain_idx]
        else:
            X_explain = X
        
        # Calculate SHAP values
        self.shap_values = self.explainer.shap_values(X_explain)
        
        logger.info("SHAP values calculated successfully")
    
    def plot_summary(self, save_path: Optional[str] = None, max_display: int = 20):
        """Plot SHAP summary plot.
        
        Args:
            save_path: Path to save the plot
            max_display: Maximum number of features to display
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated. Call explain_predictions first.")
        
        logger.info("Creating SHAP summary plot...")
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            self.shap_values,
            features=None,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"SHAP summary plot saved to {save_path}")
        else:
            plt.tight_layout()
        
        plt.close()
    
    def plot_feature_importance(self, save_path: Optional[str] = None, max_display: int = 20):
        """Plot SHAP feature importance.
        
        Args:
            save_path: Path to save the plot
            max_display: Maximum number of features to display
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated. Call explain_predictions first.")
        
        logger.info("Creating SHAP feature importance plot...")
        
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        
        # Create DataFrame for plotting
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': mean_abs_shap
        }).sort_values('importance', ascending=False).head(max_display)
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(importance_df)), importance_df['importance'])
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('Mean |SHAP value|', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.title('Feature Importance (SHAP)', fontsize=14)
        plt.gca().invert_yaxis()
        plt.grid(alpha=0.3, axis='x')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        else:
            plt.tight_layout()
        
        plt.close()
    
    def explain_single_prediction(
        self,
        X: np.ndarray,
        index: int = 0,
        save_path: Optional[str] = None
    ):
        """Explain a single prediction with force plot.
        
        Args:
            X: Data containing the prediction to explain
            index: Index of the prediction to explain
            save_path: Path to save the plot
        """
        if self.explainer is None:
            raise ValueError("Explainer not fitted. Call fit_explainer first.")
        
        logger.info(f"Explaining prediction at index {index}...")
        
        # Calculate SHAP values for single prediction
        shap_values_single = self.explainer.shap_values(X[index:index+1])
        
        # Create force plot
        shap.force_plot(
            self.explainer.expected_value,
            shap_values_single[0] if isinstance(shap_values_single, list) else shap_values_single[0],
            X[index],
            feature_names=self.feature_names,
            matplotlib=True,
            show=False
        )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Force plot saved to {save_path}")
        else:
            plt.tight_layout()
        
        plt.close()
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance as DataFrame.
        
        Returns:
            DataFrame with feature names and importance scores
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated. Call explain_predictions first.")
        
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': mean_abs_shap
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def explain_transaction(
        self,
        X: np.ndarray,
        prediction: float,
        top_n: int = 5
    ) -> Dict[str, any]:
        """Explain why a transaction was flagged as fraud.
        
        Args:
            X: Transaction features (single sample)
            prediction: Fraud probability/score
            top_n: Number of top features to return
            
        Returns:
            Dictionary with explanation
        """
        if self.explainer is None:
            raise ValueError("Explainer not fitted. Call fit_explainer first.")
        
        # Calculate SHAP values for this transaction
        shap_values_single = self.explainer.shap_values(X.reshape(1, -1))
        
        if isinstance(shap_values_single, list):
            shap_values_single = shap_values_single[0]
        else:
            shap_values_single = shap_values_single[0]
        
        # Get top contributing features
        feature_contributions = []
        for i, (feature_name, shap_val) in enumerate(zip(self.feature_names, shap_values_single)):
            feature_contributions.append({
                'feature': feature_name,
                'value': float(X[i]),
                'shap_value': float(shap_val),
                'contribution': 'increases' if shap_val > 0 else 'decreases'
            })
        
        # Sort by absolute SHAP value
        feature_contributions.sort(key=lambda x: abs(x['shap_value']), reverse=True)
        
        explanation = {
            'fraud_probability': float(prediction),
            'risk_level': 'high' if prediction > 0.7 else 'medium' if prediction > 0.4 else 'low',
            'top_contributing_features': feature_contributions[:top_n]
        }
        
        return explanation
