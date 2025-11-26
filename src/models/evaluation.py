"""Model evaluation and metrics."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report,
    f1_score, precision_score, recall_score, accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Any, Optional
from ..utils.logger import get_logger

logger = get_logger()


class ModelEvaluator:
    """Evaluate fraud detection models."""
    
    def __init__(self):
        """Initialize model evaluator."""
        self.metrics = {}
        logger.info("Initialized ModelEvaluator")
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_scores: np.ndarray,
        model_name: str = "Model"
    ) -> Dict[str, Any]:
        """Evaluate model performance.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_scores: Prediction scores/probabilities
            model_name: Name of the model
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating {model_name}...")
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else 0.0
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics.update({
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp),
                'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
                'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0.0
            })
        
        self.metrics[model_name] = metrics
        
        logger.info(f"{model_name} Evaluation Results:")
        logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
        
        return metrics
    
    def find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        metric: str = 'f1',
        threshold_range: Tuple[float, float] = (0.5, 0.99),
        steps: int = 50
    ) -> Tuple[float, Dict[str, float]]:
        """Find optimal threshold for classification.
        
        Args:
            y_true: True labels
            y_scores: Prediction scores
            metric: Metric to optimize ('f1', 'precision', 'recall')
            threshold_range: Range of thresholds to test
            steps: Number of threshold values to test
            
        Returns:
            Tuple of (optimal_threshold, metrics_at_optimal)
        """
        logger.info(f"Finding optimal threshold to maximize {metric}...")
        
        thresholds = np.linspace(threshold_range[0], threshold_range[1], steps)
        best_threshold = thresholds[0]
        best_score = 0
        best_metrics = {}
        
        for threshold in thresholds:
            y_pred = (y_scores >= threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == 'precision':
                score = precision_score(y_true, y_pred, zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_true, y_pred, zero_division=0)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
                best_metrics = {
                    'threshold': threshold,
                    'f1_score': f1_score(y_true, y_pred, zero_division=0),
                    'precision': precision_score(y_true, y_pred, zero_division=0),
                    'recall': recall_score(y_true, y_pred, zero_division=0),
                    'accuracy': accuracy_score(y_true, y_pred)
                }
        
        logger.info(f"Optimal threshold: {best_threshold:.4f}")
        logger.info(f"  F1-Score: {best_metrics['f1_score']:.4f}")
        logger.info(f"  Precision: {best_metrics['precision']:.4f}")
        logger.info(f"  Recall: {best_metrics['recall']:.4f}")
        
        return best_threshold, best_metrics
    
    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        model_name: str = "Model",
        save_path: Optional[str] = None
    ) -> None:
        """Plot ROC curve.
        
        Args:
            y_true: True labels
            y_scores: Prediction scores
            model_name: Name of the model
            save_path: Path to save the plot
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        auc = roc_auc_score(y_true, y_scores)
        
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.4f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curve - {model_name}', fontsize=14)
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {save_path}")
        else:
            plt.tight_layout()
        
        plt.close()
    
    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        model_name: str = "Model",
        save_path: Optional[str] = None
    ) -> None:
        """Plot Precision-Recall curve.
        
        Args:
            y_true: True labels
            y_scores: Prediction scores
            model_name: Name of the model
            save_path: Path to save the plot
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        
        plt.figure(figsize=(10, 6))
        plt.plot(recall, precision, label=model_name, linewidth=2)
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'Precision-Recall Curve - {model_name}', fontsize=14)
        plt.legend(loc="upper right", fontsize=10)
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Precision-Recall curve saved to {save_path}")
        else:
            plt.tight_layout()
        
        plt.close()
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model",
        save_path: Optional[str] = None
    ) -> None:
        """Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
            save_path: Path to save the plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Fraud'],
            yticklabels=['Normal', 'Fraud'],
            cbar_kws={'label': 'Count'}
        )
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.title(f'Confusion Matrix - {model_name}', fontsize=14)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        else:
            plt.tight_layout()
        
        plt.close()
    
    def plot_threshold_analysis(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        threshold_range: Tuple[float, float] = (0.5, 0.99),
        steps: int = 50,
        save_path: Optional[str] = None
    ) -> None:
        """Plot metrics across different thresholds.
        
        Args:
            y_true: True labels
            y_scores: Prediction scores
            threshold_range: Range of thresholds to test
            steps: Number of threshold values to test
            save_path: Path to save the plot
        """
        thresholds = np.linspace(threshold_range[0], threshold_range[1], steps)
        f1_scores = []
        precisions = []
        recalls = []
        
        for threshold in thresholds:
            y_pred = (y_scores >= threshold).astype(int)
            f1_scores.append(f1_score(y_true, y_pred, zero_division=0))
            precisions.append(precision_score(y_true, y_pred, zero_division=0))
            recalls.append(recall_score(y_true, y_pred, zero_division=0))
        
        plt.figure(figsize=(12, 6))
        plt.plot(thresholds, f1_scores, label='F1-Score', linewidth=2)
        plt.plot(thresholds, precisions, label='Precision', linewidth=2)
        plt.plot(thresholds, recalls, label='Recall', linewidth=2)
        plt.xlabel('Threshold', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title('Metrics vs. Threshold', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Threshold analysis plot saved to {save_path}")
        else:
            plt.tight_layout()
        
        plt.close()
    
    def generate_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "Model"
    ) -> str:
        """Generate classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
            
        Returns:
            Classification report as string
        """
        report = f"\n{'='*60}\n"
        report += f"Classification Report - {model_name}\n"
        report += f"{'='*60}\n"
        report += classification_report(
            y_true, y_pred,
            target_names=['Normal', 'Fraud'],
            zero_division=0
        )
        report += f"{'='*60}\n"
        
        logger.info(report)
        return report
    
    def compare_models(self, save_path: Optional[str] = None) -> pd.DataFrame:
        """Compare metrics across multiple models.
        
        Args:
            save_path: Path to save the comparison table
            
        Returns:
            DataFrame with model comparison
        """
        if not self.metrics:
            logger.warning("No models to compare")
            return pd.DataFrame()
        
        comparison_df = pd.DataFrame(self.metrics).T
        
        logger.info("\nModel Comparison:")
        logger.info(f"\n{comparison_df.to_string()}")
        
        if save_path:
            comparison_df.to_csv(save_path)
            logger.info(f"Model comparison saved to {save_path}")
        
        return comparison_df
