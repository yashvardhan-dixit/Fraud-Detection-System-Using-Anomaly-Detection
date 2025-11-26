"""Main training pipeline for fraud detection system."""

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

from src.utils.logger import setup_logger, get_logger
from src.utils.config import get_config
from src.data.generator import FraudDataGenerator
from src.data.preprocessing import FeatureEngineer
from src.models.isolation_forest import IsolationForestDetector
from src.models.autoencoder import AutoencoderDetector
from src.models.evaluation import ModelEvaluator
from src.visualization.explainability import ModelExplainer


def main(args):
    """Main training pipeline."""
    
    # Setup logging
    os.makedirs('logs', exist_ok=True)
    setup_logger(log_file='logs/training.log', level='INFO')
    logger = get_logger()
    
    logger.info("="*60)
    logger.info("Starting Fraud Detection System Training Pipeline")
    logger.info("="*60)
    
    # Load configuration
    config = get_config()
    
    # Create directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models/saved', exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)
    
    # 1. Generate or load data
    logger.info("\n[Step 1/7] Generating synthetic data...")
    
    if args.data_path and os.path.exists(args.data_path):
        logger.info(f"Loading data from {args.data_path}")
        df = pd.read_csv(args.data_path, parse_dates=['timestamp'])
    else:
        logger.info("Generating synthetic transaction data")
        generator = FraudDataGenerator(
            n_samples=config.get('data.sample_size', 10000),
            fraud_ratio=config.get('data.fraud_ratio', 0.02),
            random_state=config.get('data.random_state', 42)
        )
        df = generator.generate()
        
        # Save generated data
        data_path = 'data/raw/transactions.csv'
        df.to_csv(data_path, index=False)
        logger.info(f"Data saved to {data_path}")
    
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Fraud ratio: {df['is_fraud'].mean()*100:.2f}%")
    
    # 2. Feature engineering
    logger.info("\n[Step 2/7] Engineering features...")
    
    feature_engineer = FeatureEngineer()
    X, y = feature_engineer.fit_transform(df, target_col='is_fraud')
    
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Features: {feature_engineer.get_feature_names()}")
    
    # Save feature engineer
    joblib.dump(feature_engineer, 'models/saved/feature_engineer.joblib')
    logger.info("Feature engineer saved")
    
    # 3. Split data
    logger.info("\n[Step 3/7] Splitting data...")
    
    # Create indices to track the original dataframe rows
    indices = np.arange(len(X))
    
    # Split using arrays=(X, y, indices) so we get train/test for each
    split_result = train_test_split(
        X, y, indices,
        test_size=config.get('data.test_size', 0.3),
        random_state=config.get('data.random_state', 42),
        stratify=y
    )
    
    # Unpack: train_test_split returns (X_train, X_test, y_train, y_test, idx_train, idx_test)
    X_train, X_test, y_train, y_test, train_idx, test_idx = split_result
    
    logger.info(f"Training set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    
    # Save a subset of test data for demo dashboard
    # Note: fraud_score will be added after model training
    demo_df = df.iloc[test_idx].copy()
    demo_df.to_csv('data/processed/demo_data_raw.csv', index=False)
    
    # 4. Train Isolation Forest
    logger.info("\n[Step 4/7] Training Isolation Forest...")
    
    iso_forest = IsolationForestDetector(
        contamination=config.get('models.isolation_forest.contamination', 0.02),
        n_estimators=config.get('models.isolation_forest.n_estimators', 100),
        max_samples=config.get('models.isolation_forest.max_samples', 256),
        random_state=config.get('models.isolation_forest.random_state', 42),
        n_jobs=config.get('models.isolation_forest.n_jobs', -1)
    )
    
    # Train on normal transactions for better anomaly detection
    normal_mask = y_train == 0
    iso_forest.fit(X_train[normal_mask])
    
    # Save model
    iso_forest.save('models/saved/isolation_forest.joblib')
    logger.info("Isolation Forest saved")
    
    # 5. Train Autoencoder
    logger.info("\n[Step 5/7] Training Autoencoder...")
    
    autoencoder = AutoencoderDetector(
        input_dim=X_train.shape[1],
        encoding_dim=config.get('models.autoencoder.encoding_dim', 8),
        hidden_dims=config.get('models.autoencoder.hidden_dims', [32, 16]),
        dropout_rate=config.get('models.autoencoder.dropout_rate', 0.2),
        learning_rate=config.get('models.autoencoder.learning_rate', 0.001)
    )
    
    # Train on normal transactions
    autoencoder.fit(
        X_train[normal_mask],
        epochs=config.get('models.autoencoder.epochs', 50),
        batch_size=config.get('models.autoencoder.batch_size', 256),
        validation_split=config.get('models.autoencoder.validation_split', 0.2),
        verbose=1
    )
    
    # Save model
    autoencoder.save(
        'models/saved/autoencoder.h5',
        'models/saved/autoencoder_threshold.joblib'
    )
    logger.info("Autoencoder saved")
    
    # 6. Evaluate models
    logger.info("\n[Step 6/7] Evaluating models...")
    
    evaluator = ModelEvaluator()
    
    # Evaluate Isolation Forest
    iso_pred = iso_forest.predict(X_test)
    iso_scores = iso_forest.predict_proba(X_test)
    
    iso_metrics = evaluator.evaluate(
        y_test, iso_pred, iso_scores,
        model_name="Isolation Forest"
    )
    
    # Evaluate Autoencoder
    ae_pred = autoencoder.predict(X_test)
    ae_scores = autoencoder.predict_proba(X_test)
    
    ae_metrics = evaluator.evaluate(
        y_test, ae_pred, ae_scores,
        model_name="Autoencoder"
    )
    
    # Find optimal threshold for Isolation Forest
    optimal_threshold, threshold_metrics = evaluator.find_optimal_threshold(
        y_test, iso_scores,
        metric='f1',
        threshold_range=tuple(config.get('evaluation.threshold_range', [0.5, 0.99])),
        steps=config.get('evaluation.threshold_steps', 50)
    )
    
    # Generate plots
    logger.info("Generating evaluation plots...")
    
    evaluator.plot_roc_curve(
        y_test, iso_scores,
        model_name="Isolation Forest",
        save_path='results/plots/roc_curve_isolation_forest.png'
    )
    
    evaluator.plot_roc_curve(
        y_test, ae_scores,
        model_name="Autoencoder",
        save_path='results/plots/roc_curve_autoencoder.png'
    )
    
    evaluator.plot_confusion_matrix(
        y_test, iso_pred,
        model_name="Isolation Forest",
        save_path='results/plots/confusion_matrix_isolation_forest.png'
    )
    
    evaluator.plot_threshold_analysis(
        y_test, iso_scores,
        threshold_range=tuple(config.get('evaluation.threshold_range', [0.5, 0.99])),
        steps=config.get('evaluation.threshold_steps', 50),
        save_path='results/plots/threshold_analysis.png'
    )
    
    # Compare models
    comparison_df = evaluator.compare_models(save_path='results/model_comparison.csv')
    
    # Create demo data with actual fraud scores from model
    logger.info("Creating demo data with fraud scores...")
    demo_df = df.iloc[test_idx].copy()
    demo_df['fraud_score'] = iso_scores
    demo_df.to_csv('data/processed/demo_data.csv', index=False)
    logger.info("Demo data saved with actual fraud scores")
    
    # 7. Generate explainability
    logger.info("\n[Step 7/7] Generating model explanations...")
    
    try:
        explainer = ModelExplainer(
            iso_forest,
            feature_engineer.get_feature_names()
        )
        
        # Fit explainer on sample of training data
        explainer.fit_explainer(X_train[:500], background_samples=100)
        
        # Explain test predictions
        explainer.explain_predictions(X_test[:100], max_samples=100)
        
        # Generate plots
        explainer.plot_feature_importance(
            save_path='results/plots/feature_importance.png',
            max_display=15
        )
        
        # Get feature importance
        importance_df = explainer.get_feature_importance()
        importance_df.to_csv('results/feature_importance.csv', index=False)
        
        logger.info("Explainability analysis completed")
    except Exception as e:
        logger.warning(f"Could not generate explainability plots: {e}")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("Training Pipeline Completed Successfully!")
    logger.info("="*60)
    logger.info("\nModel Performance Summary:")
    logger.info(f"\nIsolation Forest:")
    logger.info(f"  ROC-AUC: {iso_metrics['roc_auc']:.4f}")
    logger.info(f"  Precision: {iso_metrics['precision']:.4f}")
    logger.info(f"  Recall: {iso_metrics['recall']:.4f}")
    logger.info(f"  F1-Score: {iso_metrics['f1_score']:.4f}")
    logger.info(f"\nAutoencoder:")
    logger.info(f"  ROC-AUC: {ae_metrics['roc_auc']:.4f}")
    logger.info(f"  Precision: {ae_metrics['precision']:.4f}")
    logger.info(f"  Recall: {ae_metrics['recall']:.4f}")
    logger.info(f"  F1-Score: {ae_metrics['f1_score']:.4f}")
    logger.info(f"\nOptimal Threshold: {optimal_threshold:.4f}")
    logger.info(f"  F1-Score at optimal: {threshold_metrics['f1_score']:.4f}")
    logger.info("\nSaved artifacts:")
    logger.info("  - Models: models/saved/")
    logger.info("  - Plots: results/plots/")
    logger.info("  - Data: data/")
    logger.info("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train fraud detection models")
    parser.add_argument(
        '--data-path',
        type=str,
        default=None,
        help='Path to existing transaction data CSV'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    main(args)
