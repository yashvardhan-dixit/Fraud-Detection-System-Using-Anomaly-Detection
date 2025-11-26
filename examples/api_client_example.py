"""Example API client for fraud detection system."""

import requests
import json
from datetime import datetime
from typing import Dict, List


class FraudDetectionClient:
    """Client for interacting with Fraud Detection API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize API client.
        
        Args:
            base_url: Base URL of the API
        """
        self.base_url = base_url
    
    def health_check(self) -> Dict:
        """Check API health status.
        
        Returns:
            Health status response
        """
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def predict_single(self, transaction: Dict) -> Dict:
        """Predict fraud for a single transaction.
        
        Args:
            transaction: Transaction data
            
        Returns:
            Prediction response
        """
        response = requests.post(
            f"{self.base_url}/predict",
            json=transaction
        )
        response.raise_for_status()
        return response.json()
    
    def predict_batch(self, transactions: List[Dict]) -> List[Dict]:
        """Predict fraud for multiple transactions.
        
        Args:
            transactions: List of transaction data
            
        Returns:
            List of prediction responses
        """
        response = requests.post(
            f"{self.base_url}/predict/batch",
            json=transactions
        )
        response.raise_for_status()
        return response.json()
    
    def get_models_info(self) -> Dict:
        """Get information about loaded models.
        
        Returns:
            Models information
        """
        response = requests.get(f"{self.base_url}/models/info")
        response.raise_for_status()
        return response.json()


def example_single_prediction():
    """Example: Single transaction prediction."""
    client = FraudDetectionClient()
    
    # Check API health
    health = client.health_check()
    print("API Health:", json.dumps(health, indent=2))
    
    # Create sample transaction
    transaction = {
        "transaction_id": "TXN_00001234",
        "timestamp": datetime.now().isoformat(),
        "user_id": "USER_000123",
        "amount": 150.50,
        "transaction_type": "purchase",
        "location": "domestic",
        "merchant_risk_score": 35.0
    }
    
    # Make prediction
    print("\nTransaction:", json.dumps(transaction, indent=2))
    
    prediction = client.predict_single(transaction)
    print("\nPrediction:", json.dumps(prediction, indent=2))
    
    # Interpret result
    if prediction['is_fraud']:
        print(f"\n⚠️  FRAUD ALERT: Risk Level {prediction['risk_level'].upper()}")
        print(f"   Fraud Probability: {prediction['fraud_probability']:.2%}")
    else:
        print(f"\n✅ Transaction appears legitimate")
        print(f"   Fraud Probability: {prediction['fraud_probability']:.2%}")


def example_batch_prediction():
    """Example: Batch transaction prediction."""
    client = FraudDetectionClient()
    
    # Create sample transactions
    transactions = [
        {
            "transaction_id": f"TXN_000{i:04d}",
            "timestamp": datetime.now().isoformat(),
            "user_id": f"USER_{i % 100:04d}",
            "amount": 50.0 + i * 10,
            "transaction_type": ["purchase", "withdrawal", "transfer"][i % 3],
            "location": ["domestic", "international"][i % 2],
            "merchant_risk_score": 20.0 + (i % 50)
        }
        for i in range(10)
    ]
    
    # Make batch prediction
    predictions = client.predict_batch(transactions)
    
    print("\nBatch Prediction Results:")
    print("=" * 80)
    
    fraud_count = 0
    for pred in predictions:
        status = "🔴 FRAUD" if pred['is_fraud'] else "🟢 OK"
        print(f"{status} | {pred['transaction_id']} | "
              f"Prob: {pred['fraud_probability']:.3f} | "
              f"Risk: {pred['risk_level']}")
        if pred['is_fraud']:
            fraud_count += 1
    
    print("=" * 80)
    print(f"\nSummary: {fraud_count}/{len(predictions)} transactions flagged as fraud")


def example_high_risk_transaction():
    """Example: High-risk transaction."""
    client = FraudDetectionClient()
    
    # Create a suspicious transaction
    suspicious_transaction = {
        "transaction_id": "TXN_SUSPICIOUS",
        "timestamp": datetime.now().isoformat(),
        "user_id": "USER_999999",
        "amount": 9999.99,  # Very high amount
        "transaction_type": "withdrawal",
        "location": "international",  # International location
        "merchant_risk_score": 95.0  # Very high risk score
    }
    
    print("Testing suspicious transaction:")
    print(json.dumps(suspicious_transaction, indent=2))
    
    prediction = client.predict_single(suspicious_transaction)
    print("\nPrediction:", json.dumps(prediction, indent=2))
    
    if prediction['fraud_probability'] > 0.7:
        print("\n🚨 HIGH RISK - Manual review recommended!")
    elif prediction['fraud_probability'] > 0.4:
        print("\n⚠️  MEDIUM RISK - Additional verification suggested")
    else:
        print("\n✅ LOW RISK - Transaction cleared")


if __name__ == "__main__":
    print("=" * 80)
    print("Fraud Detection API Client - Examples")
    print("=" * 80)
    
    try:
        # Example 1: Single prediction
        print("\n[Example 1] Single Transaction Prediction")
        print("-" * 80)
        example_single_prediction()
        
        # Example 2: Batch prediction
        print("\n\n[Example 2] Batch Transaction Prediction")
        print("-" * 80)
        example_batch_prediction()
        
        # Example 3: High-risk transaction
        print("\n\n[Example 3] High-Risk Transaction")
        print("-" * 80)
        example_high_risk_transaction()
        
        print("\n" + "=" * 80)
        print("Examples completed successfully!")
        print("=" * 80)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to API")
        print("Please make sure the API is running:")
        print("  python -m uvicorn src.api.app:app --reload")
        print("  or: ./run_api.sh")
    except Exception as e:
        print(f"\n❌ Error: {e}")
