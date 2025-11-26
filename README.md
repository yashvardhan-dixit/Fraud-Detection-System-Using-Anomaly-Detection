# 🔍 Fraud Detection System Using Anomaly Detection

A comprehensive, production-ready fraud detection system using machine learning anomaly detection techniques. This system includes data preprocessing, feature engineering, multiple ML models (Isolation Forest and Autoencoder), model evaluation with ROC/AUC metrics, threshold tuning, real-time prediction API, explainability features, and an interactive dashboard.

## 🌟 Features

- **Data Generation & Preprocessing**: Synthetic transaction data generation with realistic fraud patterns
- **Feature Engineering**: Advanced feature engineering including temporal, aggregation, velocity, and amount-based features
- **Dual Model Architecture**:
  - **Isolation Forest**: Unsupervised anomaly detection
  - **Autoencoder**: Deep learning-based reconstruction error detection
- **Comprehensive Evaluation**: ROC/AUC curves, precision-recall analysis, confusion matrices, and threshold tuning
- **Real-time API**: FastAPI-based REST API for real-time fraud predictions
- **Explainability**: SHAP-based model explanations and feature importance analysis
- **Interactive Dashboard**: Streamlit dashboard for monitoring and visualization
- **Deployment Ready**: Docker containerization and docker-compose orchestration
- **Clean Architecture**: Modular, maintainable code structure

## 📁 Project Structure

```
Fraud-Detection-System-Using-Anomaly-Detection/
├── config/
│   └── config.yaml              # Configuration file
├── data/
│   ├── raw/                     # Raw data
│   └── processed/               # Processed data
├── models/
│   └── saved/                   # Saved model files
├── notebooks/                   # Jupyter notebooks (optional)
├── results/
│   └── plots/                   # Generated plots and visualizations
├── src/
│   ├── api/
│   │   └── app.py              # FastAPI application
│   ├── data/
│   │   ├── generator.py        # Synthetic data generation
│   │   └── preprocessing.py    # Feature engineering
│   ├── models/
│   │   ├── isolation_forest.py # Isolation Forest model
│   │   ├── autoencoder.py      # Autoencoder model
│   │   └── evaluation.py       # Model evaluation
│   ├── utils/
│   │   ├── config.py           # Configuration manager
│   │   └── logger.py           # Logging utilities
│   └── visualization/
│       ├── dashboard.py        # Streamlit dashboard
│       └── explainability.py   # SHAP explainability
├── tests/
│   └── test_models.py          # Unit tests
├── train.py                    # Main training pipeline
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker container definition
├── docker-compose.yml          # Multi-container orchestration
└── README.md                   # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- pip
- (Optional) Docker and Docker Compose

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yashvardhan-dixit/Fraud-Detection-System-Using-Anomaly-Detection.git
cd Fraud-Detection-System-Using-Anomaly-Detection
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install the package and dependencies**:
```bash
# Option 1: Install dependencies only
pip install -r requirements.txt

# Option 2: Install as a package (recommended)
pip install -e .
```

### Training the Models

Run the complete training pipeline:

```bash
python train.py
```

This will:
1. Generate synthetic transaction data
2. Engineer features
3. Train Isolation Forest and Autoencoder models
4. Evaluate models with ROC/AUC metrics
5. Find optimal classification thresholds
6. Generate explainability plots
7. Save models and results

**Custom data**: To use your own data:
```bash
python train.py --data-path /path/to/your/data.csv
```

### Running the API

Start the FastAPI server for real-time predictions:

```bash
cd src/api
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Access the API documentation at: `http://localhost:8000/docs`

**Example API call**:
```python
import requests

transaction = {
    "transaction_id": "TXN_00001234",
    "timestamp": "2024-01-15T14:30:00",
    "user_id": "USER_000123",
    "amount": 150.50,
    "transaction_type": "purchase",
    "location": "domestic",
    "merchant_risk_score": 35.0
}

response = requests.post("http://localhost:8000/predict", json=transaction)
print(response.json())
```

### Running the Dashboard

Launch the interactive Streamlit dashboard:

```bash
streamlit run src/visualization/dashboard.py
```

Access the dashboard at: `http://localhost:8501`

## 🐳 Docker Deployment

### Build and Run with Docker Compose

1. **Train models**:
```bash
docker-compose --profile training up training
```

2. **Start API and Dashboard**:
```bash
docker-compose up -d api dashboard
```

3. **Access services**:
   - API: `http://localhost:8000`
   - Dashboard: `http://localhost:8501`

4. **Stop services**:
```bash
docker-compose down
```

## 🧪 Testing

Run unit tests:

```bash
pytest tests/ -v
```

Run with coverage:

```bash
pytest tests/ --cov=src --cov-report=html
```

## 📊 Model Performance

The system uses two complementary anomaly detection models:

### Isolation Forest
- **Algorithm**: Tree-based anomaly detection
- **Strengths**: Fast, efficient, works well with high-dimensional data
- **Use Case**: Real-time fraud detection with low latency requirements

### Autoencoder
- **Algorithm**: Neural network-based reconstruction error
- **Strengths**: Captures complex non-linear patterns
- **Use Case**: Batch processing and deep pattern analysis

### Evaluation Metrics

Both models are evaluated using:
- **ROC-AUC Score**: Overall model performance
- **Precision**: Accuracy of fraud predictions
- **Recall**: Coverage of actual fraud cases
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification breakdown

## 🔍 Explainability

The system includes SHAP (SHapley Additive exPlanations) integration for:
- Feature importance ranking
- Individual prediction explanations
- Model behavior understanding

Generate explainability plots:
```python
from src.visualization.explainability import ModelExplainer

explainer = ModelExplainer(model, feature_names)
explainer.fit_explainer(X_train)
explainer.explain_predictions(X_test)
explainer.plot_feature_importance(save_path='feature_importance.png')
```

## ⚙️ Configuration

Edit `config/config.yaml` to customize:

- Data generation parameters (sample size, fraud ratio)
- Model hyperparameters (contamination, encoding dimensions)
- Evaluation settings (threshold range, CV folds)
- API and dashboard settings

## 📈 Key Features Explained

### Feature Engineering

The system creates sophisticated features including:

1. **Temporal Features**: Hour, day of week, weekend indicator, night transaction flag
2. **User Aggregations**: Average amount per user, transaction count, spending patterns
3. **Velocity Features**: Transaction frequency in 1h and 24h windows
4. **Amount Features**: Z-scores, ratios to user average, deviation metrics
5. **Risk Indicators**: Merchant risk scores, location-based risk

### Threshold Tuning

Automatic threshold optimization to balance:
- **Precision**: Minimize false positives (legitimate transactions flagged as fraud)
- **Recall**: Maximize fraud detection (catch actual fraudulent transactions)

The system tests multiple thresholds and selects the optimal one based on F1-score or custom metrics.

## 🛡️ Security Considerations

- Never commit sensitive data or API keys
- Use environment variables for production credentials
- Implement rate limiting on API endpoints
- Regular model retraining with recent data
- Monitor for model drift and performance degradation

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📝 License

This project is open source and available under the MIT License.

## 📧 Contact

For questions or support, please open an issue on GitHub.

## 🙏 Acknowledgments

- Scikit-learn for machine learning utilities
- TensorFlow/Keras for deep learning
- FastAPI for the API framework
- Streamlit for the dashboard
- SHAP for model explainability
- The open-source community

## 🔄 Future Enhancements

- [ ] Add more anomaly detection algorithms (One-Class SVM, LOF)
- [ ] Implement ensemble methods
- [ ] Add real-time model monitoring and alerting
- [ ] Support for streaming data processing
- [ ] Integration with production databases
- [ ] A/B testing framework for model comparison
- [ ] Automated retraining pipeline
- [ ] Multi-model serving and fallback strategies

---

**Built with ❤️ for secure financial transactions**