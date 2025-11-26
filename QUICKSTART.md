# Quick Start Guide

This guide will help you get started with the Fraud Detection System in just a few minutes.

## Prerequisites

- Python 3.9 or higher
- pip package manager
- (Optional) Docker and Docker Compose

## Installation

### Option 1: Local Installation

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

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Option 2: Docker Installation

```bash
git clone https://github.com/yashvardhan-dixit/Fraud-Detection-System-Using-Anomaly-Detection.git
cd Fraud-Detection-System-Using-Anomaly-Detection
docker-compose build
```

## Quick Start

### 1. Train the Models (5-10 minutes)

Train the fraud detection models using synthetic data:

```bash
python train.py
```

This will:
- Generate 10,000 synthetic transactions (2% fraud rate)
- Engineer features from the data
- Train Isolation Forest and Autoencoder models
- Evaluate models and generate performance metrics
- Save trained models to `models/saved/`
- Create visualization plots in `results/plots/`

**Output**: You should see training logs and final performance metrics like:
```
Isolation Forest:
  ROC-AUC: 0.85+
  Precision: 0.70+
  Recall: 0.75+
  F1-Score: 0.72+
```

### 2. Start the API (30 seconds)

Launch the REST API for real-time predictions:

```bash
./run_api.sh
# Or: uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

**Access**: Open `http://localhost:8000/docs` in your browser to see the interactive API documentation.

**Test the API**:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "TXN_TEST001",
    "timestamp": "2024-01-15T14:30:00",
    "user_id": "USER_001",
    "amount": 250.50,
    "transaction_type": "purchase",
    "location": "domestic",
    "merchant_risk_score": 35.0
  }'
```

### 3. Launch the Dashboard (30 seconds)

Open the interactive monitoring dashboard:

```bash
./run_dashboard.sh
# Or: streamlit run src/visualization/dashboard.py
```

**Access**: Open `http://localhost:8501` in your browser.

The dashboard shows:
- Transaction volume over time
- Fraud detection metrics
- Amount distributions
- Risk score analysis
- Real-time filtering and downloads

## Using Docker (Alternative)

### Train Models
```bash
docker-compose --profile training up training
```

### Start All Services
```bash
docker-compose up -d api dashboard
```

### Check Status
```bash
docker-compose ps
docker-compose logs -f
```

### Stop Services
```bash
docker-compose down
```

## Next Steps

1. **Customize the Configuration**: Edit `config/config.yaml` to adjust model parameters
2. **Use Your Own Data**: Run `python train.py --data-path /path/to/your/data.csv`
3. **Explore Examples**: Check `examples/usage_example.py` for code examples
4. **Run Tests**: Execute `pytest tests/ -v` to verify installation
5. **Read the Full Documentation**: See `README.md` for detailed information

## Common Issues

### Import Errors
If you see "ModuleNotFoundError", make sure you've installed all dependencies:
```bash
pip install -r requirements.txt
```

### Port Already in Use
If port 8000 or 8501 is already in use, specify a different port:
```bash
uvicorn src.api.app:app --port 8001
streamlit run src/visualization/dashboard.py --server.port 8502
```

### Model Not Found
If the API can't find models, train them first:
```bash
python train.py
```

## Getting Help

- **Issues**: Open an issue on GitHub
- **Documentation**: Read the full README.md
- **Examples**: Check the `examples/` directory

## Quick Commands Reference

```bash
# Training
python train.py                    # Train with default settings
python train.py --data-path FILE   # Train with custom data

# API
uvicorn src.api.app:app --reload   # Start API with hot reload
curl localhost:8000/health         # Check API health

# Dashboard
streamlit run src/visualization/dashboard.py

# Testing
pytest tests/ -v                   # Run tests
pytest tests/ --cov=src           # Run with coverage

# Docker
docker-compose up -d               # Start all services
docker-compose down                # Stop all services
docker-compose logs -f             # View logs
```

Happy fraud detecting! 🔍
