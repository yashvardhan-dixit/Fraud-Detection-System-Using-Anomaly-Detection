# Installation Notes

## Package Installation

This project is designed to be installed as a Python package. This ensures proper import paths and dependencies.

### Recommended Installation Steps

1. **Clone the repository**:
```bash
git clone https://github.com/yashvardhan-dixit/Fraud-Detection-System-Using-Anomaly-Detection.git
cd Fraud-Detection-System-Using-Anomaly-Detection
```

2. **Create and activate a virtual environment** (highly recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install the package in development mode**:
```bash
pip install -e .
```

This will:
- Install all required dependencies from `requirements.txt`
- Make the `src` package importable from anywhere
- Allow you to edit the code and see changes immediately
- Enable running tests, examples, and scripts without path issues

### Alternative: Install Dependencies Only

If you prefer not to install as a package:
```bash
pip install -r requirements.txt
```

Note: You may encounter import errors with this approach when running examples or tests.

### Verification

Verify the installation:
```bash
python verify_setup.py
```

Or test imports:
```python
python -c "from src.models.isolation_forest import IsolationForestDetector; print('Import successful!')"
```

### Running Code

After installation, you can:

- **Train models**: `python train.py`
- **Run API**: `uvicorn src.api.app:app --reload`
- **Run dashboard**: `streamlit run src/visualization/dashboard.py`
- **Run tests**: `pytest tests/ -v`
- **Run examples**: `python examples/usage_example.py`

### Troubleshooting

**Import Errors**: If you see `ModuleNotFoundError: No module named 'src'`:
- Make sure you've run `pip install -e .`
- Verify you're in the virtual environment
- Check that you're running commands from the project root directory

**Dependency Errors**: If you see missing package errors:
- Run `pip install -r requirements.txt`
- Check your Python version (3.9+ required)

**Path Issues**: If you encounter path-related errors:
- Always run commands from the project root directory
- Use `pip install -e .` instead of manual sys.path manipulation
