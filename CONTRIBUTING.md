# Contributing to Fraud Detection System

Thank you for your interest in contributing to the Fraud Detection System! This document provides guidelines and instructions for contributing.

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive feedback
- Respect differing viewpoints and experiences

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- Clear title and description
- Steps to reproduce
- Expected vs actual behavior
- System information (OS, Python version, etc.)
- Relevant logs or error messages

### Suggesting Features

For feature requests, please:
- Check if the feature is already requested
- Provide a clear use case
- Explain why it would be valuable
- Consider implementation details

### Submitting Pull Requests

1. **Fork the repository**
2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**:
   - Follow the existing code style
   - Add tests for new functionality
   - Update documentation as needed

4. **Run tests**:
   ```bash
   pytest tests/ -v
   ```

5. **Commit your changes**:
   ```bash
   git commit -m "Add: brief description of changes"
   ```
   
   Use conventional commits:
   - `Add:` for new features
   - `Fix:` for bug fixes
   - `Update:` for updates to existing features
   - `Remove:` for removing code/features
   - `Docs:` for documentation changes

6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request**

## Development Setup

1. **Clone your fork**:
   ```bash
   git clone https://github.com/YOUR-USERNAME/Fraud-Detection-System-Using-Anomaly-Detection.git
   cd Fraud-Detection-System-Using-Anomaly-Detection
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```

4. **Run tests**:
   ```bash
   pytest tests/ -v
   ```

## Coding Standards

### Python Style Guide

- Follow PEP 8
- Use type hints where applicable
- Maximum line length: 120 characters
- Use descriptive variable names
- Add docstrings to all functions and classes

### Example:

```python
def calculate_fraud_score(
    amount: float,
    location: str,
    risk_score: float
) -> float:
    """Calculate fraud score for a transaction.
    
    Args:
        amount: Transaction amount
        location: Transaction location
        risk_score: Merchant risk score
        
    Returns:
        Fraud probability score (0-1)
    """
    # Implementation
    pass
```

### Testing

- Write tests for all new features
- Maintain or improve code coverage
- Use descriptive test names
- Test edge cases and error conditions

### Example Test:

```python
def test_fraud_detection_high_amount():
    """Test that high amounts increase fraud score."""
    model = FraudDetector()
    score = model.predict_score(amount=10000.0)
    assert score > 0.5
```

## Documentation

- Update README.md for significant changes
- Add docstrings to new functions/classes
- Update QUICKSTART.md if needed
- Add examples for new features

## Project Structure

```
src/
├── api/          # API endpoints
├── data/         # Data processing
├── models/       # ML models
├── utils/        # Utilities
└── visualization/# Dashboards and plots

tests/            # Test files
examples/         # Example scripts
config/           # Configuration files
```

## Review Process

1. **Automated checks** must pass:
   - Tests
   - Linting
   - Code coverage

2. **Manual review** by maintainers:
   - Code quality
   - Documentation
   - Design decisions

3. **Approval and merge**

## Questions?

- Open an issue for general questions
- Join discussions in pull requests
- Contact maintainers for sensitive issues

## Recognition

Contributors will be acknowledged in:
- README.md
- Release notes
- Project documentation

Thank you for contributing! 🎉
