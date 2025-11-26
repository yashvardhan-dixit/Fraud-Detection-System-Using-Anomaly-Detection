FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs data/raw data/processed models/saved results/plots

# Expose API port
EXPOSE 8000

# Expose dashboard port
EXPOSE 8501

# Default command (can be overridden)
CMD ["python", "train.py"]
