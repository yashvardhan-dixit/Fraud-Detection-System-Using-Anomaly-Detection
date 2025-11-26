#!/bin/bash
# Run the FastAPI server

echo "Starting Fraud Detection API..."
echo "API will be available at http://localhost:8000"
echo "API documentation at http://localhost:8000/docs"
echo ""

cd "$(dirname "$0")"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run the API
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
