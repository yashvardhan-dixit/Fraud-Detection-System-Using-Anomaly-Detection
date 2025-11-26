#!/bin/bash
# Run the Streamlit dashboard

echo "Starting Fraud Detection Dashboard..."
echo "Dashboard will be available at http://localhost:8501"
echo ""

cd "$(dirname "$0")"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run the dashboard
streamlit run src/visualization/dashboard.py --server.port 8501 --server.address 0.0.0.0
