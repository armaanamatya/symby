#!/bin/bash
# AI Research Impact Observatory - Run Script
# DGX Spark Frontier Hackathon

set -e

echo "=========================================="
echo "AI Research Impact Observatory"
echo "DGX Spark Frontier Hackathon - Symby AI"
echo "=========================================="

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found"
    exit 1
fi

# Check if streamlit is installed
if ! python3 -c "import streamlit" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

MODE=${1:-dashboard}

case $MODE in
    dashboard)
        echo ""
        echo "Starting Streamlit Dashboard..."
        echo "Open: http://localhost:8501"
        echo ""
        streamlit run app.py --server.port 8501
        ;;
    api)
        echo ""
        echo "Starting FastAPI Backend..."
        echo "Docs: http://localhost:8000/docs"
        echo ""
        uvicorn api:app --host 0.0.0.0 --port 8000 --reload
        ;;
    test)
        echo ""
        echo "Running tests..."
        echo ""
        python3 src/data/loader.py
        python3 src/metrics/impact_metrics.py
        echo ""
        echo "All tests passed!"
        ;;
    *)
        echo ""
        echo "Usage: ./run.sh [dashboard|api|test]"
        echo ""
        echo "  dashboard - Run Streamlit dashboard (default)"
        echo "  api       - Run FastAPI backend"
        echo "  test      - Run component tests"
        echo ""
        ;;
esac
