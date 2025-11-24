#!/bin/bash

# SmartSant-IoT Streamlit App Launcher
# Quick start script for the visual disease classification web application

echo "🏥 SmartSant-IoT: Disease Prediction System"
echo "==========================================="
echo ""

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Virtual environment not activated!"
    echo "Activating virtual environment..."
    
    if [ -d ".venv" ]; then
        source .venv/bin/activate
    elif [ -d "venv" ]; then
        source venv/bin/activate
    else
        echo "❌ No virtual environment found. Creating one..."
        python3 -m venv .venv
        source .venv/bin/activate
    fi
fi

# Check if Streamlit is installed
if ! python3 -c "import streamlit" 2>/dev/null; then
    echo "📦 Installing required dependencies..."
    pip install -r requirements.txt
fi

echo ""
echo "🚀 Starting Streamlit application..."
echo "📍 URL: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run Streamlit app
streamlit run app.py --server.port 8501 --server.address localhost
