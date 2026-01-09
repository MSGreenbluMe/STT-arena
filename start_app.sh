#!/bin/bash

echo "========================================"
echo "   STT Arena - Starting Application"
echo "========================================"
echo ""

# Check if virtual environment exists
if [ ! -f "venv/bin/activate" ]; then
    echo "[ERROR] Virtual environment not found!"
    echo "Please run: python3 -m venv venv"
    echo ""
    exit 1
fi

# Activate virtual environment
echo "[1/3] Activating virtual environment..."
source venv/bin/activate

# Check if requirements are installed
echo "[2/3] Checking dependencies..."
if ! pip list | grep -q "streamlit"; then
    echo "[INFO] Installing dependencies..."
    pip install -r requirements.txt
fi

# Check if .env exists
if [ ! -f ".env" ]; then
    echo ""
    echo "[WARNING] .env file not found!"
    echo "Please copy .env.example to .env and add your API keys."
    echo ""
    read -p "Do you want to continue anyway? (y/N) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Start Streamlit app
echo "[3/3] Starting STT Arena..."
echo ""
echo "========================================"
echo " App will open in your browser at:"
echo " http://localhost:8501"
echo "========================================"
echo ""

streamlit run app.py
