#!/bin/bash

# Quick Start Script for DML Python Implementation

echo "Distance Metric Learning - Python Implementation"
echo "=============================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not found."
    echo "Please install Python 3.7 or higher."
    exit 1
fi

echo "Python found: $(python3 --version)"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Setup complete! 🎉"
echo ""
echo "To run the implementation:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Run tests: python test_implementation.py"  
echo "3. Run experiments: python main.py"
echo ""
echo "For more information, see README.md"