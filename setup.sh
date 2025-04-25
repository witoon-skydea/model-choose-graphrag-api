#!/bin/bash

# Model-Choose GraphRAG API Setup Script

# Exit on any error
set -e

# Check if Python 3.8+ is installed
python_version=$(python3 --version 2>&1 | awk '{print $2}')
python_major=$(echo $python_version | cut -d. -f1)
python_minor=$(echo $python_version | cut -d. -f2)

if [ "$python_major" -lt 3 ] || ([ "$python_major" -eq 3 ] && [ "$python_minor" -lt 8 ]); then
    echo "Error: Python 3.8 or higher is required. Found Python $python_version"
    exit 1
fi

echo "Using Python $python_version"

# Set up virtual environment
echo "Setting up virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p uploads
mkdir -p visualizations
mkdir -p companies
mkdir -p db

# Setup complete
echo ""
echo "Setup complete! Run the API server with:"
echo "source venv/bin/activate && python app.py"
echo ""
echo "The API will be available at: http://localhost:8765"
echo "API documentation will be available at: http://localhost:8765/docs"
