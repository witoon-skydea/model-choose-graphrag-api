#!/bin/bash

# Model-Choose GraphRAG API Run Script

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Please run setup.sh first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Run API server
echo "Starting GraphRAG API server..."
echo "API will be available at: http://localhost:8765"
echo "API documentation will be available at: http://localhost:8765/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python app.py
