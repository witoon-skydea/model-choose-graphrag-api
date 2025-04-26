#!/bin/bash
# Update script for Thai language support in GraphRAG API

echo "Updating GraphRAG API with enhanced Thai language support..."

# Create backup directory
BACKUP_DIR="./backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
echo "Created backup directory at $BACKUP_DIR"

# Backup original files
echo "Creating backups of original files..."
cp rag/knowledge_graph/graph.py "$BACKUP_DIR/graph.py"
cp rag/llm/llm.py "$BACKUP_DIR/llm.py"
cp rag/llm/thai_entity_extraction.py "$BACKUP_DIR/thai_entity_extraction.py"
cp rag/retrieval/retrieval.py "$BACKUP_DIR/retrieval.py"
cp mcp-mmgrag.py "$BACKUP_DIR/mcp-mmgrag.py"

# Check if backup was successful
if [ $? -ne 0 ]; then
    echo "Error creating backups. Update aborted."
    exit 1
fi

echo "Backups created successfully."

# Make files executable
chmod +x test_thai_support.py
chmod +x update_thai_support.sh

echo "Testing connectivity with Ollama..."
OLLAMA_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:11434/api/version || echo "failed")

if [ "$OLLAMA_STATUS" != "200" ]; then
    echo "Warning: Ollama service appears to be unavailable. Please ensure Ollama is running before using Thai language features."
else
    echo "Ollama service is available."
    
    # Check for Thai language models
    MODELS=$(curl -s http://localhost:11434/api/tags)
    if [[ $MODELS == *"llama3"* ]]; then
        echo "Found Llama 3 model for Thai language processing."
    else
        echo "Warning: Llama 3 model not found. Please run 'ollama pull llama3:8b' to enable Thai language processing."
    fi
fi

echo "Running tests to verify Thai language support..."
python test_thai_support.py

# Check if tests passed
if [ $? -ne 0 ]; then
    echo "Warning: Some tests failed. You may still proceed, but there might be issues with Thai language support."
else
    echo "Tests completed successfully."
fi

echo "Thai language support is ready to use!"
echo "Please refer to README-THAI-SUPPORT.md for usage instructions and examples."
