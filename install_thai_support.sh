#!/bin/bash
# Installation script for Thai language support in GraphRAG API

# Colors for better readability
RED="\033[0;31m"
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
BLUE="\033[0;34m"
RESET="\033[0m"

echo -e "${BLUE}=== การติดตั้งส่วนรองรับภาษาไทยสำหรับ GraphRAG API ===${RESET}"
echo

# Function to check if a command exists
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${RED}Error: $1 not found. Please install $1 first.${RESET}"
        return 1
    fi
    return 0
}

# Check for Python
check_command python3 || check_command python || { echo -e "${RED}Python is required but not found.${RESET}"; exit 1; }

# Check for pip
check_command pip3 || check_command pip || { echo -e "${RED}pip is required but not found.${RESET}"; exit 1; }

# Determine pip command
PIP_CMD="pip"
if command -v pip3 &> /dev/null; then
    PIP_CMD="pip3"
fi

# Create test directory
if [ ! -d "test_data" ]; then
    echo -e "${BLUE}Creating test data directory...${RESET}"
    mkdir -p test_data
fi

# Install required Python packages
echo -e "${BLUE}Installing required Python packages...${RESET}"
$PIP_CMD install langchain>=0.1.11 langchain-community>=0.0.27 langchain-core>=0.1.30 langchain-ollama>=0.1.1
$PIP_CMD install networkx>=3.1 matplotlib>=3.7.2 scikit-learn>=1.0.0 requests>=2.0.0

# Check if installation was successful
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}Some packages could not be installed. Thai language support may not work correctly.${RESET}"
    echo -e "${YELLOW}You may need to install these packages manually.${RESET}"
else
    echo -e "${GREEN}Package installation completed successfully.${RESET}"
fi

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo -e "${YELLOW}Warning: Ollama not found in path.${RESET}"
    echo -e "${YELLOW}Please install Ollama from: https://ollama.com/download${RESET}"
    echo -e "${YELLOW}Ollama is required for Thai language processing.${RESET}"
else
    echo -e "${GREEN}Ollama found in path.${RESET}"
    
    # Check if Ollama service is running
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:11434/api/version; then
        echo -e "${GREEN}Ollama service is running.${RESET}"
        
        # Check for Thai language models
        if ollama list 2>/dev/null | grep -q "llama3"; then
            echo -e "${GREEN}Llama 3 model found for Thai language processing.${RESET}"
        else
            echo -e "${BLUE}Downloading Llama 3 model for Thai language processing...${RESET}"
            ollama pull llama3:8b
            
            if [ $? -ne 0 ]; then
                echo -e "${YELLOW}Could not download Llama 3 model.${RESET}"
                echo -e "${YELLOW}You may need to run 'ollama pull llama3:8b' manually.${RESET}"
            else
                echo -e "${GREEN}Llama 3 model downloaded successfully.${RESET}"
            fi
        fi
    else
        echo -e "${YELLOW}Warning: Ollama service is not running.${RESET}"
        echo -e "${YELLOW}Please start Ollama service before using Thai language features.${RESET}"
    fi
fi

# Apply Thai language support updates
echo -e "${BLUE}Applying Thai language support updates...${RESET}"

# Create backup directory
BACKUP_DIR="./backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
echo -e "${GREEN}Created backup directory at $BACKUP_DIR${RESET}"

# Backup original files
echo -e "${BLUE}Creating backups of original files...${RESET}"
cp rag/knowledge_graph/graph.py "$BACKUP_DIR/graph.py" 2>/dev/null || echo -e "${YELLOW}Warning: Could not backup graph.py${RESET}"
cp rag/llm/llm.py "$BACKUP_DIR/llm.py" 2>/dev/null || echo -e "${YELLOW}Warning: Could not backup llm.py${RESET}"
cp rag/llm/thai_entity_extraction.py "$BACKUP_DIR/thai_entity_extraction.py" 2>/dev/null || echo -e "${YELLOW}Warning: Could not backup thai_entity_extraction.py${RESET}"
cp rag/retrieval/retrieval.py "$BACKUP_DIR/retrieval.py" 2>/dev/null || echo -e "${YELLOW}Warning: Could not backup retrieval.py${RESET}"
cp mcp-mmgrag.py "$BACKUP_DIR/mcp-mmgrag.py" 2>/dev/null || echo -e "${YELLOW}Warning: Could not backup mcp-mmgrag.py${RESET}"

# Make files executable
chmod +x test_thai_support.py 2>/dev/null || echo -e "${YELLOW}Warning: Could not make test_thai_support.py executable${RESET}"
chmod +x update_thai_support.sh 2>/dev/null || echo -e "${YELLOW}Warning: Could not make update_thai_support.sh executable${RESET}"
chmod +x demo_thai_support.py 2>/dev/null || echo -e "${YELLOW}Warning: Could not make demo_thai_support.py executable${RESET}"

echo
echo -e "${GREEN}Installation complete!${RESET}"
echo
echo -e "${BLUE}Next steps:${RESET}"
echo -e "1. To test Thai language support, run: ${GREEN}python test_thai_support.py${RESET}"
echo -e "2. To see Thai support in action, run: ${GREEN}python demo_thai_support.py${RESET}"
echo -e "3. For comprehensive documentation, see: ${GREEN}README-THAI-SUPPORT.md${RESET}"
echo
