#!/bin/bash

# Install MCP configuration for GraphRAG API
echo "Installing MCP configuration for GraphRAG API..."

# Create MCP config directory if it doesn't exist
mkdir -p ~/.config/mcp

# Check if tools.json exists
if [ -f ~/.config/mcp/tools.json ]; then
    echo "Existing tools.json found. Will merge with our configuration."
    
    # Create backup of existing file
    cp ~/.config/mcp/tools.json ~/.config/mcp/tools.json.bak
    
    # Extract our tool configuration
    GRAPHRAG_CONFIG=$(cat mcp_config.json | jq '.tools["graphrag-api"], .tools["graphrag-fallback"]')
    
    # Merge with existing configuration
    EXISTING_CONFIG=$(cat ~/.config/mcp/tools.json)
    
    # Add our tools to the existing config
    jq --argjson graphrag_api "$(echo $GRAPHRAG_CONFIG | jq '.[0]')" \
       --argjson graphrag_fallback "$(echo $GRAPHRAG_CONFIG | jq '.[1]')" \
       '.tools["graphrag-api"] = $graphrag_api | .tools["graphrag-fallback"] = $graphrag_fallback' \
       ~/.config/mcp/tools.json > ~/.config/mcp/tools.json.new
       
    # Replace the old file with the new one
    mv ~/.config/mcp/tools.json.new ~/.config/mcp/tools.json
else
    echo "No existing tools.json found. Creating new configuration."
    cp mcp_config.json ~/.config/mcp/tools.json
fi

echo "MCP configuration installed successfully."
echo "You can now use the 'graphrag-api' tool with Claude or other MCP-enabled AI tools."
echo "If you encounter issues, try using the 'graphrag-fallback' tool which uses in-memory storage only."
