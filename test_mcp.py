#!/usr/bin/env python3
"""
Test script for MCP (Model Communication Protocol) module
"""
import os
import json
from dotenv import load_dotenv
from mcp import MCPClient, ClaudeAPI

# Load environment variables
load_dotenv()

def test_claude_api():
    """Test direct Claude API integration"""
    print("\n===== Testing Claude API =====")
    
    # Check API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key or api_key == "your_anthropic_api_key":
        print("⚠️ No valid API key found. Please set ANTHROPIC_API_KEY in .env file.")
        return False
    
    # Initialize Claude API
    claude = ClaudeAPI()
    
    # Test simple query
    test_query = "What is your model name?"
    print(f"Sending query to Claude API: '{test_query}'")
    
    response = claude.generate(
        system_prompt="You are Claude AI assistant. Be brief and concise.",
        query=test_query,
        temperature=0.7
    )
    
    if "error" in response:
        print(f"❌ Error: {response['error']}")
        if "details" in response:
            print(f"Details: {response['details']}")
        return False
    
    print(f"✅ Response received:")
    print(f"Content: {response.get('content', '')}")
    print(f"Model: {response.get('model', '')}")
    print(f"Message ID: {response.get('message_id', '')}")
    
    return True

def test_mcp_client():
    """Test MCP Client functionality"""
    print("\n===== Testing MCP Client =====")
    
    # Check API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key or api_key == "your_anthropic_api_key":
        print("⚠️ No valid API key found. Please set ANTHROPIC_API_KEY in .env file.")
        return False
    
    # Initialize MCP client
    mcp = MCPClient()
    
    # Example documents (would come from RAG system)
    example_docs = [
        {
            "content": "GraphRAG is a system that combines vector search with knowledge graph capabilities for enhanced retrieval.",
            "metadata": {"source": "documentation.pdf"}
        },
        {
            "content": "The system supports multiple LLM models including Claude, OpenAI GPT models, and local models via Ollama.",
            "metadata