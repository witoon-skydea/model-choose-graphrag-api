#!/usr/bin/env python3
"""
MCP (Model Communication Protocol) module for integration with Claude API

This module provides functionality to communicate with Claude's API
for enhancing GraphRAG capabilities with advanced AI models.
"""
import os
import json
import logging
import requests
import time
from typing import Dict, List, Any, Optional, Union
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class ClaudeAPI:
    """
    Claude API client for interacting with Claude models
    """
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-opus-20240229"):
        """
        Initialize Claude API client
        
        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model: Claude model to use (defaults to Claude 3 Opus)
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            logger.warning("No API key provided for Claude API. Set ANTHROPIC_API_KEY environment variable.")
        
        self.model = model
        self.base_url = "https://api.anthropic.com/v1"
        self.headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
    
    def _create_chat_messages(self, 
                             system_prompt: str, 
                             query: str, 
                             context: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Create chat messages array for Claude API
        
        Args:
            system_prompt: System prompt for the model
            query: User query
            context: Additional context from RAG
            
        Returns:
            List of message objects for Claude API
        """
        messages = []
        
        # Add user message with context
        user_content = query
        if context:
            user_content = f"Context:\n{context}\n\nQuestion: {query}"
        
        messages.append({"role": "user", "content": user_content})
        
        return messages
    
    def generate(self, 
                system_prompt: str, 
                query: str, 
                context: Optional[str] = None, 
                temperature: float = 0.7,
                max_tokens: int = 1000) -> Dict[str, Any]:
        """
        Generate a response from Claude
        
        Args:
            system_prompt: System prompt for the model
            query: User query
            context: Additional context from RAG
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Response from Claude API
        """
        if not self.api_key:
            logger.error("Cannot generate response: No API key provided")
            return {"error": "No API key provided. Set ANTHROPIC_API_KEY environment variable."}
        
        messages = self._create_chat_messages(system_prompt, query, context)
        
        payload = {
            "model": self.model,
            "messages": messages,
            "system": system_prompt,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/messages",
                headers=self.headers,
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "content": data.get("content", [{"text": ""}])[0].get("text", ""),
                    "model": data.get("model", self.model),
                    "message_id": data.get("id", ""),
                    "usage": data.get("usage", {})
                }
            else:
                logger.error(f"Error from Claude API: {response.status_code} - {response.text}")
                return {
                    "error": f"API error: {response.status_code}",
                    "details": response.text
                }
                
        except Exception as e:
            logger.error(f"Exception while calling Claude API: {str(e)}")
            return {"error": str(e)}
    
    def format_context_from_documents(self, documents: List[Dict[str, Any]]) -> str:
        """
        Format document chunks into a context string for Claude
        
        Args:
            documents: List of document chunks
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, doc in enumerate(documents):
            content = doc.get("content", "")
            source = doc.get("metadata", {}).get("source", "Unknown")
            context_parts.append(f"[Document {i+1}] (Source: {source})\n{content}\n")
        
        return "\n".join(context_parts)


class MCPClient:
    """
    MCP (Model Communication Protocol) client for GraphRAG API
    """
    def __init__(self, claude_api_key: Optional[str] = None):
        """
        Initialize MCP client
        
        Args:
            claude_api_key: Anthropic API key for Claude models
        """
        self.claude = ClaudeAPI(api_key=claude_api_key)
        
        # Default system prompts
        self._default_system_prompt = """You are a helpful AI assistant. You answer questions based on the provided context.
If the answer cannot be found in the context, say that you don't know based on the available information.
Only use information from the provided context to answer. Do not make up information."""
        
        # Custom system prompts based on retrieval type
        self._system_prompts = {
            "default": self._default_system_prompt,
            "vector": """You are a helpful AI assistant. Answer the user's question based solely on the provided vector search results.
If the answer cannot be found in the provided documents, say that you don't know based on the available information.
Do not make up information.""",
            "graph": """You are a helpful AI assistant with knowledge graph capabilities. Answer the user's question based on the provided
knowledge graph entities and relationships. Explain connections between entities when relevant.
If the answer cannot be found in the provided knowledge graph data, say that you don't know based on the available information.
Do not make up information.""",
            "hybrid": """You are a helpful AI assistant with hybrid retrieval capabilities. Answer the user's question using 
both the vector search results and knowledge graph data. When possible, explain connections between concepts.
If the answer cannot be found in the provided information, say that you don't know based on the available information.
Do not make up information."""
        }
    
    def get_system_prompt(self, retrieval_type: str, custom_prompt: Optional[str] = None) -> str:
        """
        Get system prompt based on retrieval type
        
        Args:
            retrieval_type: Type of retrieval (vector, graph, hybrid)
            custom_prompt: Custom system prompt (overrides default)
            
        Returns:
            System prompt string
        """
        if custom_prompt:
            return custom_prompt
        
        return self._system_prompts.get(retrieval_type, self._default_system_prompt)
    
    def enhance_response(self, 
                      query: str, 
                      documents: List[Dict[str, Any]], 
                      retrieval_type: str = "hybrid",
                      custom_prompt: Optional[str] = None,
                      temperature: float = 0.7) -> Dict[str, Any]:
        """
        Enhance RAG response using Claude
        
        Args:
            query: User query
            documents: Retrieved documents/chunks
            retrieval_type: Type of retrieval (vector, graph, hybrid)
            custom_prompt: Custom system prompt
            temperature: Generation temperature
            
        Returns:
            Enhanced response from Claude
        """
        # Get appropriate system prompt
        system_prompt = self.get_system_prompt(retrieval_type, custom_prompt)
        
        # Format context from documents
        context = self.claude.format_context_from_documents(documents)
        
        # Generate response
        response = self.claude.generate(
            system_prompt=system_prompt,
            query=query,
            context=context,
            temperature=temperature
        )
        
        return response
    
    def process_with_prompt_template(self, 
                                  template_name: str,
                                  query: str,
                                  documents: List[Dict[str, Any]],
                                  temperature: float = 0.7) -> Dict[str, Any]:
        """
        Process query with a specific prompt template
        
        Args:
            template_name: Name of the prompt template file (without .txt extension)
            query: User query
            documents: Retrieved documents/chunks
            temperature: Generation temperature
            
        Returns:
            Response from Claude
        """
        # Load prompt template
        template_path = os.path.join("prompts", f"{template_name}.txt")
        
        try:
            with open(template_path, "r", encoding="utf-8") as f:
                template = f.read()
            
            # Format context from documents
            context = self.claude.format_context_from_documents(documents)
            
            # Generate response
            response = self.claude.generate(
                system_prompt=template,
                query=query,
                context=context,
                temperature=temperature
            )
            
            return response
            
        except FileNotFoundError:
            logger.error(f"Prompt template not found: {template_path}")
            return {"error": f"Prompt template not found: {template_name}"}
        
        except Exception as e:
            logger.error(f"Error processing with prompt template: {str(e)}")
            return {"error": str(e)}


# Example usage
if __name__ == "__main__":
    # Example usage of MCPClient
    mcp = MCPClient()
    
    # Example documents (would come from RAG system)
    example_docs = [
        {
            "content": "GraphRAG is a system that combines vector search with knowledge graph capabilities for enhanced retrieval.",
            "metadata": {"source": "documentation.pdf"}
        },
        {
            "content": "The system supports multiple LLM models including Claude, OpenAI GPT models, and local models via Ollama.",
            "metadata": {"source": "models.md"}
        }
    ]
    
    # Example query
    query = "What is GraphRAG and which models does it support?"
    
    # Get enhanced response
    response = mcp.enhance_response(query, example_docs)
    
    print(f"Query: {query}")
    print(f"Response: {response.get('content', '')}")
