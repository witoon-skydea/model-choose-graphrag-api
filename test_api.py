#!/usr/bin/env python3
"""
Test script for the Model-Choose GraphRAG API
"""
import os
import sys
import time
import json
import requests
from typing import Dict, Any, List

# API base URL
BASE_URL = "http://localhost:8765"

def print_section(title: str):
    """Print a section title"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def pretty_print_json(data: Dict[str, Any]):
    """Print JSON data in a pretty format"""
    print(json.dumps(data, indent=2))

def test_root():
    """Test the root endpoint"""
    print_section("Testing Root Endpoint")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status Code: {response.status_code}")
    pretty_print_json(response.json())

def test_company_management():
    """Test company management endpoints"""
    print_section("Testing Company Management")
    
    # List companies
    print("\n>> Listing all companies")
    response = requests.get(f"{BASE_URL}/companies")
    print(f"Status Code: {response.status_code}")
    companies = response.json()
    pretty_print_json(companies)
    
    # Create a test company
    print("\n>> Creating a test company")
    company_data = {
        "id": "test_company",
        "name": "Test Company",
        "description": "API Test Company",
        "set_active": True
    }
    response = requests.post(f"{BASE_URL}/companies", json=company_data)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        pretty_print_json(response.json())
    else:
        print(f"Error: {response.text}")
    
    # Get active company
    print("\n>> Getting active company")
    response = requests.get(f"{BASE_URL}/companies/active")
    print(f"Status Code: {response.status_code}")
    active_company = response.json()
    pretty_print_json(active_company)
    
    return active_company

def test_model_management():
    """Test model management endpoints"""
    print_section("Testing Model Management")
    
    # List LLM models
    print("\n>> Listing LLM models")
    response = requests.get(f"{BASE_URL}/models/llm")
    print(f"Status Code: {response.status_code}")
    llm_models = response.json()
    pretty_print_json(llm_models)
    
    # List embedding models
    print("\n>> Listing embedding models")
    response = requests.get(f"{BASE_URL}/models/embeddings")
    print(f"Status Code: {response.status_code}")
    embedding_models = response.json()
    pretty_print_json(embedding_models)
    
    # Get system settings
    print("\n>> Getting system settings")
    response = requests.get(f"{BASE_URL}/settings")
    print(f"Status Code: {response.status_code}")
    settings = response.json()
    pretty_print_json(settings)
    
    return llm_models, embedding_models, settings

def test_file_upload(test_file_path: str):
    """Test file upload endpoint"""
    print_section("Testing File Upload")
    
    if not os.path.exists(test_file_path):
        print(f"Error: Test file {test_file_path} not found")
        return None
    
    print(f"\n>> Uploading file: {test_file_path}")
    
    with open(test_file_path, "rb") as f:
        files = {"files": (os.path.basename(test_file_path), f)}
        response = requests.post(f"{BASE_URL}/ingest/upload", files=files)
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        upload_result = response.json()
        pretty_print_json(upload_result)
        return upload_result.get("uploaded_files", [])
    else:
        print(f"Error: {response.text}")
        return None

def test_ingest_process(file_paths: List[str], company_id: str = None):
    """Test ingest process endpoint"""
    print_section("Testing Ingest Process")
    
    if not file_paths:
        print("Error: No file paths provided")
        return None
    
    print(f"\n>> Processing files: {file_paths}")
    
    data = {
        "file_paths": file_paths,
        "build_graph": True,
        "visualize_graph": True
    }
    
    if company_id:
        data["company_id"] = company_id
    
    response = requests.post(f"{BASE_URL}/ingest/process", json=data)
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        process_result = response.json()
        pretty_print_json(process_result)
        
        # Wait for processing to complete
        task_id = process_result.get("task_id")
        if task_id:
            print("\n>> Waiting for processing to complete...")
            status = "queued"
            
            while status in ["queued", "running"]:
                time.sleep(2)
                status_response = requests.get(f"{BASE_URL}/ingest/status/{task_id}")
                
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    status = status_data.get("status")
                    print(f"Status: {status} - {status_data.get('message')}")
                    
                    if status == "completed":
                        return status_data
                else:
                    print(f"Error checking status: {status_response.text}")
                    break
        
        return process_result
    else:
        print(f"Error: {response.text}")
        return None

def test_query(question: str, company_id: str = None):
    """Test query endpoint"""
    print_section("Testing Query Endpoint")
    
    print(f"\n>> Querying: {question}")
    
    data = {
        "question": question,
        "retrieval_method": "hybrid",
        "explain": True
    }
    
    if company_id:
        data["company_id"] = company_id
    
    response = requests.post(f"{BASE_URL}/query", json=data)
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        query_result = response.json()
        
        # Print formatted results
        print("\nQuestion:")
        print(question)
        print("\nAnswer:")
        print(query_result.get("answer"))
        print("\nMetadata:")
        print(f"Model: {query_result.get('model')}")
        print(f"Temperature: {query_result.get('temperature')}")
        print(f"Retrieval Method: {query_result.get('retrieval_method')}")
        print(f"Chunks Used: {query_result.get('num_chunks')}")
        
        if "explanation_url" in query_result:
            print(f"\nExplanation visualization available at: {BASE_URL}{query_result.get('explanation_url')}")
        
        return query_result
    else:
        print(f"Error: {response.text}")
        return None

def run_tests(test_file_path: str):
    """Run all tests"""
    # Test root endpoint
    test_root()
    
    # Test company management
    active_company = test_company_management()
    company_id = active_company.get("id") if active_company else None
    
    # Test model management
    test_model_management()
    
    if test_file_path:
        # Test file upload
        uploaded_files = test_file_upload(test_file_path)
        
        if uploaded_files:
            file_paths = [file.get("path") for file in uploaded_files]
            
            # Test ingest process
            ingest_result = test_ingest_process(file_paths, company_id)
            
            if ingest_result and ingest_result.get("status") == "completed":
                # Test query
                test_query("What is the main topic discussed in the document?", company_id)
                test_query("Summarize the key points from the document", company_id)

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python test_api.py <test_file_path>")
        print("Example: python test_api.py sample.pdf")
        return
    
    test_file_path = sys.argv[1]
    run_tests(test_file_path)

if __name__ == "__main__":
    main()
