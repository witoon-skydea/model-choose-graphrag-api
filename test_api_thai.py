#!/usr/bin/env python3
"""
Simple test script for Thai language support in GraphRAG API
"""
import requests
import json

BASE_URL = "http://localhost:8765"

def test_query(question):
    """Test query API with Thai question"""
    print(f"Testing query with: {question}")
    
    payload = {
        "question": question,
        "retrieval_method": "hybrid",
        "num_chunks": 3
    }
    
    try:
        response = requests.post(f"{BASE_URL}/query", json=payload)
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Answer: {result.get('answer', 'No answer')}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Exception: {e}")
        return False

def test_active_company():
    """Test getting active company"""
    try:
        response = requests.get(f"{BASE_URL}/companies/active")
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Active company: {result.get('name')} (ID: {result.get('id')})")
            return result
        else:
            print(f"Error: {response.text}")
            return None
    except Exception as e:
        print(f"Exception: {e}")
        return None

def main():
    print("=== Testing Thai Support in GraphRAG API ===")
    
    # Test getting active company
    print("\n1. Testing active company")
    active_company = test_active_company()
    
    # Test Thai queries
    thai_questions = [
        "ผู้บริหารของบริษัทคือใคร",
        "บริษัทมีโครงการอะไรบ้าง",
        "ข้อมูลเกี่ยวกับพลังงานทดแทน"
    ]
    
    for i, question in enumerate(thai_questions):
        print(f"\n{i+2}. Testing Thai query")
        test_query(question)

if __name__ == "__main__":
    main()
