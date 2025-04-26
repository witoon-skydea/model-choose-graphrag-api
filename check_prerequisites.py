#!/usr/bin/env python3
"""
Check prerequisites for Model-Choose GraphRAG API
"""
import sys
import importlib
import subprocess

def check_module(module_name):
    """Check if a module is installed"""
    try:
        # Try to import the module
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False

def check_ollama():
    """Check if Ollama is running and available"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/version", timeout=2)
        if response.status_code == 200:
            return True, response.json().get("version", "unknown")
        else:
            return False, None
    except Exception as e:
        return False, None

def main():
    """Check all prerequisites for GraphRAG API"""
    print("Checking prerequisites for Model-Choose GraphRAG API...")
    print("\n=== Required Python Modules ===")
    
    required_modules = [
        "langchain",
        "langchain_community",
        "langchain_core",
        "chromadb",
        "fastapi",
        "networkx",
        "matplotlib",
        "pyvis",
        "numpy",
        "requests"
    ]
    
    all_ok = True
    missing_modules = []
    
    for module_name in required_modules:
        installed = check_module(module_name)
        
        if installed:
            status = "✅ OK"
        else:
            status = "❌ MISSING"
            missing_modules.append(module_name)
            all_ok = False
            
        print(f"{module_name:<20} {status}")
    
    print("\n=== External Services ===")
    
    # Check Ollama
    ollama_running = False
    if check_module("requests"):
        ollama_running, ollama_version = check_ollama()
        if ollama_running:
            print(f"Ollama             ✅ Running (version {ollama_version})")
        else:
            print("Ollama             ❌ Not running or unavailable")
            all_ok = False
    else:
        print("Ollama             ❓ Can't check (requests module missing)")
        all_ok = False
    
    # Summary
    print("\n=== Summary ===")
    if all_ok:
        print("✅ All prerequisites are satisfied!")
    else:
        print("⚠️ Some prerequisites are missing or outdated:")
        
        if missing_modules:
            print("\nMissing modules:")
            for module in missing_modules:
                print(f"  - {module}")
            print("\nInstall missing modules with:")
            print(f"  pip install {' '.join(missing_modules)}")
        
        if not ollama_running:
            print("\nOllama service is not running:")
            print("  - Make sure Ollama is installed: https://ollama.com/download")
            print("  - Start Ollama service before running GraphRAG API")
            print("  - Check if the service is running on port 11434")
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
