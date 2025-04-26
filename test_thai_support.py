#!/usr/bin/env python3
"""
Test script for Thai language support in GraphRAG
"""
import os
import sys
import importlib
from rag.knowledge_graph.graph import contains_thai

# Check if required modules are installed
def check_module(module_name):
    """Check if a module is installed"""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False

# Set up mock LLM for offline testing
class MockLLM:
    """Mock LLM for testing without Ollama"""
    def __init__(self, model="mock", temperature=0.1):
        self.model = model
        self.temperature = temperature
    
    def invoke(self, prompt):
        """Simulate LLM response"""
        if "extract entities" in prompt.lower() or "สกัดเอนทิตี" in prompt:
            if "thai" in self.model or contains_thai(prompt):
                # Return mock Thai entity extraction response
                return """```json
{
  "entities": [
    {"id": "person_1", "name": "สมชาย ใจดี", "type": "person"},
    {"id": "company_1", "name": "บริษัท พลังงานไทย", "type": "organization"},
    {"id": "gov_1", "name": "กระทรวงพลังงาน", "type": "organization"}
  ],
  "relations": [
    {"source": "person_1", "target": "company_1", "type": "works_for"},
    {"source": "gov_1", "target": "company_1", "type": "regulates"}
  ]
}```"""
            else:
                # Return mock English entity extraction response
                return """```json
{
  "entities": [
    {"id": "person_1", "name": "John Smith", "type": "person"},
    {"id": "company_1", "name": "Energy Corp", "type": "organization"}
  ],
  "relations": [
    {"source": "person_1", "target": "company_1", "type": "works_for"}
  ]
}```"""
        elif "extract" in prompt.lower() and "key entities" in prompt.lower():
            # Mock entity extraction from query
            return '["Energy", "Thailand", "Renewable"]'
        else:
            # Generic mock response
            return "This is a mock LLM response for testing purposes."

# Import required modules if available
if check_module('langchain_community.llms'):
    from langchain_community.llms import Ollama
    has_langchain = True
else:
    has_langchain = False

# Import other required modules
if has_langchain:
    from rag.llm.thai_entity_extraction import extract_thai_entities_relations
    from rag.llm.llm import extract_query_entities
    from rag.vector_store import get_vector_store
    from rag.retrieval.retrieval import hybrid_retrieval
    from rag.config import CompanyConfig
    from rag.knowledge_graph.graph import KnowledgeGraph

def test_thai_detection():
    """Test Thai language detection"""
    print("\n=== Testing Thai Language Detection ===")
    
    test_cases = [
        "This is English text only",
        "ทดสอบภาษาไทย",
        "This has some ภาษาไทย mixed in",
        "ข้อความภาษาไทยและ English mixed"
    ]
    
    for text in test_cases:
        is_thai = contains_thai(text)
        print(f"Text: {text}")
        print(f"Contains Thai: {is_thai}")
        print()

def test_thai_entity_extraction():
    """Test Thai entity extraction"""
    print("\n=== Testing Thai Entity Extraction ===")
    
    if not has_langchain:
        print("Skipping test: langchain modules not available")
        return
    
    # Try to use Ollama if available, otherwise use mock LLM
    try:
        llm = Ollama(model="llama3:8b", temperature=0.1)
        print("Using Ollama for testing...")
    except Exception as e:
        print(f"Ollama not available ({e}), using mock LLM...")
        llm = MockLLM(model="mock_thai", temperature=0.1)
    
    # Read from test file if available, otherwise use sample text
    test_file = os.path.join(os.path.dirname(__file__), "test_data", "thai_sample.txt")
    if os.path.exists(test_file):
        with open(test_file, 'r', encoding='utf-8') as f:
            content = f.read()
            # Extract first paragraph
            paragraphs = [p for p in content.split('\n\n') if p.strip()]
            if paragraphs:
                test_cases = [paragraphs[0]]
                if len(paragraphs) > 2:
                    test_cases.append(paragraphs[2])
            else:
                test_cases = ["บริษัท พลังงานไทย จำกัด (มหาชน) มีนายสมชาย ใจดี เป็นประธานกรรมการ"]
    else:
        test_cases = [
            "บริษัท ปตท. จำกัด (มหาชน) ได้ลงนามในสัญญาร่วมทุนกับบริษัท เอบีซี จำกัด เพื่อพัฒนาโครงการพลังงานสะอาดในประเทศไทย โดยมีนายสมชาย ใจดี เป็นประธานในพิธี",
            "กระทรวงพลังงานประกาศนโยบายส่งเสริมการใช้พลังงานทดแทน โดยตั้งเป้าหมายให้มีการใช้พลังงานทดแทน 30% ภายในปี 2573"
        ]
    
    for text in test_cases:
        print(f"Text: {text[:100]}...")
        try:
            entities, relations = extract_thai_entities_relations(llm, text)
            
            print(f"Extracted {len(entities)} entities:")
            for entity in entities:
                print(f"- {entity['name']} (Type: {entity['type']}, ID: {entity['id']})")
            
            print(f"Extracted {len(relations)} relations:")
            for relation in relations:
                entity_from = next((e['name'] for e in entities if e['id'] == relation['source']), relation['source'])
                entity_to = next((e['name'] for e in entities if e['id'] == relation['target']), relation['target'])
                print(f"- {entity_from} --[{relation['type']}]--> {entity_to}")
                
        except Exception as e:
            print(f"Error extracting entities: {e}")
        print()

def test_thai_query_processing():
    """Test Thai query entity extraction"""
    print("\n=== Testing Thai Query Processing ===")
    
    if not has_langchain:
        print("Skipping test: langchain modules not available")
        return
    
    # Try to use Ollama if available, otherwise use mock LLM
    try:
        llm = Ollama(model="llama3:8b", temperature=0.1)
        print("Using Ollama for testing...")
    except Exception as e:
        print(f"Ollama not available ({e}), using mock LLM...")
        llm = MockLLM(model="mock_thai", temperature=0.1)
    
    test_queries = [
        "What is the capital of France?",
        "ใครเป็นประธานบริษัท ปตท?",
        "นโยบายพลังงานทดแทนของประเทศไทยมีอะไรบ้าง?",
        "What are the renewable energy policies in Thailand?"
    ]
    
    for query in test_queries:
        print(f"Query: {query}")
        try:
            is_thai = contains_thai(query)
            print(f"Contains Thai: {is_thai}")
            
            entities = extract_query_entities(llm, query)
            print(f"Extracted entities: {entities}")
        except Exception as e:
            print(f"Error processing query: {e}")
        print()

def test_hybrid_retrieval():
    """Test hybrid retrieval with Thai content"""
    print("\n=== Testing Hybrid Retrieval ===")
    
    # Initialize company config
    config = CompanyConfig()
    try:
        active_company = config.get_active_company()["id"]
    except Exception as e:
        print(f"Error getting active company: {e}")
        print("Using 'default' company")
        active_company = "default"
    
    # Get database path
    try:
        db_path = config.get_db_path(active_company)
        print(f"Using database at: {db_path}")
    except Exception as e:
        print(f"Error getting database path: {e}")
        return
    
    # Initialize vector store
    try:
        vector_store = get_vector_store(db_path)
    except Exception as e:
        print(f"Error initializing vector store: {e}")
        return
    
    # Initialize knowledge graph
    graph_dir = os.path.join(os.path.dirname(db_path), "graph")
    knowledge_graph = None
    if os.path.exists(graph_dir) and os.path.exists(os.path.join(graph_dir, "knowledge_graph.pkl")):
        try:
            knowledge_graph = KnowledgeGraph(graph_dir)
            print(f"Loaded knowledge graph with {knowledge_graph.graph.number_of_nodes()} nodes")
        except Exception as e:
            print(f"Error loading knowledge graph: {e}")
    else:
        print(f"Knowledge graph not found at {graph_dir}")
    
    # Initialize LLM
    llm = Ollama(model="llama3:8b", temperature=0.1)
    
    # Test queries
    test_queries = [
        "Tell me about renewable energy",
        "ข้อมูลเกี่ยวกับพลังงานทดแทน",
        "Who are the key stakeholders?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        is_thai = contains_thai(query)
        print(f"Contains Thai: {is_thai}")
        
        try:
            results = hybrid_retrieval(vector_store, knowledge_graph, query, llm, k=3)
            
            print(f"Retrieved {len(results)} documents:")
            for i, doc in enumerate(results):
                print(f"\nDocument {i+1}:")
                print(f"Content: {doc.page_content[:100]}...")
                print(f"Source: {doc.metadata.get('source', 'Unknown')}")
                print(f"Score: {doc.metadata.get('priority_score', 'N/A')}")
                print(f"Contains Thai: {doc.metadata.get('contains_thai', 'N/A')}")
        except Exception as e:
            print(f"Error in hybrid retrieval: {e}")

def main():
    """Run all tests"""
    print("Testing Thai Language Support in GraphRAG API")
    print("============================================")
    
    # Check required modules
    missing_modules = []
    for module in ["langchain", "langchain_community", "networkx", "matplotlib"]:
        if not check_module(module):
            missing_modules.append(module)
    
    if missing_modules:
        print(f"Warning: The following modules are missing: {', '.join(missing_modules)}")
        print("Some tests may be skipped. Install missing modules with:")
        print(f"  pip install {' '.join(missing_modules)}")
        print()
    
    # Run tests
    test_thai_detection()
    test_thai_entity_extraction()
    test_thai_query_processing()
    
    # Only run hybrid retrieval test if vector store modules are available
    if has_langchain and check_module("chromadb"):
        test_hybrid_retrieval()
    else:
        print("\n=== Skipping Hybrid Retrieval Test ===")
        print("Required modules not available.")

if __name__ == "__main__":
    main()
