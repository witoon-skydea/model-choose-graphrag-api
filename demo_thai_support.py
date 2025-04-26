#!/usr/bin/env python3
"""
Demo script for Thai language support in GraphRAG API
"""
import os
import sys
import time
import json
import importlib
from pathlib import Path

# Check if required modules are installed
missing_modules = []
for module in ["langchain", "langchain_community", "networkx", "fastapi"]:
    try:
        importlib.import_module(module)
    except ImportError:
        missing_modules.append(module)

if missing_modules:
    print(f"Error: Required modules not found: {', '.join(missing_modules)}")
    print(f"Please install missing modules: pip install {' '.join(missing_modules)}")
    sys.exit(1)

# Import required modules
from langchain_community.llms import Ollama
from rag.knowledge_graph.graph import KnowledgeGraph, contains_thai
from rag.llm.thai_entity_extraction import extract_thai_entities_relations
from rag.llm.llm import extract_query_entities, generate_response
from langchain_core.documents import Document
from rag.config import CompanyConfig

class ThaiGraphRAGDemo:
    """Demo class for Thai language support in GraphRAG"""

    def __init__(self):
        """Initialize the demo"""
        # Check Ollama availability
        try:
            self.llm = Ollama(model="llama3:8b", temperature=0.7)
            print("✅ Connected to Ollama service")
        except Exception as e:
            print(f"❌ Could not connect to Ollama: {e}")
            print("Please make sure Ollama is running")
            sys.exit(1)
        
        # Initialize company config
        self.config = CompanyConfig()
        
        # Test company name (will be created for demo)
        self.test_company_id = "thai_demo"
        self.test_company_name = "Thai Demo Company"
        
        # Test data path
        self.test_data_dir = os.path.join(os.path.dirname(__file__), "test_data")
        os.makedirs(self.test_data_dir, exist_ok=True)
    
    def setup_test_company(self):
        """Set up test company for the demo"""
        print("\n=== Setting up test company ===")
        
        companies = self.config.get_companies()
        if self.test_company_id in companies:
            print(f"Company '{self.test_company_id}' already exists")
        else:
            try:
                from rag.vector_store import add_company
                add_company(
                    self.test_company_id,
                    self.test_company_name,
                    description="Test company for Thai language demo",
                    llm_model="llama3:8b",
                    embedding_model="mxbai-embed-large:latest"
                )
                print(f"Created company: {self.test_company_name}")
            except Exception as e:
                print(f"Error creating company: {e}")
                return False
        
        # Set as active company
        try:
            from rag.vector_store import set_active_company
            set_active_company(self.test_company_id)
            print(f"Set '{self.test_company_id}' as active company")
            return True
        except Exception as e:
            print(f"Error setting active company: {e}")
            return False
    
    def process_thai_document(self, filepath=None):
        """Process a Thai document"""
        print("\n=== Processing Thai Document ===")
        
        # Use provided file or default test file
        if filepath is None:
            filepath = os.path.join(self.test_data_dir, "thai_sample.txt")
            
            # Check if test file exists
            if not os.path.exists(filepath):
                print(f"Test file not found: {filepath}")
                print("Creating sample test file...")
                
                # Create a simple Thai test file
                sample_content = """
บริษัท พลังงานไทย จำกัด (มหาชน) หรือ "TEC" ได้ดำเนินธุรกิจด้านพลังงานในประเทศไทยมากกว่า 20 ปี 
โดยมีนายสมชาย ใจดี เป็นประธานกรรมการ และนางสาวสมหญิง มั่นคง เป็นประธานเจ้าหน้าที่บริหาร (CEO) 
ภายใต้การกำกับดูแลของกระทรวงพลังงาน

บริษัทมุ่งเน้นการลงทุนในธุรกิจพลังงานสะอาด โดยมีโครงการโซลาร์ฟาร์มที่จังหวัดพระนครศรีอยุธยา
และโครงการกังหันลมที่จังหวัดนครศรีธรรมราช
                """
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(sample_content)
                print(f"Created sample file: {filepath}")
        
        # Ensure file exists
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            return False
        
        print(f"Processing file: {filepath}")
        
        # Get document content
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Check if content contains Thai
        is_thai = contains_thai(content)
        print(f"Contains Thai: {is_thai}")
        
        if not is_thai:
            print("Warning: File does not contain Thai text")
        
        # Process with GraphRAG
        print("Extracting entities and relations...")
        try:
            entities, relations = extract_thai_entities_relations(self.llm, content[:2000])  # Process first 2000 chars
            
            print(f"\nExtracted {len(entities)} entities:")
            for entity in entities:
                print(f"- {entity['name']} (Type: {entity['type']}, ID: {entity['id']})")
            
            print(f"\nExtracted {len(relations)} relations:")
            for relation in relations:
                entity_from = next((e['name'] for e in entities if e['id'] == relation['source']), relation['source'])
                entity_to = next((e['name'] for e in entities if e['id'] == relation['target']), relation['target'])
                print(f"- {entity_from} --[{relation['type']}]--> {entity_to}")
            
            # Create knowledge graph
            knowledge_graph = self._create_demo_knowledge_graph(entities, relations)
            return knowledge_graph
            
        except Exception as e:
            print(f"Error processing document: {e}")
            return None
    
    def _create_demo_knowledge_graph(self, entities, relations):
        """Create a simple knowledge graph from entities and relations"""
        # Create a temporary graph directory
        graph_dir = os.path.join(self.test_data_dir, "demo_graph")
        os.makedirs(graph_dir, exist_ok=True)
        
        try:
            # Create knowledge graph
            knowledge_graph = KnowledgeGraph(graph_dir)
            
            # Add entities to graph
            for entity in entities:
                entity_id = entity.get('id')
                entity_name = entity.get('name')
                entity_type = entity.get('type')
                
                attributes = entity.get('attributes', {})
                if not attributes:
                    attributes = {"source": "demo"}
                
                knowledge_graph.add_entity(entity_id, entity_name, entity_type, attributes)
            
            # Add relations to graph
            for relation in relations:
                source_id = relation.get('source')
                target_id = relation.get('target')
                relation_type = relation.get('type')
                
                attributes = relation.get('attributes', {})
                if not attributes:
                    attributes = {"source": "demo"}
                
                knowledge_graph.add_relationship(source_id, target_id, relation_type, attributes)
            
            # Save the graph
            knowledge_graph.save_graph()
            print(f"\nCreated knowledge graph with {knowledge_graph.graph.number_of_nodes()} nodes and {knowledge_graph.graph.number_of_edges()} edges")
            
            # Try to visualize if matplotlib is available
            viz_path = os.path.join(self.test_data_dir, "demo_graph.png")
            try:
                knowledge_graph.visualize(viz_path, max_nodes=20)
                print(f"Knowledge graph visualization saved to: {viz_path}")
            except Exception as viz_error:
                print(f"Could not create visualization: {viz_error}")
            
            return knowledge_graph
            
        except Exception as e:
            print(f"Error creating knowledge graph: {e}")
            return None
    
    def demo_thai_queries(self, knowledge_graph=None):
        """Demonstrate Thai query processing"""
        print("\n=== Demonstrating Thai Queries ===")
        
        # Test queries (both Thai and English)
        test_queries = [
            "ใครเป็นผู้บริหารของบริษัทพลังงานไทย?",
            "What energy projects are mentioned in the document?",
            "บริษัทมีโครงการพลังงานทดแทนอะไรบ้าง?",
            "Who regulates the energy company?"
        ]
        
        # Create test documents
        test_documents = [
            Document(
                page_content="บริษัท พลังงานไทย จำกัด (มหาชน) หรือ TEC ได้ดำเนินธุรกิจด้านพลังงานในประเทศไทยมากกว่า 20 ปี โดยมีนายสมชาย ใจดี เป็นประธานกรรมการ",
                metadata={"source": "company_info.txt"}
            ),
            Document(
                page_content="นางสาวสมหญิง มั่นคง เป็นประธานเจ้าหน้าที่บริหาร (CEO) ของบริษัท พลังงานไทย",
                metadata={"source": "management.txt"}
            ),
            Document(
                page_content="โครงการโซลาร์ฟาร์มบางปะอินตั้งอยู่ที่จังหวัดพระนครศรีอยุธยา มีกำลังการผลิต 250 เมกะวัตต์",
                metadata={"source": "projects.txt"}
            ),
            Document(
                page_content="The wind farm project in Nakhon Si Thammarat province has a capacity of 120 megawatts and began operation in Q4 2023",
                metadata={"source": "projects_en.txt"}
            ),
        ]
        
        # Process each query
        for i, query in enumerate(test_queries):
            print(f"\nQuery {i+1}: {query}")
            
            # Check if query contains Thai
            is_thai = contains_thai(query)
            print(f"Contains Thai: {is_thai}")
            
            # Extract entities from query
            try:
                entities = extract_query_entities(self.llm, query)
                print(f"Extracted entities: {entities}")
            except Exception as e:
                print(f"Error extracting entities: {e}")
                entities = []
            
            # Generate response
            try:
                # Add graph data if available
                graph_data = None
                if knowledge_graph and entities:
                    graph_data = []
                    for entity in entities:
                        # Search for entity in graph
                        entity_ids = knowledge_graph.search_entities(entity, limit=2)
                        for entity_id in entity_ids:
                            # Get neighbors
                            neighbors = knowledge_graph.get_neighbors(entity_id, max_hops=1)
                            graph_data.extend(neighbors)
                
                # Generate response
                response = generate_response(self.llm, test_documents, query, graph_data=graph_data)
                
                print("\nGenerated response:")
                print(f"{response}")
                
            except Exception as e:
                print(f"Error generating response: {e}")
        
        print("\nThai query demonstration completed")

def main():
    """Main function to run the demo"""
    print("=== GraphRAG Thai Language Support Demo ===")
    
    # Create demo instance
    demo = ThaiGraphRAGDemo()
    
    # Setup test company
    demo.setup_test_company()
    
    # Process Thai document
    knowledge_graph = demo.process_thai_document()
    
    # Demonstrate Thai queries
    demo.demo_thai_queries(knowledge_graph)
    
    print("\nDemo completed!")

if __name__ == "__main__":
    main()
