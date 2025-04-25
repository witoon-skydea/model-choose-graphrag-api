#!/usr/bin/env python3
"""
Model-Choose GraphRAG - MCP Server Implementation

This module implements the Model Context Protocol (MCP) server for
the GraphRAG system, providing tools for RAG with knowledge graph capabilities.
"""
import os
import sys
import json
import shutil
import uuid
import tempfile
from typing import List, Dict, Any, Optional
from mcp.server.fastmcp import FastMCP

# Core imports from GraphRAG system
from rag.document_loader import load_document, is_supported_file
from rag.vector_store import (
    get_vector_store, add_documents, similarity_search,
    list_companies, get_active_company, add_company, remove_company, set_active_company,
    set_company_models, get_system_settings, set_system_settings
)
from rag.llm import get_llm_model, generate_response, list_available_llm_models
from rag.embeddings import get_embeddings_model, list_available_embedding_models
from rag.knowledge_graph import KnowledgeGraph
from rag.retrieval import hybrid_retrieval, convert_graph_to_documents
from rag.visualization import visualize_graph, visualize_query_path
from rag.config import CompanyConfig, SystemConfig

# Base directory for file operations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Constants
DEFAULT_GRAPH_DIR = "graph"

# Use temporary directories for file operations in MCP environment
import tempfile
temp_dir = tempfile.gettempdir()
MCP_DIR = os.path.join(temp_dir, "mcp-graphrag")
UPLOAD_DIR = os.path.join(MCP_DIR, "uploads")
VISUALIZATION_DIR = os.path.join(MCP_DIR, "visualizations")

# For access to companies and db
COMPANIES_DIR = os.path.join(BASE_DIR, "companies")
DB_DIR = os.path.join(BASE_DIR, "db")

# Create necessary directories
try:
    # Try to create all directories
    for directory in [MCP_DIR, UPLOAD_DIR, VISUALIZATION_DIR, COMPANIES_DIR, DB_DIR]:
        os.makedirs(directory, exist_ok=True)
    print(f"Using base directory: {BASE_DIR}", file=sys.stderr)
    print(f"Using temporary directories: {MCP_DIR}", file=sys.stderr)
except OSError as e:
    print(f"Warning: Could not create directories: {e}", file=sys.stderr)
    print(f"Attempting to load fallback version...", file=sys.stderr)
    try:
        fallback_path = os.path.join(BASE_DIR, "mcp-mmgrag-fallback.py")
        if os.path.exists(fallback_path):
            with open(fallback_path) as f:
                exec(f.read())
                sys.exit(0)
    except Exception as e2:
        print(f"Could not load fallback version: {e2}", file=sys.stderr)

# Create MCP server instance
mcp = FastMCP("Model-Choose GraphRAG MCP Server")

class GraphRAGManager:
    """Manager class for GraphRAG functionality"""
    
    def __init__(self):
        """Initialize the GraphRAG manager"""
        self.config = CompanyConfig()
        self.sys_config = SystemConfig()
        self.background_tasks = {}
    
    def get_company_info(self, company_id=None):
        """Get company information"""
        if company_id:
            try:
                db_path = self.config.get_db_path(company_id)
                company_models = self.config.get_company_model_settings(company_id)
                company_details = self.config.get_company_details(company_id)
                return {
                    "id": company_id,
                    "name": company_details.get("name"),
                    "description": company_details.get("description"),
                    "active": company_id == self.config.get_active_company()["id"],
                    "llm_model": company_models.get("llm_model"),
                    "embedding_model": company_models.get("embedding_model"),
                    "db_dir": db_path
                }
            except ValueError:
                return None
        else:
            active_company = self.config.get_active_company()
            return {
                "id": active_company["id"],
                "name": active_company["name"],
                "description": active_company["description"],
                "active": True,
                "llm_model": active_company["llm_model"],
                "embedding_model": active_company["embedding_model"],
                "db_dir": active_company["db_dir"]
            }
    
    def process_file(self, file_path, company_id=None, embedding_model=None, 
                     chunk_size=None, chunk_overlap=None, build_graph=False,
                     llm_model=None, ocr_enabled=False, ocr_options=None):
        """Process a document file for a company"""
        # Get company info
        if company_id:
            try:
                db_path = self.config.get_db_path(company_id)
                company_models = self.config.get_company_model_settings(company_id)
                active_company = company_id
            except ValueError as e:
                return {"error": str(e)}
        else:
            active_company = self.config.get_active_company()["id"]
            db_path = self.config.get_db_path()
            company_models = self.config.get_company_model_settings()
        
        # Get embedding model
        emb_model = embedding_model or company_models.get("embedding_model")
        vector_store = get_vector_store(db_path, embedding_model=emb_model)
        
        # Get knowledge graph if building graph is requested
        knowledge_graph = None
        if build_graph:
            graph_dir = os.path.join(os.path.dirname(db_path), DEFAULT_GRAPH_DIR)
            knowledge_graph = KnowledgeGraph(graph_dir)
        
        # Get chunk settings
        chunk_settings = self.sys_config.get_chunk_settings()
        chunk_size_value = chunk_size or chunk_settings.get("chunk_size")
        chunk_overlap_value = chunk_overlap or chunk_settings.get("chunk_overlap")
        
        # Process file
        try:
            documents = load_document(
                file_path, 
                ocr_enabled=ocr_enabled,
                ocr_options=ocr_options,
                chunk_size=chunk_size_value,
                chunk_overlap=chunk_overlap_value
            )
            add_documents(vector_store, documents)
            
            # Build knowledge graph if requested
            if build_graph and documents:
                llm = get_llm_model(llm_model or company_models.get("llm_model"))
                knowledge_graph.extract_and_add_from_documents(documents, llm)
            
            return {
                "success": True,
                "company_id": active_company,
                "file": os.path.basename(file_path),
                "chunks": len(documents),
                "embedding_model": emb_model,
                "build_graph": build_graph
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "company_id": active_company,
                "file": os.path.basename(file_path)
            }
    
    def query_rag(self, question, company_id=None, retrieval_method="hybrid", 
                 llm_model=None, embedding_model=None, temperature=None,
                 num_chunks=None, num_hops=1, raw_chunks=False, explain=False):
        """Query the RAG system"""
        # Get company info
        if company_id:
            try:
                db_path = self.config.get_db_path(company_id)
                company_models = self.config.get_company_model_settings(company_id)
                active_company = company_id
            except ValueError as e:
                return {"error": str(e)}
        else:
            active_company = self.config.get_active_company()["id"]
            db_path = self.config.get_db_path()
            company_models = self.config.get_company_model_settings()
        
        if not os.path.exists(db_path):
            return {"error": f"Vector store not found at {db_path}"}
        
        # Get embedding model
        embedding_model_value = embedding_model or company_models.get("embedding_model")
        
        # Get vector store
        vector_store = get_vector_store(db_path, embedding_model=embedding_model_value)
        
        # Get top_k from arguments or system settings
        top_k = num_chunks or self.sys_config.get_top_k()
        
        # Get knowledge graph if using graph or hybrid retrieval
        knowledge_graph = None
        graph_data = None
        if retrieval_method in ["graph", "hybrid"]:
            graph_dir = os.path.join(os.path.dirname(db_path), DEFAULT_GRAPH_DIR)
            if not os.path.exists(graph_dir) or not os.path.exists(os.path.join(graph_dir, "knowledge_graph.pkl")):
                if retrieval_method == "graph":
                    return {"error": f"Knowledge graph not found at {graph_dir}"}
                else:
                    # Fall back to vector search for hybrid mode
                    retrieval_method = "vector"
            else:
                knowledge_graph = KnowledgeGraph(graph_dir)
        
        # Get documents based on retrieval method
        if retrieval_method == "vector" or knowledge_graph is None:
            # Use vector search only
            documents = similarity_search(vector_store, question, k=top_k)
        elif retrieval_method == "graph":
            # Use graph search only
            llm_model_value = llm_model or company_models.get("llm_model")
            temperature_value = temperature or self.sys_config.get_temperature()
            llm = get_llm_model(llm_model_value, temperature_value)
            
            from rag.llm.llm import extract_query_entities
            entities = extract_query_entities(llm, question)
            
            graph_data = []
            for entity in entities:
                entity_ids = knowledge_graph.search_entities(entity, limit=2)
                for entity_id in entity_ids:
                    neighbors = knowledge_graph.get_neighbors(entity_id, max_hops=num_hops)
                    graph_data.extend(neighbors)
            
            # Convert to documents format
            documents = convert_graph_to_documents(graph_data)
        else:
            # Use hybrid search (default)
            llm_model_value = llm_model or company_models.get("llm_model")
            temperature_value = temperature or self.sys_config.get_temperature()
            llm = get_llm_model(llm_model_value, temperature_value)
            
            documents = hybrid_retrieval(
                vector_store, 
                knowledge_graph, 
                question, 
                llm,
                k=top_k,
                max_hops=num_hops
            )
            
            # Extract graph data for explanation if needed
            if explain:
                graph_data = []
                for doc in documents:
                    if doc.metadata.get('source') == 'knowledge_graph':
                        entity_id = doc.metadata.get('entity_id')
                        if entity_id:
                            neighbors = knowledge_graph.get_neighbors(entity_id, max_hops=1)
                            for item in neighbors:
                                if item not in graph_data:
                                    graph_data.append(item)
        
        if raw_chunks:
            # Return raw chunks without LLM processing
            chunks = []
            for i, doc in enumerate(documents):
                chunks.append({
                    "index": i + 1,
                    "content": doc.page_content,
                    "source": doc.metadata.get('source', 'Unknown'),
                    "metadata": doc.metadata
                })
            
            return {"chunks": chunks}
        else:
            # Get LLM model (default or override)
            llm_model_value = llm_model or company_models.get("llm_model")
            temperature_value = temperature or self.sys_config.get_temperature()
            
            llm = get_llm_model(llm_model_value, temperature_value)
            
            # Generate response
            response = generate_response(llm, documents, question, custom_template=None, graph_data=graph_data)
            
            result = {
                "question": question,
                "answer": response.strip(),
                "model": llm_model_value,
                "temperature": temperature_value,
                "retrieval_method": retrieval_method,
                "num_chunks": len(documents),
            }
            
            # Create explanation visualization if requested
            if explain and graph_data:
                try:
                    viz_path = os.path.join(VISUALIZATION_DIR, f"query_explanation_{active_company}.png")
                    visualize_query_path(knowledge_graph, graph_data, viz_path)
                    result["explanation_path"] = viz_path
                except Exception as e:
                    print(f"Warning: Could not create visualization: {e}", file=sys.stderr)
                    result["visualization_error"] = str(e)
            
            return result
            
    def build_knowledge_graph(self, company_id=None, llm_model=None, embedding_model=None,
                             query=None, num_docs=50, visualize=False):
        """Build knowledge graph from vector store"""
        # Get company info
        if company_id:
            try:
                db_path = self.config.get_db_path(company_id)
                company_models = self.config.get_company_model_settings(company_id)
                active_company = company_id
            except ValueError as e:
                return {"error": str(e)}
        else:
            active_company = self.config.get_active_company()["id"]
            db_path = self.config.get_db_path()
            company_models = self.config.get_company_model_settings()
        
        if not os.path.exists(db_path):
            return {"error": f"Vector store not found at {db_path}"}
        
        try:
            # Get vector store
            emb_model = embedding_model or company_models.get("embedding_model")
            vector_store = get_vector_store(db_path, embedding_model=emb_model)
            
            # Get knowledge graph
            graph_dir = os.path.join(os.path.dirname(db_path), DEFAULT_GRAPH_DIR)
            knowledge_graph = KnowledgeGraph(graph_dir)
            
            # Get LLM model
            llm = get_llm_model(llm_model or company_models.get("llm_model"))
            
            # Get documents from vector store
            if query:
                documents = similarity_search(vector_store, query, k=num_docs)
            else:
                documents = similarity_search(vector_store, "summarize all information", k=num_docs)
            
            # Build knowledge graph
            knowledge_graph.extract_and_add_from_documents(documents, llm)
            
            # Visualize graph if requested
            viz_path = None
            if visualize:
                try:
                    viz_path = os.path.join(VISUALIZATION_DIR, f"{active_company}_knowledge_graph.png")
                    visualize_graph(knowledge_graph, viz_path, max_nodes=50)
                    print(f"Visualization created at {viz_path}", file=sys.stderr)
                except Exception as e:
                    print(f"Warning: Could not create visualization: {e}", file=sys.stderr)
                    viz_path = None
            
            return {
                "success": True,
                "company_id": active_company,
                "graph_dir": graph_dir,
                "documents_processed": len(documents),
                "nodes": knowledge_graph.graph.number_of_nodes(),
                "edges": knowledge_graph.graph.number_of_edges(),
                "visualization_path": viz_path
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "company_id": active_company
            }

# Initialize the manager
graphrag_manager = GraphRAGManager()

# Define MCP tools
@mcp.tool()
def list_companies() -> str:
    """แสดงรายชื่อบริษัททั้งหมด"""
    companies = list_companies()
    return json.dumps(companies, ensure_ascii=False, indent=2)

@mcp.tool()
def add_company(name: str) -> str:
    """เพิ่มบริษัทใหม่"""
    company_id = name.lower().replace(" ", "_")
    try:
        add_company(company_id, name, "", None, None)
        return json.dumps({"success": True, "id": company_id, "name": name}, ensure_ascii=False, indent=2)
    except ValueError as e:
        return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False, indent=2)

@mcp.tool()
def select_company(name: str) -> str:
    """เลือกบริษัทที่ต้องการทำงานด้วย"""
    try:
        set_active_company(name)
        return json.dumps({"success": True, "message": f"บริษัท '{name}' ถูกตั้งค่าเป็นบริษัทที่ใช้งานอยู่"}, ensure_ascii=False, indent=2)
    except ValueError as e:
        return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False, indent=2)

@mcp.tool()
def get_current_company() -> str:
    """แสดงบริษัทที่กำลังเลือกอยู่"""
    company = graphrag_manager.get_company_info()
    return json.dumps(company, ensure_ascii=False, indent=2)

@mcp.tool()
def add_file(company_name: str, file_path: str, description: str = "") -> str:
    """เพิ่มไฟล์ให้กับบริษัท"""
    try:
        if not os.path.exists(file_path):
            return json.dumps({"success": False, "error": f"ไม่พบไฟล์ {file_path}"}, ensure_ascii=False, indent=2)
        
        if not is_supported_file(file_path):
            return json.dumps({"success": False, "error": f"ไฟล์ {file_path} ไม่ได้รับการสนับสนุน"}, ensure_ascii=False, indent=2)
        
        # Create a unique filename to avoid collisions
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{unique_id}_{os.path.basename(file_path)}"
        upload_path = os.path.join(UPLOAD_DIR, filename)
        
        # Try to copy file to uploads directory
        try:
            shutil.copy2(file_path, upload_path)
            print(f"File copied to {upload_path}", file=sys.stderr)
        except (OSError, IOError) as e:
            print(f"Warning: Could not copy file: {e}", file=sys.stderr)
            # If we can't copy, use the original path
            upload_path = file_path
        
        # Process file
        result = graphrag_manager.process_file(
            upload_path, 
            company_id=company_name,
            build_graph=True,
            ocr_enabled=True
        )
        
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error in add_file: {e}", file=sys.stderr)
        return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False, indent=2)

@mcp.tool()
def add_chat_text(company_name: str, text: str, description: str = "") -> str:
    """เพิ่มข้อความแชทให้กับบริษัท"""
    # Create a temporary text file
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", encoding="utf-8", delete=False) as tmp:
        tmp.write(text)
        temp_path = tmp.name
    
    try:
        # Process file
        result = graphrag_manager.process_file(
            temp_path, 
            company_id=company_name,
            build_graph=True
        )
        
        return json.dumps(result, ensure_ascii=False, indent=2)
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)

@mcp.tool()
def retrieve_chunks(company_name: str, limit: int = 10, data_type: str = "all") -> str:
    """ดึงข้อมูลชังค์จากบริษัท

    company_name: ชื่อบริษัทที่ต้องการดึงข้อมูล
    limit: จำนวนชังค์ที่ต้องการดึง (0 = ไม่จำกัด)
    data_type: ประเภทข้อมูลที่ต้องการ (all, vector, graph)
    """
    try:
        # Get company info
        company_info = graphrag_manager.get_company_info(company_name)
        if not company_info:
            return json.dumps({"success": False, "error": f"ไม่พบบริษัท {company_name}"}, ensure_ascii=False, indent=2)
        
        db_path = company_info["db_dir"]
        
        # Get vector store
        vector_store = get_vector_store(db_path, embedding_model=company_info["embedding_model"])
        
        # Get knowledge graph
        graph_dir = os.path.join(os.path.dirname(db_path), DEFAULT_GRAPH_DIR)
        knowledge_graph = None
        if data_type in ["all", "graph"] and os.path.exists(graph_dir) and os.path.exists(os.path.join(graph_dir, "knowledge_graph.pkl")):
            knowledge_graph = KnowledgeGraph(graph_dir)
        
        # Retrieve chunks
        chunks = []
        
        # Get vector chunks
        if data_type in ["all", "vector"]:
            # Use a generic query to retrieve documents
            vector_docs = similarity_search(vector_store, "summarize all information", k=limit if limit > 0 else 100)
            
            for i, doc in enumerate(vector_docs):
                chunks.append({
                    "type": "vector",
                    "index": i + 1,
                    "content": doc.page_content,
                    "source": doc.metadata.get('source', 'Unknown'),
                    "metadata": doc.metadata
                })
        
        # Get graph chunks
        if data_type in ["all", "graph"] and knowledge_graph:
            # Get a sample of entities from graph
            entity_sample = []
            for i, node in enumerate(knowledge_graph.graph.nodes()):
                if limit > 0 and i >= limit:
                    break
                entity_sample.append(node)
            
            # Get neighbors for each entity
            graph_data = []
            for entity_id in entity_sample:
                neighbors = knowledge_graph.get_neighbors(entity_id, max_hops=1)
                graph_data.extend(neighbors)
            
            # Convert to documents
            graph_docs = convert_graph_to_documents(graph_data)
            
            for i, doc in enumerate(graph_docs):
                chunks.append({
                    "type": "graph",
                    "index": i + 1,
                    "content": doc.page_content,
                    "source": doc.metadata.get('source', 'knowledge_graph'),
                    "metadata": doc.metadata
                })
        
        # Format chunks for displaying
        formatted_chunks = []
        for chunk in chunks:
            formatted_chunks.append(f"[{chunk['type']}] {chunk['content']}")
        
        return "\n\n---\n\n".join(formatted_chunks)
        
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False, indent=2)

@mcp.tool()
def retrieve_raw_chunks(company_name: str, limit: int = 10, data_type: str = "all") -> str:
    """ดึงข้อมูลชังค์ดิบจากบริษัท (raw format)

    company_name: ชื่อบริษัทที่ต้องการดึงข้อมูล
    limit: จำนวนชังค์ที่ต้องการดึง (0 = ไม่จำกัด)
    data_type: ประเภทข้อมูลที่ต้องการ (all, vector, graph)
    """
    try:
        # Get company info
        company_info = graphrag_manager.get_company_info(company_name)
        if not company_info:
            return json.dumps({"success": False, "error": f"ไม่พบบริษัท {company_name}"}, ensure_ascii=False, indent=2)
        
        db_path = company_info["db_dir"]
        
        # Get vector store
        vector_store = get_vector_store(db_path, embedding_model=company_info["embedding_model"])
        
        # Get knowledge graph
        graph_dir = os.path.join(os.path.dirname(db_path), DEFAULT_GRAPH_DIR)
        knowledge_graph = None
        if data_type in ["all", "graph"] and os.path.exists(graph_dir) and os.path.exists(os.path.join(graph_dir, "knowledge_graph.pkl")):
            knowledge_graph = KnowledgeGraph(graph_dir)
        
        # Retrieve chunks
        chunks = []
        
        # Get vector chunks
        if data_type in ["all", "vector"]:
            # Use a generic query to retrieve documents
            vector_docs = similarity_search(vector_store, "summarize all information", k=limit if limit > 0 else 100)
            
            for i, doc in enumerate(vector_docs):
                chunks.append({
                    "type": "vector",
                    "index": i + 1,
                    "content": doc.page_content,
                    "source": doc.metadata.get('source', 'Unknown'),
                    "metadata": doc.metadata
                })
        
        # Get graph chunks
        if data_type in ["all", "graph"] and knowledge_graph:
            # Get a sample of entities from graph
            entity_sample = []
            for i, node in enumerate(knowledge_graph.graph.nodes()):
                if limit > 0 and i >= limit:
                    break
                entity_sample.append(node)
            
            # Get neighbors for each entity
            graph_data = []
            for entity_id in entity_sample:
                neighbors = knowledge_graph.get_neighbors(entity_id, max_hops=1)
                graph_data.extend(neighbors)
            
            # Convert to documents
            graph_docs = convert_graph_to_documents(graph_data)
            
            for i, doc in enumerate(graph_docs):
                chunks.append({
                    "type": "graph",
                    "index": i + 1,
                    "content": doc.page_content,
                    "source": doc.metadata.get('source', 'knowledge_graph'),
                    "metadata": doc.metadata
                })
        
        return json.dumps(chunks, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    mcp.run(transport="stdio")
