#!/usr/bin/env python3
"""
Model-Choose GraphRAG API - Enhanced RAG API with knowledge graph and model selection capabilities
"""
import os
import sys
import logging
import shutil
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from rag.document_loader import load_document, scan_directory, is_supported_file
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

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DEFAULT_GRAPH_DIR = "graph"
UPLOAD_DIR = "uploads"
VISUALIZATION_DIR = "visualizations"

# Create necessary directories
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="Model-Choose GraphRAG API",
    description="Enhanced RAG API with knowledge graph and model selection capabilities",
    version="1.0.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (visualizations, etc.)
app.mount("/static", StaticFiles(directory=VISUALIZATION_DIR), name="static")

# Pydantic models for request/response
class Company(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    active: Optional[bool] = None
    llm_model: Optional[str] = None
    embedding_model: Optional[str] = None
    db_dir: Optional[str] = None

class CompanyCreate(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    llm_model: Optional[str] = None
    embedding_model: Optional[str] = None
    set_active: Optional[bool] = False

class CompanyUpdate(BaseModel):
    llm_model: Optional[str] = None
    embedding_model: Optional[str] = None

class SystemSettingsUpdate(BaseModel):
    default_llm_model: Optional[str] = None
    default_embedding_model: Optional[str] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None

class QueryRequest(BaseModel):
    question: str
    company_id: Optional[str] = None
    retrieval_method: str = "hybrid"
    llm_model: Optional[str] = None
    embedding_model: Optional[str] = None
    temperature: Optional[float] = None
    num_chunks: Optional[int] = None
    num_hops: int = 1
    raw_chunks: bool = False
    explain: bool = False

class IngestStatus(BaseModel):
    task_id: str
    status: str
    message: str
    total_files: int
    processed_files: int
    successful_files: int

# In-memory task storage
background_tasks = {}

# Helper functions
def get_company_info(company_id=None):
    """Get company information"""
    config = CompanyConfig()
    if company_id:
        try:
            db_path = config.get_db_path(company_id)
            company_models = config.get_company_model_settings(company_id)
            company_details = config.get_company_details(company_id)
            return {
                "id": company_id,
                "name": company_details.get("name"),
                "description": company_details.get("description"),
                "active": company_id == config.get_active_company()["id"],
                "llm_model": company_models.get("llm_model"),
                "embedding_model": company_models.get("embedding_model"),
                "db_dir": db_path
            }
        except ValueError:
            return None
    else:
        active_company_id = config.get_active_company()
        active_company_details = config.get_company_details(active_company_id)
        return {
            "id": active_company_id,
            "name": active_company_details.get("name"),
            "description": active_company_details.get("description"),
            "active": True,
            "llm_model": active_company_details.get("llm_model"),
            "embedding_model": active_company_details.get("embedding_model"),
            "db_dir": active_company_details.get("db_dir")
        }

async def ingest_files_task(
    task_id: str, 
    files: List[str], 
    company_id: Optional[str] = None,
    embedding_model: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    build_graph: bool = False,
    llm_model: Optional[str] = None,
    visualize_graph: bool = False,
    ocr_enabled: bool = False,
    ocr_options: Optional[Dict[str, Any]] = None
):
    """Background task for ingesting files"""
    try:
        # Update task status
        background_tasks[task_id] = {
            "status": "running",
            "message": "Starting ingestion",
            "total_files": len(files),
            "processed_files": 0,
            "successful_files": 0
        }
        
        # Get company info
        config = CompanyConfig()
        sys_config = SystemConfig()
        
        if company_id:
            try:
                db_path = config.get_db_path(company_id)
                company_models = config.get_company_model_settings(company_id)
                active_company = company_id
            except ValueError as e:
                background_tasks[task_id] = {
                    "status": "failed",
                    "message": f"Error: {str(e)}",
                    "total_files": len(files),
                    "processed_files": 0,
                    "successful_files": 0
                }
                return
        else:
            active_company = config.get_active_company()
            db_path = config.get_db_path()
            company_models = config.get_company_model_settings()
        
        # Update task status
        background_tasks[task_id]["message"] = f"Processing for company: {active_company}"
        
        # Get embedding model
        emb_model = embedding_model or company_models.get("embedding_model")
        vector_store = get_vector_store(db_path, embedding_model=emb_model)
        
        # Get knowledge graph if building graph is requested
        knowledge_graph = None
        if build_graph:
            graph_dir = os.path.join(os.path.dirname(db_path), DEFAULT_GRAPH_DIR)
            knowledge_graph = KnowledgeGraph(graph_dir)
        
        # Get chunk settings
        chunk_settings = sys_config.get_chunk_settings()
        chunk_size_value = chunk_size or chunk_settings.get("chunk_size")
        chunk_overlap_value = chunk_overlap or chunk_settings.get("chunk_overlap")
        
        # Process each file
        successful_files = 0
        all_documents = []
        
        for i, file_path in enumerate(files):
            background_tasks[task_id]["message"] = f"Processing file {i+1}/{len(files)}: {os.path.basename(file_path)}"
            background_tasks[task_id]["processed_files"] = i + 1
            
            try:
                documents = load_document(
                    file_path, 
                    ocr_enabled=ocr_enabled,
                    ocr_options=ocr_options,
                    chunk_size=chunk_size_value,
                    chunk_overlap=chunk_overlap_value
                )
                add_documents(vector_store, documents)
                all_documents.extend(documents)
                successful_files += 1
                background_tasks[task_id]["successful_files"] = successful_files
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
        
        # Build knowledge graph if requested
        if build_graph and all_documents:
            background_tasks[task_id]["message"] = "Building knowledge graph"
            llm = get_llm_model(llm_model or company_models.get("llm_model"))
            knowledge_graph.extract_and_add_from_documents(all_documents, llm)
            
            # Visualize graph if requested
            if visualize_graph:
                background_tasks[task_id]["message"] = "Visualizing knowledge graph"
                viz_path = os.path.join(VISUALIZATION_DIR, f"{active_company}_knowledge_graph.png")
                visualize_graph(knowledge_graph, viz_path, max_nodes=50)
        
        # Update task status on completion
        background_tasks[task_id] = {
            "status": "completed",
            "message": f"Ingestion complete: {successful_files}/{len(files)} files processed successfully",
            "total_files": len(files),
            "processed_files": len(files),
            "successful_files": successful_files
        }
        
    except Exception as e:
        logger.error(f"Error in ingestion task: {e}")
        background_tasks[task_id] = {
            "status": "failed",
            "message": f"Error: {str(e)}",
            "total_files": len(files),
            "processed_files": 0,
            "successful_files": 0
        }

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Welcome to Model-Choose GraphRAG API"}

# Company management endpoints
@app.get("/companies", response_model=List[Company])
async def get_companies():
    """List all companies"""
    companies_list = list_companies()
    return companies_list

@app.get("/companies/active", response_model=Company)
async def get_active_company_endpoint():
    """Get the active company"""
    company = get_company_info()
    return company

@app.post("/companies", response_model=Company)
async def create_company(company: CompanyCreate):
    """Create a new company"""
    try:
        add_company(
            company.id, 
            company.name, 
            company.description or "", 
            company.llm_model,
            company.embedding_model
        )
        
        if company.set_active:
            set_active_company(company.id)
        
        return get_company_info(company.id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.put("/companies/{company_id}/active")
async def set_active_company_endpoint(company_id: str):
    """Set the active company"""
    try:
        set_active_company(company_id)
        return {"message": f"Company '{company_id}' set as active"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.put("/companies/{company_id}/models", response_model=Company)
async def update_company_models(company_id: str, models: CompanyUpdate):
    """Update company models"""
    try:
        if models.llm_model or models.embedding_model:
            set_company_models(company_id, models.llm_model, models.embedding_model)
        return get_company_info(company_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.delete("/companies/{company_id}")
async def delete_company(company_id: str):
    """Delete a company"""
    try:
        remove_company(company_id)
        return {"message": f"Company '{company_id}' removed successfully"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

# Model management endpoints
@app.get("/models/llm")
async def get_llm_models():
    """List available LLM models"""
    return list_available_llm_models()

@app.get("/models/embeddings")
async def get_embedding_models():
    """List available embedding models"""
    return list_available_embedding_models()

@app.get("/settings")
async def get_settings():
    """Get system settings"""
    return get_system_settings()

@app.put("/settings")
async def update_settings(settings: SystemSettingsUpdate):
    """Update system settings"""
    settings_dict = settings.dict(exclude_unset=True, exclude_none=True)
    if settings_dict:
        set_system_settings(settings_dict)
    return get_system_settings()

# Document ingestion endpoints
@app.post("/ingest/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None
):
    """Upload files for later ingestion"""
    uploaded_files = []
    
    for file in files:
        if is_supported_file(file.filename):
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            uploaded_files.append({
                "filename": file.filename,
                "path": file_path,
                "size": os.path.getsize(file_path)
            })
        else:
            logger.warning(f"Unsupported file: {file.filename}")
    
    if not uploaded_files:
        raise HTTPException(status_code=400, detail="No supported files were uploaded")
    
    return {"uploaded_files": uploaded_files}

@app.post("/ingest/process", response_model=IngestStatus)
async def process_files(
    background_tasks: BackgroundTasks,
    file_paths: List[str],
    company_id: Optional[str] = None,
    embedding_model: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    build_graph: bool = False,
    llm_model: Optional[str] = None,
    visualize_graph: bool = False,
    ocr_enabled: bool = False,
    ocr_lang: str = "eng",
    ocr_engine: str = "tesseract",
    ocr_dpi: int = 300,
    use_gpu: bool = True
):
    """Process uploaded files for ingestion"""
    # Check if files exist
    existing_files = []
    for file_path in file_paths:
        if os.path.exists(file_path) and is_supported_file(file_path):
            existing_files.append(file_path)
    
    if not existing_files:
        raise HTTPException(status_code=400, detail="No valid files found")
    
    # OCR options
    ocr_options = {
        "engine": ocr_engine,
        "lang": ocr_lang,
        "dpi": ocr_dpi,
        "use_gpu": use_gpu
    }
    
    # Generate task ID
    import uuid
    task_id = str(uuid.uuid4())
    
    # Create initial task status
    background_tasks[task_id] = {
        "status": "queued",
        "message": "Task queued",
        "total_files": len(existing_files),
        "processed_files": 0,
        "successful_files": 0
    }
    
    # Start background task
    background_tasks.add_task(
        ingest_files_task,
        task_id,
        existing_files,
        company_id,
        embedding_model,
        chunk_size,
        chunk_overlap,
        build_graph,
        llm_model,
        visualize_graph,
        ocr_enabled,
        ocr_options
    )
    
    return {
        "task_id": task_id,
        "status": "queued",
        "message": "Task queued",
        "total_files": len(existing_files),
        "processed_files": 0,
        "successful_files": 0
    }

@app.get("/ingest/status/{task_id}", response_model=IngestStatus)
async def get_ingestion_status(task_id: str):
    """Get status of an ingestion task"""
    if task_id not in background_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task_info = background_tasks[task_id]
    return {
        "task_id": task_id,
        "status": task_info.get("status", "unknown"),
        "message": task_info.get("message", ""),
        "total_files": task_info.get("total_files", 0),
        "processed_files": task_info.get("processed_files", 0),
        "successful_files": task_info.get("successful_files", 0)
    }

# Knowledge graph endpoints
@app.post("/graph/build")
async def build_knowledge_graph(
    background_tasks: BackgroundTasks,
    company_id: Optional[str] = None,
    llm_model: Optional[str] = None,
    embedding_model: Optional[str] = None,
    query: Optional[str] = None,
    num_docs: int = 50,
    visualize: bool = False
):
    """Build knowledge graph from vector store"""
    # Get company info
    config = CompanyConfig()
    
    if company_id:
        try:
            db_path = config.get_db_path(company_id)
            company_models = config.get_company_model_settings(company_id)
            active_company = company_id
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
    else:
        active_company = config.get_active_company()
        db_path = config.get_db_path()
        company_models = config.get_company_model_settings()
    
    if not os.path.exists(db_path):
        raise HTTPException(status_code=404, detail=f"Vector store not found at {db_path}")
    
    # Generate task ID
    import uuid
    task_id = str(uuid.uuid4())
    
    # Create background task
    async def build_graph_task():
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
            if visualize:
                viz_path = os.path.join(VISUALIZATION_DIR, f"{active_company}_knowledge_graph.png")
                visualize_graph(knowledge_graph, viz_path, max_nodes=50)
                return {"status": "completed", "visualization_url": f"/static/{active_company}_knowledge_graph.png"}
            
            return {"status": "completed"}
        except Exception as e:
            logger.error(f"Error building graph: {e}")
            return {"status": "failed", "error": str(e)}
    
    background_tasks.add_task(build_graph_task)
    
    return {"task_id": task_id, "status": "queued", "message": "Knowledge graph building started"}

@app.get("/graph/visualize")
async def visualize_knowledge_graph_endpoint(
    company_id: Optional[str] = None,
    max_nodes: int = 50,
    format: str = "png"
):
    """Visualize knowledge graph"""
    # Get company info
    config = CompanyConfig()
    
    if company_id:
        try:
            db_path = config.get_db_path(company_id)
            active_company = company_id
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
    else:
        active_company = config.get_active_company()
        db_path = config.get_db_path()
    
    # Get knowledge graph
    graph_dir = os.path.join(os.path.dirname(db_path), DEFAULT_GRAPH_DIR)
    if not os.path.exists(graph_dir) or not os.path.exists(os.path.join(graph_dir, "knowledge_graph.pkl")):
        raise HTTPException(status_code=404, detail=f"Knowledge graph not found at {graph_dir}")
    
    knowledge_graph = KnowledgeGraph(graph_dir)
    
    # Generate visualization
    if format.lower() == "mermaid":
        output_path = os.path.join(VISUALIZATION_DIR, f"{active_company}_knowledge_graph.md")
        knowledge_graph.to_mermaid(output_path, max_nodes=max_nodes)
    else:
        output_path = os.path.join(VISUALIZATION_DIR, f"{active_company}_knowledge_graph.png")
        visualize_graph(knowledge_graph, output_path, max_nodes=max_nodes)
    
    # Return visualization URL
    filename = os.path.basename(output_path)
    return {"visualization_url": f"/static/{filename}"}

# Query endpoints
@app.post("/query")
async def query_endpoint(query_request: QueryRequest):
    """Query the RAG system"""
    # Get company info
    config = CompanyConfig()
    sys_config = SystemConfig()
    
    if query_request.company_id:
        try:
            db_path = config.get_db_path(query_request.company_id)
            company_models = config.get_company_model_settings(query_request.company_id)
            active_company = query_request.company_id
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
    else:
        active_company = config.get_active_company()
        db_path = config.get_db_path()
        company_models = config.get_company_model_settings()
    
    if not os.path.exists(db_path):
        raise HTTPException(status_code=404, detail=f"Vector store not found at {db_path}")
    
    # Get embedding model
    embedding_model = query_request.embedding_model or company_models.get("embedding_model")
    
    # Get vector store
    vector_store = get_vector_store(db_path, embedding_model=embedding_model)
    
    # Get top_k from arguments or system settings
    top_k = query_request.num_chunks or sys_config.get_top_k()
    
    # Get knowledge graph if using graph or hybrid retrieval
    knowledge_graph = None
    graph_data = None
    if query_request.retrieval_method in ["graph", "hybrid"]:
        graph_dir = os.path.join(os.path.dirname(db_path), DEFAULT_GRAPH_DIR)
        if not os.path.exists(graph_dir) or not os.path.exists(os.path.join(graph_dir, "knowledge_graph.pkl")):
            if query_request.retrieval_method == "graph":
                raise HTTPException(status_code=404, detail=f"Knowledge graph not found at {graph_dir}")
            else:
                # Fall back to vector search for hybrid mode
                query_request.retrieval_method = "vector"
                logger.warning("Knowledge graph not found, falling back to vector search")
        else:
            knowledge_graph = KnowledgeGraph(graph_dir)
    
    # Get documents based on retrieval method
    if query_request.retrieval_method == "vector" or knowledge_graph is None:
        # Use vector search only
        documents = similarity_search(vector_store, query_request.question, k=top_k)
    elif query_request.retrieval_method == "graph":
        # Use graph search only
        llm_model = query_request.llm_model or company_models.get("llm_model")
        temperature = query_request.temperature or sys_config.get_temperature()
        llm = get_llm_model(llm_model, temperature)
        
        from rag.llm.llm import extract_query_entities
        entities = extract_query_entities(llm, query_request.question)
        
        graph_data = []
        for entity in entities:
            entity_ids = knowledge_graph.search_entities(entity, limit=2)
            for entity_id in entity_ids:
                neighbors = knowledge_graph.get_neighbors(entity_id, max_hops=query_request.num_hops)
                graph_data.extend(neighbors)
        
        # Convert to documents format
        documents = convert_graph_to_documents(graph_data)
    else:
        # Use hybrid search (default)
        llm_model = query_request.llm_model or company_models.get("llm_model")
        temperature = query_request.temperature or sys_config.get_temperature()
        llm = get_llm_model(llm_model, temperature)
        
        documents = hybrid_retrieval(
            vector_store, 
            knowledge_graph, 
            query_request.question, 
            llm,
            k=top_k,
            max_hops=query_request.num_hops
        )
        
        # Extract graph data for explanation if needed
        if query_request.explain:
            graph_data = []
            for doc in documents:
                if doc.metadata.get('source') == 'knowledge_graph':
                    entity_id = doc.metadata.get('entity_id')
                    if entity_id:
                        neighbors = knowledge_graph.get_neighbors(entity_id, max_hops=1)
                        for item in neighbors:
                            if item not in graph_data:
                                graph_data.append(item)
    
    if query_request.raw_chunks:
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
        llm_model = query_request.llm_model or company_models.get("llm_model")
        temperature = query_request.temperature or sys_config.get_temperature()
        
        llm = get_llm_model(llm_model, temperature)
        
        # Generate response
        response = generate_response(llm, documents, query_request.question, custom_template=None, graph_data=graph_data)
        
        result = {
            "question": query_request.question,
            "answer": response.strip(),
            "model": llm_model,
            "temperature": temperature,
            "retrieval_method": query_request.retrieval_method,
            "num_chunks": len(documents),
        }
        
        # Create explanation visualization if requested
        if query_request.explain and graph_data:
            viz_path = os.path.join(VISUALIZATION_DIR, f"query_explanation_{active_company}.png")
            visualize_query_path(knowledge_graph, graph_data, viz_path)
            result["explanation_url"] = f"/static/query_explanation_{active_company}.png"
        
        return result

# Serve uploaded files for download if needed
@app.get("/uploads/{filename}")
async def get_upload(filename: str):
    """Get an uploaded file"""
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)

# Serve visualization files
@app.get("/visualizations/{filename}")
async def get_visualization(filename: str):
    """Get a visualization file"""
    file_path = os.path.join(VISUALIZATION_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Visualization not found")
    return FileResponse(file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)
