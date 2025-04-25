# Model-Choose GraphRAG API

A powerful API service that combines RAG (Retrieval-Augmented Generation), Knowledge Graph, and Model Selection capabilities.

## Features

- **📚 Multi-Document Support**: Process various document formats including PDF, DOCX, TXT, CSV, and more
- **🔍 Advanced OCR**: Extract text from scanned documents and images
- **🌐 Knowledge Graph**: Build and utilize knowledge graphs from documents
- **🤖 Model Selection**: Choose from different LLM and embedding models
- **🏢 Multi-Company Support**: Manage multiple separate knowledge bases
- **🔄 Hybrid Retrieval**: Combine vector search and knowledge graph for better results
- **📊 Visualizations**: Generate visual representations of knowledge graphs
- **🔄 Background Processing**: Asynchronous document ingestion and graph building

## API Endpoints

### Root
- `GET /`: Welcome message

### Company Management
- `GET /companies`: List all companies
- `GET /companies/active`: Get the active company
- `POST /companies`: Create a new company
- `PUT /companies/{company_id}/active`: Set a company as active
- `PUT /companies/{company_id}/models`: Update company models
- `DELETE /companies/{company_id}`: Delete a company

### Model Management
- `GET /models/llm`: List available LLM models
- `GET /models/embeddings`: List available embedding models
- `GET /settings`: Get system settings
- `PUT /settings`: Update system settings

### Document Ingestion
- `POST /ingest/upload`: Upload files for ingestion
- `POST /ingest/process`: Process uploaded files (background task)
- `GET /ingest/status/{task_id}`: Check ingestion task status

### Knowledge Graph
- `POST /graph/build`: Build knowledge graph from vector store
- `GET /graph/visualize`: Visualize knowledge graph

### Query Endpoints
- `POST /query`: Query the RAG system with various options

### Utility Endpoints
- `GET /uploads/{filename}`: Retrieve an uploaded file
- `GET /visualizations/{filename}`: Retrieve a visualization file

## Setup

1. Ensure you have Python 3.8+ installed
2. Clone this repository
3. Run the setup script:
   ```
   chmod +x setup.sh
   ./setup.sh
   ```

## Running the API

Start the API server:
```
chmod +x run.sh
./run.sh
```

The API will be available at http://localhost:8765, and the interactive API documentation is available at http://localhost:8765/docs

## Examples

### Creating a Company

```
curl -X 'POST' \
  'http://localhost:8000/companies' \
  -H 'Content-Type: application/json' \
  -d '{
  "id": "company1",
  "name": "My Company",
  "description": "Example company",
  "set_active": true
}'
```

### Ingesting Documents

1. Upload files:
```
curl -X 'POST' \
  'http://localhost:8000/ingest/upload' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'files=@document1.pdf' \
  -F 'files=@document2.docx'
```

2. Process files:
```
curl -X 'POST' \
  'http://localhost:8000/ingest/process' \
  -H 'Content-Type: application/json' \
  -d '{
  "file_paths": ["/path/to/uploaded/document1.pdf", "/path/to/uploaded/document2.docx"],
  "build_graph": true
}'
```

### Querying

```
curl -X 'POST' \
  'http://localhost:8000/query' \
  -H 'Content-Type: application/json' \
  -d '{
  "question": "What are the key features of our product?",
  "retrieval_method": "hybrid",
  "num_chunks": 5
}'
```

## Dependencies

- FastAPI: Web framework
- Langchain: RAG components
- ChromaDB: Vector store
- NetworkX: Knowledge graph
- PyMuPDF, pytesseract, easyocr: Document processing and OCR
- And more (see requirements.txt)

## License

MIT License
