"""
Retrieval module for GraphRAG system
"""
from typing import List, Dict, Any
import json
from langchain_core.documents import Document
from rag.vector_store import similarity_search
from rag.knowledge_graph.graph import KnowledgeGraph
from rag.llm.llm import get_llm_model, extract_query_entities

def merge_and_rank_results(vector_results: List[Document], graph_results: List[Document], query: str) -> List[Document]:
    """
    Merge and rank results from vector search and graph search
    
    Args:
        vector_results: Results from vector search
        graph_results: Results from graph search
        query: Original query
        
    Returns:
        Merged and ranked results with improved ranking
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    
    # Function to compute relevance score using TF-IDF
    def compute_relevance_scores(docs: List[Document], query: str) -> List[float]:
        if not docs:
            return []
            
        # Extract text content
        texts = [doc.page_content for doc in docs] + [query]
        
        try:
            # Create TF-IDF vectorizer
            vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # Compute similarity between query and each document
            query_vector = tfidf_matrix[-1:] 
            doc_vectors = tfidf_matrix[:-1]
            similarities = cosine_similarity(query_vector, doc_vectors)[0]
            
            return similarities.tolist()
        except Exception as e:
            print(f"Error computing TF-IDF scores: {e}")
            # Fallback: return uniform scores
            return [0.5] * len(docs)
    
    # Combine all results and score them
    combined_results = []
    vector_contents = {doc.page_content for doc in vector_results}
    
    # Add vector results with source tag
    for i, doc in enumerate(vector_results):
        doc.metadata["score_source"] = "vector"
        doc.metadata["original_rank"] = i
        combined_results.append(doc)
    
    # Add graph results if not duplicates
    for i, doc in enumerate(graph_results):
        if doc.page_content not in vector_contents:
            doc.metadata["score_source"] = "graph"
            doc.metadata["original_rank"] = i
            combined_results.append(doc)
    
    # Compute relevance scores
    scores = compute_relevance_scores(combined_results, query)
    
    # Add scores to metadata and prioritize
    for i, (doc, score) in enumerate(zip(combined_results, scores)):
        # Create a priority score that considers both TF-IDF and source
        # Weight vector results slightly higher (proven by direct matching)
        source_boost = 1.2 if doc.metadata.get("score_source") == "vector" else 1.0
        
        # Boost entities that directly match a query term
        entity_name = doc.metadata.get("entity_name", "").lower()
        name_match_boost = 1.3 if entity_name and any(term.lower() in entity_name for term in query.split()) else 1.0
        
        # Combine all factors
        doc.metadata["relevance_score"] = score
        doc.metadata["priority_score"] = score * source_boost * name_match_boost
    
    # Sort by priority score
    combined_results.sort(key=lambda doc: doc.metadata.get("priority_score", 0), reverse=True)
    
    return combined_results

def convert_graph_to_documents(graph_data: List[Dict[str, Any]]) -> List[Document]:
    """
    Convert graph data to documents
    
    Args:
        graph_data: Graph data from KnowledgeGraph
        
    Returns:
        List of documents
    """
    documents = []
    
    # Group by entities
    entities = {}
    relations = []
    
    for item in graph_data:
        if item["type"] == "entity":
            entity_id = item["id"]
            entities[entity_id] = item
        elif item["type"] == "relation":
            relations.append(item)
    
    # Create documents for entities with their relations
    for entity_id, entity in entities.items():
        # Find relations for this entity
        entity_relations = [r for r in relations if r["source"] == entity_id or r["target"] == entity_id]
        
        # Create content
        content = f"Entity: {entity['name']} (Type: {entity['entity_type']})\n"
        
        # Add attributes
        attributes = entity.get("attributes", {})
        if attributes:
            content += "Attributes:\n"
            for key, value in attributes.items():
                if key != "sources":
                    content += f"- {key}: {value}\n"
        
        # Add relations
        if entity_relations:
            content += "Relationships:\n"
            for rel in entity_relations:
                if rel["source"] == entity_id:
                    content += f"- {rel['relation_type']} -> {rel['target_name']}\n"
                else:
                    content += f"- {rel['source_name']} -> {rel['relation_type']} -> {entity['name']}\n"
        
        # Create document
        sources = attributes.get("sources", ["knowledge_graph"])
        doc = Document(
            page_content=content,
            metadata={
                "source": "knowledge_graph",
                "entity_id": entity_id,
                "entity_name": entity["name"],
                "entity_type": entity["entity_type"],
                "original_sources": sources
            }
        )
        documents.append(doc)
    
    return documents

def hybrid_retrieval(vector_store, knowledge_graph: KnowledgeGraph, query: str, llm=None, k: int = 4, max_hops: int = 1) -> List[Document]:
    """
    Perform hybrid retrieval using both vector store and knowledge graph
    
    Args:
        vector_store: Vector store
        knowledge_graph: Knowledge graph
        query: Query to search for
        llm: LLM model (if None, a new one will be created)
        k: Number of results to return
        max_hops: Maximum number of hops for graph traversal
        
    Returns:
        List of documents sorted by relevance
    """
    if llm is None:
        llm = get_llm_model()
    
    # Detect query language for better processing
    from rag.knowledge_graph.graph import contains_thai
    is_thai_query = contains_thai(query)
    
    # Step 1: Vector search to find relevant documents
    print("Performing vector search...")
    try:
        vector_results = similarity_search(vector_store, query, k=k)
    except Exception as e:
        print(f"Vector search error: {e}")
        vector_results = []
    
    # Step 2: Extract entities from the query with error handling
    print("Extracting entities from query...")
    try:
        entities = extract_query_entities(llm, query)
        print(f"Extracted entities: {entities}")
    except Exception as e:
        print(f"Entity extraction error: {e}. Using fallback method.")
        # Simple fallback entity extraction: nouns from the query
        import re
        entities = [word for word in re.findall(r'\b[A-Z][a-z]*\b', query)]  # Capitalized words
        if not entities:
            entities = [word for word in query.split() if len(word) > 3]  # Words longer than 3 chars
        print(f"Fallback entity extraction: {entities}")
    
    # Step 3: Find related entities in the graph
    print("Searching knowledge graph...")
    graph_results = []
    
    if knowledge_graph and entities:
        for entity in entities:
            try:
                # Search for entity in graph
                entity_ids = knowledge_graph.search_entities(entity, limit=2)
                
                for entity_id in entity_ids:
                    # Get neighbors up to max_hops away
                    neighbors = knowledge_graph.get_neighbors(entity_id, max_hops=max_hops)
                    
                    # Convert graph results to documents
                    if neighbors:
                        graph_docs = convert_graph_to_documents(neighbors)
                        graph_results.extend(graph_docs)
            except Exception as e:
                print(f"Error searching entity '{entity}' in knowledge graph: {e}")
    
    print(f"Found {len(graph_results)} relevant items in knowledge graph")
    
    # Step 4: Combine and rank results
    combined_results = merge_and_rank_results(vector_results, graph_results, query)
    
    # Handle empty results case with meaningful error message
    if len(combined_results) == 0:
        # Create a special document indicating no results found
        from langchain_core.documents import Document
        error_message = f"ไม่พบข้อมูลที่เกี่ยวข้องกับคำค้นหา" if is_thai_query else "No relevant information found for the query"
        fallback_doc = Document(
            page_content=error_message,
            metadata={
                "source": "system_message",
                "error": "no_results_found",
                "query": query,
                "retrieval_method": "hybrid"
            }
        )
        return [fallback_doc]
    
    # Calculate optimal return size based on result quality
    if len(combined_results) > 0:
        # Calculate a dynamic size based on available high-quality results
        # (those with priority score above a certain threshold)
        high_quality_count = sum(1 for doc in combined_results 
                               if doc.metadata.get("priority_score", 0) > 0.5)
        
        # At minimum return k, or up to 2*k if we have good results
        return_size = max(k, min(k + high_quality_count, k*2))
    else:
        return_size = k
    
    # Add retrieval metadata to each returned document
    for i, doc in enumerate(combined_results[:return_size]):
        doc.metadata["retrieval_rank"] = i
        doc.metadata["retrieval_method"] = "hybrid"
        # Add language detection for better processing
        doc.metadata["contains_thai"] = contains_thai(doc.page_content)
    
    return combined_results[:return_size]
