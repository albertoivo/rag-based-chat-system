import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from typing import Dict, List, Optional
from pathlib import Path
import os

def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory"""
    backends = {}
    current_dir = Path(".")
    
    # Look for ChromaDB directories
    # Create list of directories that match specific criteria (directory type and name pattern)
    chroma_dirs = [d for d in current_dir.iterdir() if d.is_dir() and 'chroma' in d.name.lower()]

    # Loop through each discovered directory
    for chroma_dir in chroma_dirs:
        # Wrap connection attempt in try-except block for error handling
        try:
            # Initialize database client with directory path
            client = chromadb.PersistentClient(path=str(chroma_dir))
            
            # Retrieve list of available collections from the database
            collections = client.list_collections()
            
            # Loop through each collection found
            for collection in collections:
                # Create unique identifier key combining directory and collection names
                backend_key = f"{chroma_dir.name}_{collection.name}"
                
                # Build information dictionary containing:
                backend_info = {
                    # Store directory path as string
                    "directory": str(chroma_dir),
                    # Store collection name
                    "collection_name": collection.name,
                    # Create user-friendly display name
                    "display_name": f"{chroma_dir.name} - {collection.name}",
                }
                
                # Get document count with fallback for unsupported operations
                try:
                    doc_count = collection.count()
                    backend_info["doc_count"] = doc_count
                except:
                    backend_info["doc_count"] = 0
                
                # Add collection information to backends dictionary
                backends[backend_key] = backend_info
        
        # Handle connection or access errors gracefully
        except Exception as e:
            # Create fallback entry for inaccessible directories
            error_msg = str(e)[:50]
            # Include error information in display name with truncation
            backends[chroma_dir.name] = {
                "directory": str(chroma_dir),
                "collection_name": "error",
                # Set appropriate fallback values for missing information
                "display_name": f"{chroma_dir.name} (Error: {error_msg}...)",
                "doc_count": 0
            }

    # Return complete backends dictionary with all discovered collections
    return backends

def initialize_rag_system(chroma_dir: str, collection_name: str, 
                          openai_api_key: Optional[str] = None,
                          embedding_model: str = "text-embedding-3-small"):
    """
    Initialize the RAG system with specified backend.
    
    Args:
        chroma_dir: Path to ChromaDB persistence directory
        collection_name: Name of the collection to use
        openai_api_key: OpenAI API key for embedding queries. 
                        If not provided, will try to use CHROMA_OPENAI_API_KEY env var.
        embedding_model: OpenAI embedding model to use (default: text-embedding-3-small)
    
    Returns:
        Tuple of (collection, success: bool, error: str or None)
    """
    # Get API key from parameter or environment variable
    api_key = openai_api_key or os.environ.get("CHROMA_OPENAI_API_KEY")
    
    if not api_key:
        return None, False, "OpenAI API key not provided and CHROMA_OPENAI_API_KEY not set"
    
    # Create OpenAI embedding function for query embedding
    # This ensures the user question is properly embedded when querying
    embedding_function = OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name=embedding_model
    )
    
    # Create a ChromaDB PersistentClient
    client = chromadb.PersistentClient(path=chroma_dir)
    
    # Return the collection with the embedding function
    # This ensures collection.query() can embed query_texts properly
    collection = client.get_collection(
        name=collection_name,
        embedding_function=embedding_function
    )
    return collection, True, None

def retrieve_documents(collection, query: str, n_results: int = 3, 
                      mission_filter: Optional[str] = None) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering"""

    # Initialize filter variable to None (represents no filtering)
    filter_dict = None

    # Check if filter parameter exists and is not set to "all" or equivalent
    # If filter conditions are met, create filter dictionary with appropriate field-value pairs
    if mission_filter and mission_filter != "all":
        filter_dict = {"mission": {"$eq": mission_filter}}

    # Execute database query with the following parameters:
    results = collection.query(
        # Pass search query in the required format
        query_texts=[query],
        # Set maximum number of results to return
        n_results=n_results,
        # Apply conditional filter (None for no filtering, dictionary for specific filtering)
        where=filter_dict
    )

    # Return query results to caller
    return results

def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Format retrieved documents into context"""
    if not documents:
        return ""
    
    # Initialize list with header text for context section
    context_parts = ["=== RETRIEVED CONTEXT ==="]
    
    seen = set()
    count = 0

    # Loop through paired documents and their metadata using enumeration
    for doc, metadata in zip(documents, metadatas):
        if doc in seen:
            continue
        seen.add(doc)
        count += 1

        # Extract mission information from metadata with fallback value
        mission = metadata.get("mission", "Unknown Mission")
        # Clean up mission name formatting (replace underscores, capitalize)
        mission = mission.replace("_", " ").title()
        
        # Extract source information from metadata with fallback value  
        source = metadata.get("source", "Unknown Source")
        
        # Extract category information from metadata with fallback value
        # Note: ingestion uses 'document_category' as the metadata key
        category = metadata.get("document_category", "Unknown Category")
        # Clean up category name formatting (replace underscores, capitalize)
        category = category.replace("_", " ").title()
        
        # Create formatted source header with index number and extracted information
        source_header = f"\n[Source {count}] Mission: {mission} | Category: {category} | File: {source}"
        # Add source header to context parts list
        context_parts.append(source_header)
        
        # Check document length and truncate if necessary
        if len(doc) > 500:
            # Add truncated or full document content to context parts list
            context_parts.append(doc[:500] + "...")
        else:
            # Add truncated or full document content to context parts list
            context_parts.append(doc)

    # Join all context parts with newlines and return formatted string
    return "\n".join(context_parts)