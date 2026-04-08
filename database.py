"""
Qdrant Client setup and Collection initialization.
Handles connection to Qdrant (Cloud or Docker) and ensures collections exist.
"""
import os
import logging
from typing import Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, CollectionStatus, PayloadSchemaType

logger = logging.getLogger(__name__)

# Collection name for storing document chunks
COLLECTION_NAME = "document_chunks"
# Vector dimension for sentence-transformers/all-MiniLM-L6-v2
VECTOR_DIM = 384


def get_qdrant_client() -> QdrantClient:
    """
    Initialize and return QdrantClient.
    Supports both Qdrant Cloud (via URL + API key) and local Docker instance.
    
    Environment Variables:
        QDRANT_URL: Qdrant server URL (e.g., "https://xxx.qdrant.io" or "http://localhost:6333")
        QDRANT_API_KEY: API key for Qdrant Cloud (optional for local)
    
    Returns:
        QdrantClient instance
    """
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    
    if not qdrant_url:
        raise ValueError("QDRANT_URL environment variable is required")
    
    try:
        if qdrant_api_key:
            # Qdrant Cloud connection
            client = QdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key
            )
            logger.info(f"Connected to Qdrant Cloud at {qdrant_url}")
        else:
            # Local Docker connection
            client = QdrantClient(url=qdrant_url)
            logger.info(f"Connected to local Qdrant at {qdrant_url}")
        
        return client
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {e}", exc_info=True)
        raise


def ensure_collection_exists(client: QdrantClient, collection_name: str = COLLECTION_NAME) -> None:
    """
    Ensure the Qdrant collection exists. Create it if it doesn't.
    Also ensures user_id, pdf_name, and chapter_no payload indexes exist for efficient filtering.
    
    Args:
        client: QdrantClient instance
        collection_name: Name of the collection to ensure exists
    """
    try:
        # Check if collection exists
        collections = client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        collection_created = False
        if collection_name not in collection_names:
            logger.info(f"Collection '{collection_name}' not found. Creating...")
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=VECTOR_DIM,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Collection '{collection_name}' created successfully")
            collection_created = True
        else:
            logger.debug(f"Collection '{collection_name}' already exists")
        
        # Ensure required indexes exist for filtering
        # These indexes are required for efficient queries in Qdrant
        indexes_to_create = [
            ("user_id", "user isolation"),
            ("pdf_name", "PDF filtering"),
            ("chapter_no", "chapter filtering"),
        ]
        
        try:
            # Check if collection exists by trying to get collection info
            collection_info = client.get_collection(collection_name)
            # If we get here, collection exists - now create indexes
            
            for field_name, description in indexes_to_create:
                try:
                    client.create_payload_index(
                        collection_name=collection_name,
                        field_name=field_name,
                        field_schema=PayloadSchemaType.KEYWORD
                    )
                    logger.info(f"Created payload index for '{field_name}' ({description}) in collection '{collection_name}'")
                except Exception as index_error:
                    # Index might already exist, which is fine
                    if "already exists" in str(index_error).lower() or "duplicate" in str(index_error).lower():
                        logger.debug(f"Payload index for '{field_name}' already exists in collection '{collection_name}'")
                    else:
                        logger.warning(f"Could not create payload index for '{field_name}': {index_error}")
        except Exception as e:
            logger.warning(f"Error ensuring indexes exist: {e}")
            # Don't raise - index creation is best effort, filtering will still work but may be slower
            
    except Exception as e:
        logger.error(f"Error ensuring collection exists: {e}", exc_info=True)
        raise


def get_collection_info(client: QdrantClient, collection_name: str = COLLECTION_NAME) -> dict:
    """
    Get information about the collection.
    
    Args:
        client: QdrantClient instance
        collection_name: Name of the collection
    
    Returns:
        Dictionary with collection information
    """
    try:
        collection_info = client.get_collection(collection_name)
        return {
            "name": collection_name,
            "vectors_count": collection_info.points_count,
            "status": collection_info.status
        }
    except Exception as e:
        logger.error(f"Error getting collection info: {e}", exc_info=True)
        raise

