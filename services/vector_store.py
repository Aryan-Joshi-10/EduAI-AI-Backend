"""
Qdrant upload/retrieval logic with user isolation.
Works across ALL qdrant-client versions (REST / old / new).
"""

import uuid
import logging
from typing import List, Dict, Optional, Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
    FilterSelector,
    PointStruct
)
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "document_chunks"
BATCH_SIZE = 64  # <--- NEW: Upload in small batches to prevent Timeout

# ---------------------------------------------------------------------
# UPLOAD CHUNKS
# ---------------------------------------------------------------------

def upload_chunks_to_qdrant(
    client: QdrantClient,
    chunks: List[Document],
    embeddings: HuggingFaceEmbeddings,
    user_id: str,
    collection_name: str = COLLECTION_NAME,
) -> int:
    """
    Uploads chunks to Qdrant with mandatory user_id isolation.
    Uses Batching to prevent WriteTimeouts.
    """
    
    # Unwrap LangChain wrapper if present
    if hasattr(client, "client"):
        client = client.client

    if not chunks:
        logger.warning("No chunks provided for upload")
        return 0

    if not user_id:
        raise ValueError("user_id is required for data isolation")

    logger.info(f"Generating embeddings for {len(chunks)} chunks...")
    texts = [c.page_content for c in chunks]
    vectors = embeddings.embed_documents(texts)

    points = []
    for chunk, vector in zip(chunks, vectors):
        metadata = chunk.metadata.copy()
        metadata["user_id"] = user_id

        # Qdrant payload must be JSON serializable
        payload = {"page_content": chunk.page_content}
        for k, v in metadata.items():
            payload[k] = v if isinstance(v, (str, int, float, bool, type(None))) else str(v)

        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload=payload
            )
        )

    # --- BATCH UPLOAD LOGIC ---
    total_points = len(points)
    for i in range(0, total_points, BATCH_SIZE):
        batch = points[i : i + BATCH_SIZE]
        logger.info(f"Uploading batch {i//BATCH_SIZE + 1} ({len(batch)} chunks)...")
        try:
            client.upsert(collection_name=collection_name, points=batch)
        except Exception as e:
            logger.error(f"Failed to upload batch starting at index {i}: {e}")
            raise e

    logger.info(f"Successfully uploaded {total_points} chunks for user {user_id}")
    return total_points

# ---------------------------------------------------------------------
# VECTOR SEARCH (FIXED ARGUMENT NAMES)
# ---------------------------------------------------------------------

def search_similar_chunks(
    client: Any, 
    query_text: str,
    embeddings: HuggingFaceEmbeddings,
    user_id: str,
    k: int = 10,
    collection_name: str = COLLECTION_NAME,
    additional_filters: Optional[Dict] = None,
) -> List[Document]:
    """
    Searches Qdrant with strict user_id filtering.
    """

    # 1. OPTIMISTIC UNWRAP
    if hasattr(client, "client") and not isinstance(client, QdrantClient):
        client = client.client

    if not user_id:
        raise ValueError("user_id is required")

    logger.info(f"Searching similar chunks for user {user_id}")
    query_vector = embeddings.embed_query(query_text)

    # 2. Build Strict User Filter
    must_conditions = [
        FieldCondition(key="user_id", match=MatchValue(value=user_id))
    ]

    if additional_filters:
        for kf, vf in additional_filters.items():
            if vf is not None:
                must_conditions.append(
                    FieldCondition(key=kf, match=MatchValue(value=vf))
                )

    search_filter = Filter(must=must_conditions)
    results = []

    # 3. Execute Search
    try:
        if hasattr(client, "search"):
            results = client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=k,
                query_filter=search_filter, 
            )
        else:
             raise AttributeError("search method missing")

    except (TypeError, ValueError, AttributeError):
        try:
            # Fallback for newer Qdrant versions
             results = client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=k,
                query_filter=search_filter,
            ).points
        except Exception as e:
             logger.error(f"All search methods failed: {e}")
             return []
                 
    except Exception as e:
        logger.error(f"Qdrant search failed: {e}")
        return []

    # 4. Convert Results
    documents: List[Document] = []
    for point in results:
        payload = point.payload.copy() if point.payload else {}
        page_content = payload.pop("page_content", "")
        payload.pop("user_id", None)

        documents.append(
            Document(
                page_content=page_content,
                metadata=payload,
            )
        )

    logger.info(f"Retrieved {len(documents)} chunks")
    return documents

# ---------------------------------------------------------------------
# DELETE BY PDF
# ---------------------------------------------------------------------

def delete_chunks_by_pdf(
    client: QdrantClient,
    pdf_name: str,
    user_id: str,
    collection_name: str = COLLECTION_NAME,
) -> int:
    
    if hasattr(client, "client"):
        client = client.client

    delete_filter = Filter(
        must=[
            FieldCondition(key="user_id", match=MatchValue(value=user_id)),
            FieldCondition(key="pdf_name", match=MatchValue(value=pdf_name)),
        ]
    )

    client.delete(
        collection_name=collection_name,
        points_selector=FilterSelector(filter=delete_filter),
    )

    logger.info(f"Deleted chunks for PDF '{pdf_name}' (user={user_id})")
    return 1

# ---------------------------------------------------------------------
# SCROLL / GET ALL
# ---------------------------------------------------------------------

def get_chunks_by_filter(
    client: QdrantClient,
    user_id: str,
    collection_name: str = COLLECTION_NAME,
    filters: Optional[Dict] = None,
    limit: int = 10000,
) -> List[Document]:
    
    if hasattr(client, "client"):
        client = client.client

    must_conditions = [
        FieldCondition(key="user_id", match=MatchValue(value=user_id))
    ]

    if filters:
        for kf, vf in filters.items():
            if vf is not None:
                must_conditions.append(
                    FieldCondition(key=kf, match=MatchValue(value=vf))
                )

    search_filter = Filter(must=must_conditions)

    points, _ = client.scroll(
        collection_name=collection_name,
        scroll_filter=search_filter,
        limit=limit,
    )

    documents = []
    for point in points:
        payload = point.payload.copy() if point.payload else {}
        page_content = payload.pop("page_content", "")
        payload.pop("user_id", None)

        documents.append(
            Document(
                page_content=page_content,
                metadata=payload,
            )
        )

    return documents