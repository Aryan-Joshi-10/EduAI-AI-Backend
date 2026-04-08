"""
Dependency injection (LLM init, Auth checks).
Uses lru_cache to ensure singletons are initialized only once.
"""
import os
import base64
import logging
from functools import lru_cache
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser
from google import genai
from config import (
    EMBEDDINGS_MODEL_NAME,
    GROQ_API_KEY,
    GROQ_MODEL,
    GCP_PROJECT,
    GCP_CREDS_B64,
    OPENAI_API_KEY,
    OPENAI_MODEL
)

logger = logging.getLogger(__name__)

# Global Gemini client (created once, used everywhere)
_gemini_client = None


def get_gemini_client():
    """Get or create the global Gemini client."""
    global _gemini_client
    if _gemini_client is None:
        try:
            # Handle Google Cloud Credentials
            if GCP_CREDS_B64:
                try:
                    # Decode the Base64 string
                    creds_json = base64.b64decode(GCP_CREDS_B64)
                    # Define a temporary path in /tmp (which is writable)
                    creds_path = "/tmp/gcp_creds.json"
                    # Write the decoded JSON to the temp file
                    with open(creds_path, "wb") as f:
                        f.write(creds_json)
                    # Set the environment variable for Vertex AI
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
                    logger.info("Google Cloud credentials loaded from secret.")
                except Exception as e:
                    logger.error(f"Failed to decode/write GCP credentials: {e}", exc_info=True)
                    raise
            
            # Create global client
            _gemini_client = genai.Client(
                vertexai=True,
                project=GCP_PROJECT,
                location="global"
            )
            logger.info("Gemini client initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}", exc_info=True)
            raise
    return _gemini_client


def initialize_vertex_ai():
    """Initialize Gemini client (kept for backward compatibility)."""
    get_gemini_client()


@lru_cache()
def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Get embeddings model instance (singleton).
    Uses lru_cache to ensure only one instance is created.
    """
    try:
        logger.info(f"Initializing embeddings with model: {EMBEDDINGS_MODEL_NAME}")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)
        logger.info("Embeddings initialized successfully")
        return embeddings
    except Exception as e:
        logger.error(f"Failed to initialize embeddings: {e}", exc_info=True)
        raise


@lru_cache()
def get_llm() -> ChatGroq:
    """
    Get LLM instance (singleton).
    Uses lru_cache to ensure only one instance is created.
    """
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY environment variable is not set")
    
    try:
        logger.info("Initializing ChatGroq LLM")
        llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY)
        logger.info("LLM initialized successfully")
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}", exc_info=True)
        raise


@lru_cache()
def get_json_parser() -> JsonOutputParser:
    """
    Get JSON output parser instance (singleton).
    """
    return JsonOutputParser()


# Global OpenAI client (created once, used everywhere)
_openai_client = None


def get_openai_client():
    """Get or create the global OpenAI client."""
    global _openai_client
    if _openai_client is None:
        try:
            if not OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY environment variable is not set")
            
            from openai import OpenAI
            _openai_client = OpenAI(api_key=OPENAI_API_KEY)
            logger.info("OpenAI client initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}", exc_info=True)
            raise
    return _openai_client


def get_user_id_from_request(user_id: str = None) -> str:
    """
    Extract and validate user_id from request.
    In production, this should validate against auth tokens.
    
    Args:
        user_id: User ID from request (header/form)
    
    Returns:
        Validated user_id string
    
    Raises:
        ValueError: If user_id is missing or invalid
    """
    if not user_id:
        raise ValueError("user_id is required for data isolation")
    
    # TODO: Add authentication/authorization logic here
    # For now, just return the user_id as-is
    
    return user_id
