"""
Settings & Environment variables loading.
"""
import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Qdrant Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # Optional for local

# Embeddings Configuration
EMBEDDINGS_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DIM = 384  # Dimension for all-MiniLM-L6-v2

# LLM Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"

# Google Vertex AI Configuration
GCP_PROJECT = "gen-lang-client-0238295665"
GCP_LOCATION = "us-central1"
GCP_CREDS_B64 = os.getenv("GCP_CREDS_B64")  # Base64 encoded credentials
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4o-mini"  # Default model for PDF processing

# Qdrant Collection Name
COLLECTION_NAME = "document_chunks"

# CORS Configuration
CORS_ORIGINS = ["http://localhost:5173"]

# Logging Configuration
LOGGING_LEVEL = os.getenv("LOGGING_LEVEL", "INFO")
