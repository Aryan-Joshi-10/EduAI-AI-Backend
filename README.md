# 📚 EDUAI Assessment Engine (Backend)

**A Scalable, Multi-Tenant RAG System for Automated Exam Generation & Evaluation.**

This backend engine leverages **Retrieval Augmented Generation (RAG)** to ingest educational textbooks (PDFs), understand their semantic structure, and generate high-quality question papers or evaluate student answers using **Google Gemini** and **Qdrant Vector Database**.

---

## 🚀 Key Architectural Highlights

This project was recently refactored from a monolithic script to a distributed micro-service architecture to support concurrency and scale.

### 1. Hybrid PDF Processing Engine ("Map-Then-Process")

Standard PDF chunking loses context (e.g., a chunk from page 50 loses the fact that it belongs to "Chapter 3"). We implemented a custom 2-phase pipeline:

- **Phase 1 (Stateful):** Sequentially scans the first 15-20 pages to extract the **Table of Contents** and build a `Page -> Chapter` map.
- **Phase 2 (Parallel):** Uses a `ThreadPoolExecutor` to process the remaining pages in parallel (10x speedup). The "Chapter Map" is injected into every thread, ensuring every text chunk is "stamped" with its correct metadata before embedding.

### 2. Multi-Tenancy & Data Isolation

Moved from local file-based storage (FAISS) to **Qdrant Cloud**.

- **Strict Isolation:** Every document chunk is tagged with a `user_id` in its payload.
- **Filtered Retrieval:** All RAG queries enforce a strict metadata filter (`must: [{key: "user_id", match: ...}]`), ensuring users can only retrieve from their own uploaded textbooks.

### 3. Resilient Data Ingestion

- **Batch Uploading:** Vectors are uploaded in batches of 64 to prevent `httpx.WriteTimeout` errors common with large textbooks (300+ pages).
- **Background Tasks:** PDF processing is offloaded to `fastapi.BackgroundTasks`, returning an immediate "Processing Started" response to the client to prevent HTTP blocking.

---

## 🛠 Tech Stack

| Component         | Technology                        | Role                                            |
| ----------------- | --------------------------------- | ----------------------------------------------- |
| **Framework**     | **FastAPI** (Python 3.10+)        | High-performance Async API                      |
| **Vector DB**     | **Qdrant Cloud**                  | Scalable vector storage with Metadata Filtering |
| **LLM**           | **Google Gemini 3 Flash Preview** | OCR, Reasoning, and Content Generation          |
| **Orchestration** | **LangChain**                     | Document Splitting & Embedding workflows        |
| **Embeddings**    | **all-MiniLM-L6-v2**              | 384-dimensional dense vector embeddings         |
| **Concurrency**   | **ThreadPoolExecutor**            | Parallelizing CPU-bound OCR tasks               |

---

## 📂 Directory Structure

```text
backend/
├── main.py                 # Application Entry Point & Exception Handlers
├── config.py               # Environment Configuration (Pydantic)
├── database.py             # Qdrant Client Setup & Index Initialization
├── dependencies.py         # Singleton Dependency Injection (LLM/Embeddings)
├── requirements.txt        # Pinned dependencies
├── services/               # --- Business Logic Layer ---
│   ├── pdf_processing.py   # OCR logic, Text Cleaning, Hybrid Parsing
│   ├── vector_store.py     # Batch Uploads, Search, & Delete logic
│   ├── exam_generator.py   # RAG Context Retrieval & Exam Generation
│   └── grading_service.py  # Student Answer Evaluation Logic
└── routers/                # --- API Interface Layer ---
    ├── pdf_routes.py       # Endpoints for File Uploads
    ├── exam_routes.py      # Endpoints for Question Generation
    └── eval_routes.py      # Endpoints for Grading

```

---

## ⚡ Setup & Installation

### 1. Prerequisites

- Python 3.12
- Qdrant Cloud Account (Free Tier is sufficient)
- Groq API Key / Open AI API key

### 2. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd backend

# Create Virtual Environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install Dependencies
pip install -r requirements.txt

```

### 3. Configuration (.env)

Create a `.env` file in the root directory:

```ini
# Server
PORT=8000
ENVIRONMENT=development

# Qdrant Vector Database
QDRANT_URL=https://<your-cluster>.us-east4-0.gcp.cloud.qdrant.io:6333
QDRANT_API_KEY=<your-qdrant-api-key>

OPENAI_API_KEY =
    OR
GROQ_API_KEY = 

# Model Config
EMBEDDINGS_MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"

```

### 4. Run the Server

```bash
# The application will auto-index required fields on startup
uvicorn main:app --reload

```

---

## 📡 API Reference

**Authentication:** All endpoints require the `X-User-ID` header to identify the tenant.

### 1. Document Management

#### `POST /api/docs/process_pdf/`

Uploads a textbook, performs OCR, extracts chapters, chunks text, and indexes it in Qdrant.

- **Headers:** `X-User-ID: <string>`
- **Body (Form-Data):**
- `file`: (Binary PDF)
- `subject_data`: (JSON String) `{"subject": "Math", "class": "9", "pdf_name": "math_book"}`

- **Response:**

```json
{
  "status": "Processing started",
  "filename": "math_book.pdf",
  "message": "The file is being processed in the background."
}
```

#### `GET /api/docs/chunks/`

Debug endpoint to verify uploaded data for a user.

- **Headers:** `X-User-ID: <string>`
- **Response:** List of stored text chunks and metadata.

---

### 2. Exam Generation

#### `POST /api/exams/generate_question_paper/`

Generates a structured exam paper based on specific chapters/topics.

- **Headers:** `X-User-ID: <string>`
- **Body (JSON):**

```json
{
  "subject": "Maths",
  "class": "9",
  "pdf_name": "math_book",
  "questions": [
    {
      "type": "multiple choice",
      "numQuestions": 5,
      "marks": 1,
      "topics": ["Polynomials", "Number Systems"],
      "llm_note": "Focus on difficult conceptual questions"
    }
  ]
}
```

- **Response:** A structured JSON object containing the exam questions and an answer key.

---

### 3. Evaluation

#### `POST /api/eval/evaluate_answer_paper/`

Compares a student's handwritten (scanned) answer sheet against the generated question paper.

- **Headers:** `X-User-ID: <string>`
- **Body (Form-Data):**
- `file`: (Binary PDF/Image of Student Answers)
- `question_paper_str`: (String) The JSON output from the generation step.

- **Response:** Detailed grading report, marks per question, and suggestions for improvement.

---

## 🧠 Architecture Decision Records (ADR)

### Why Qdrant over FAISS?

- **Problem:** FAISS (Local) is not thread-safe for concurrent writes. If User A uploaded a file while User B generated an exam, the index file would lock or corrupt.
- **Solution:** Qdrant is a server-based Vector DB. It handles concurrency natively and allows filtering by metadata (`user_id`), enabling secure multi-tenancy on a single database instance.

### Why Batch Uploads?

- **Problem:** Uploading 300+ pages of vector embeddings in a single HTTP request resulted in `httpx.WriteTimeout` exceptions.
- **Solution:** We implemented a batching logic in `services/vector_store.py` that splits payloads into groups of 64 chunks. This ensures stability even on slow network connections.

### Why Hybrid Parsing?

- **Problem:** Pure parallel processing is fast but stateless (Page 10 doesn't know about Page 9). Pure sequential processing is context-aware but slow (8 mins/book).
- **Solution:** We scan the TOC sequentially to map page numbers to chapters, then pass this map to parallel workers. This gives us **Context Awareness + Parallel Speed**.

---

## 📄 License

Question Paper Generation @2025
