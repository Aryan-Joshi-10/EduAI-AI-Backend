"""
Entry point (FastAPI app init & routes inclusion).
"""
import os
import json
import logging
import tempfile
from typing import Optional, List, Dict
from fastapi import FastAPI, HTTPException, UploadFile, Form, Body, Header
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai 
from config import CORS_ORIGINS
from database import get_qdrant_client, ensure_collection_exists
from dependencies import get_gemini_client, get_user_id_from_request
from models import (
    ProcessPDFResponse, GeneratePaperRequest, GeneratePaperResponse,
    EvaluateAnswerRequest, EvaluateAnswerResponse, DeleteBookRequest,
    DeleteBookResponse, ChunkResponse, ChunkStatsResponse
)
from models import SemesterReportRequest
from services.pdf_processing import pil_to_part, safe_vertex_generate
from services.pdf_processing import process_pdf
from services.vector_store import upload_chunks_to_qdrant, get_chunks_by_filter, delete_chunks_by_pdf
from services.exam_generator import (
    generate_multiple_papers_with_summaries,
    process_paper_visuals,
    generate_realistic_image,
    generate_svg_visual,
    get_exam_generator_token_stats,
    reset_exam_generator_token_stats,
)
from services.grading import extract_contents_from_pdf, evaluate_answers, generate_semester_report
from dependencies import get_embeddings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Initialize Gemini client
try:
    get_gemini_client()
except Exception as e:
    logger.warning(f"Gemini client initialization warning: {e}")

# Initialize Qdrant
try:
    qdrant_client = get_qdrant_client()
    ensure_collection_exists(qdrant_client)
    logger.info("Qdrant initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Qdrant: {e}", exc_info=True)
    raise

# Create FastAPI app
app = FastAPI(title="EDUAI Backend API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/process_pdf/")
async def process_pdf_endpoint(
    file: UploadFile = Form(...),
    subject_data: str = Form(...),
    user_id: str = Header(..., alias="X-User-ID")
):
    """Process PDF and upload to Qdrant with real-time progress streaming (SSE)."""
    file_path = None
    try:
        # Validate file
        if not file or not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="File must be a PDF")
        
        # Validate user_id
        try:
            user_id = get_user_id_from_request(user_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        logger.info(f"Processing PDF: {file.filename} for user {user_id}")
        
        # Save file temporarily
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, f"temp_{user_id}_{file.filename}")
        
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Parse subject_data
        try:
            subject_data_dict = json.loads(subject_data)
            if not isinstance(subject_data_dict, dict):
                raise ValueError("subject_data must be a valid JSON object")
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON in subject_data: {str(e)}")
        
        # Generator function for streaming
        def generate():
            try:
                embeddings = get_embeddings()
                # Process PDF with generator - pass qdrant dependencies
                for progress_json in process_pdf(
                    subject_data_dict, 
                    file_path,
                    qdrant_client=qdrant_client,
                    embeddings=embeddings,
                    user_id=user_id
                ):
                    # Format as SSE: data: {json}\n\n
                    yield f"data: {progress_json}\n\n"
            except Exception as e:
                logger.error(f"Error processing PDF: {e}", exc_info=True)
                error_json = json.dumps({
                    "progress": 100,
                    "status": "Error",
                    "message": str(e)
                })
                yield f"data: {error_json}\n\n"
            finally:
                # Clean up temp file
                if file_path and os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        logger.warning(f"Error removing temp file {file_path}: {e}")
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in process_pdf_endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/generate_question_paper/", response_model=GeneratePaperResponse)
async def generate_question_paper(
    request_data: GeneratePaperRequest = Body(...),
    user_id: str = Header(..., alias="X-User-ID")
):
    """Generate question paper from processed PDF."""
    try:
        # Validate user_id
        try:
            user_id = get_user_id_from_request(user_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        # Convert Pydantic model to dict
        request_dict = request_data.dict(exclude_none=True)
        request_dict["user_id"] = user_id
        
        # Transform backend format to service format
        # 1. Handle className - derive from class if not provided
        class_value = request_dict.get("class") or request_dict.get("class_name")
        if class_value:
            if not request_dict.get("className"):
                request_dict["className"] = str(class_value)
            if not request_dict.get("class_name"):
                request_dict["class_name"] = str(class_value)
        
        # 2. Handle maxMarks - derive from totalMarks if not provided
        if not request_dict.get("maxMarks"):
            if request_dict.get("totalMarks"):
                request_dict["maxMarks"] = request_dict["totalMarks"]
            else:
                # Calculate from questions if available
                total = 0
                if "questions" in request_dict and isinstance(request_dict["questions"], list):
                    for q in request_dict["questions"]:
                        if isinstance(q, dict):
                            num_q = q.get("numQuestions", 0)
                            marks_per = q.get("marksPerQuestion", 0)
                            total += num_q * marks_per
                request_dict["maxMarks"] = total if total > 0 else 10  # Default to 10 if can't calculate
        
        # 3. Handle timeAllowed - derive from duration if not provided
        if not request_dict.get("timeAllowed"):
            if request_dict.get("duration") and isinstance(request_dict["duration"], dict):
                duration = request_dict["duration"]
                hours = duration.get("hours", 0)
                minutes = duration.get("minutes", 0)
                parts = []
                if hours > 0:
                    parts.append(f"{hours} hour{'s' if hours > 1 else ''}")
                if minutes > 0:
                    parts.append(f"{minutes} minute{'s' if minutes > 1 else ''}")
                request_dict["timeAllowed"] = " ".join(parts) if parts else "2 hours"
            else:
                request_dict["timeAllowed"] = "2 hours"  # Default fallback
        
        # 4. Handle instructions - provide defaults if not provided
        if not request_dict.get("instructions"):
            request_dict["instructions"] = [
                "All questions are compulsory.",
                "Use a blue or black ballpoint pen for clear visibility.",
                "Read each question carefully.",
                "The question paper is divided into Sections. Start each Section from a new page.",
                "Please write the Section number & name clearly before attempting the questions.",
                "Write the Question Number (e.g., Q1, Q2) clearly in the margin or at the start of each answer.",
                "If an answer continues to the next page, please write 'Q[No] (continued)' at the top of the new page (e.g., 'Q3 continued.').",
                "If you skip a question and attempt it later, ensure you write the correct Question Number, Section number & name again.",
            ]
        
        # 5. Handle numberOfPapers - check both spellings
        num_papers = request_dict.get("numberOfPapers") or request_dict.get("numberofPapers") or 1
        if not isinstance(num_papers, int) or num_papers <= 0:
            raise HTTPException(status_code=400, detail="numberOfPapers must be a positive integer")
        request_dict["numberOfPapers"] = num_papers
        
        # 6. Handle sectionName in questions - derive from type if not provided
        if "questions" in request_dict and isinstance(request_dict["questions"], list):
            for i, question in enumerate(request_dict["questions"]):
                if isinstance(question, dict) and not question.get("sectionName"):
                    # Derive section name from type or use default
                    q_type = question.get("type", "")
                    if q_type:
                        # Use type as section name, or create a default
                        question["sectionName"] = f"Section {i + 1}: {q_type}"
                    else:
                        question["sectionName"] = f"Section {i + 1}"
        
        logger.info(f"Generating {num_papers} question paper(s) for user {user_id}")
        reset_exam_generator_token_stats()

        # Generate papers
        try:
            exam_papers = generate_multiple_papers_with_summaries(
                client=qdrant_client,
                request_data=request_dict,
                user_id=user_id,
                num_papers=num_papers,
                k=10
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
        except Exception as e:
            logger.error(f"Error generating question papers: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error generating papers: {str(e)}")
        
        if not exam_papers or not isinstance(exam_papers, list):
            return GeneratePaperResponse(
                success=False,
                message="AI service did not return a valid array of papers."
            )
        
        if len(exam_papers) == 0:
            return GeneratePaperResponse(
                success=False,
                message="No papers were generated. Please check the request data and try again."
            )
        
        # 2. Generate Visuals (The Artist) - NEW STEP
        final_papers = []
        for paper in exam_papers:
            # This injects 'image_data' into questions that need it
            enhanced_paper = process_paper_visuals(paper)
            final_papers.append(enhanced_paper)

        token_usage = get_exam_generator_token_stats()
        logger.info(
            "Successfully generated %s question paper(s). API calls: %s, input tokens: %s, output tokens: %s",
            len(final_papers),
            token_usage.get("api_calls", 0),
            token_usage.get("input_tokens", 0),
            token_usage.get("output_tokens", 0),
        )
        return GeneratePaperResponse(
            success=True,
            question_paper=final_papers,
            token_usage={
                "input_tokens": token_usage.get("input_tokens", 0),
                "output_tokens": token_usage.get("output_tokens", 0),
                "total_tokens": token_usage.get("input_tokens", 0) + token_usage.get("output_tokens", 0),
                "api_calls": token_usage.get("api_calls", 0),
            },
        )
        
        logger.info(f"Successfully generated {len(exam_papers)} question paper(s) for user {user_id}")
        return GeneratePaperResponse(
            success=True,
            question_paper=exam_papers
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in generate_question_paper: {e}", exc_info=True)
        return GeneratePaperResponse(
            success=False,
            message=f"Internal server error: {str(e)}"
        )


@app.post("/evaluate_answer_paper/", response_model=EvaluateAnswerResponse)
async def evaluate_answer_paper(
    file: UploadFile = Form(...),
    question_paper_str: str = Form(...),
    user_id: Optional[str] = Header(None, alias="X-User-ID")
):
    """Evaluate a student's handwritten answers."""
    file_path = None
    try:
        # Save file locally
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, f"temp_{file.filename}")
        
        with open(file_path, "wb") as f:
            f.write(await file.read())

        try:
            question_paper = json.loads(question_paper_str)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON format in question_paper: {e}")

        logger.info("Extracting content from answer sheet...")
        extracted_answer_paper = extract_contents_from_pdf(file_path)
        if not extracted_answer_paper.strip():
            raise HTTPException(status_code=400, detail="Failed to extract text from the uploaded PDF file.")

        report = evaluate_answers(question_paper, extracted_answer_paper)
        logger.info(f"Evaluation successful for {file.filename}")

        # Clean up temp file
        if os.path.exists(file_path):
            os.remove(file_path)
        
        return EvaluateAnswerResponse(**report)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"CRITICAL ERROR in evaluate_answer_paper: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")
    finally:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass


@app.post("/regenerate_image/")
async def regenerate_image(
    question: str = Form(...),
    prompt: str = Form(...),
    type: str = Form(...),
):
    logger.info(f"Regenerating the image of type '{type}' for '{question}'")

    try:
        if type == "svg":
            result = generate_svg_visual(prompt)
        else:
            result = generate_realistic_image(prompt)
            
        if result and "content" in result:
            # Wrap the result to match the Express server's expectations
            return {
                "success": True,
                "imageUrl": result["content"],
                "type": result["type"]
            }
        
        raise HTTPException(status_code=500, detail="AI Service returned empty data")
        
    except Exception as e:
        logger.error(f"Error in regenerate_image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chunks/", response_model=ChunkResponse)
async def get_chunks(
    page: int = 1,
    page_size: int = 20,
    chapter_no: Optional[str] = None,
    subject: Optional[str] = None,
    className: Optional[str] = None,
    user_id: str = Header(..., alias="X-User-ID")
):
    """Retrieve chunks from Qdrant with pagination and filtering."""
    try:
        # Validate inputs
        if page < 1:
            raise HTTPException(status_code=400, detail="page must be >= 1")
        
        if page_size < 1 or page_size > 100:
            raise HTTPException(status_code=400, detail="page_size must be between 1 and 100")
        
        # Validate user_id
        try:
            user_id = get_user_id_from_request(user_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        # Build filters
        filters = {}
        if chapter_no:
            filters["chapter_no"] = chapter_no
        if subject:
            filters["subject"] = subject
        if className:
            filters["class"] = className
        
        # Get chunks from Qdrant
        embeddings = get_embeddings()
        all_docs = get_chunks_by_filter(
            client=qdrant_client,
            user_id=user_id,
            filters=filters,
            limit=10000
        )
        
        # Calculate pagination
        total_chunks = len(all_docs)
        total_pages = (total_chunks + page_size - 1) // page_size
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        
        # Get paginated chunks
        paginated_docs = all_docs[start_idx:end_idx]
        
        # Format response
        chunks_data = []
        for doc in paginated_docs:
            chunks_data.append({
                "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                "content_preview": doc.page_content[:200],
                "metadata": {
                    "chapter_no": doc.metadata.get("chapter_no", ""),
                    "chapter_name": doc.metadata.get("chapter_name", ""),
                    "page": doc.metadata.get("page", 0),
                    "content_type": doc.metadata.get("content_type", ""),
                    "subject": doc.metadata.get("subject", ""),
                    "class": doc.metadata.get("class", ""),
                    "pdf_name": doc.metadata.get("pdf_name", ""),
                    "chunk_index": doc.metadata.get("chunk_index", None)
                }
            })
        
        logger.info(f"Retrieved {len(paginated_docs)} chunks (page {page}/{total_pages}) for user {user_id}")
        
        return ChunkResponse(
            success=True,
            pagination={
                "page": page,
                "page_size": page_size,
                "total_chunks": total_chunks,
                "total_pages": total_pages,
                "has_next": page < total_pages,
                "has_prev": page > 1
            },
            filters={
                "chapter_no": chapter_no,
                "subject": subject,
                "className": className
            },
            chunks=chunks_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_chunks: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/chunks/stats/", response_model=ChunkStatsResponse)
async def get_chunks_stats(
    user_id: str = Header(..., alias="X-User-ID")
):
    """Get statistics about the chunks in Qdrant for a user."""
    try:
        # Validate user_id
        try:
            user_id = get_user_id_from_request(user_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        # Get all chunks for user
        all_docs = get_chunks_by_filter(
            client=qdrant_client,
            user_id=user_id,
            filters={},
            limit=10000
        )
        
        # Collect statistics
        total_chunks = len(all_docs)
        chapters = {}
        subjects = set()
        classes = set()
        content_types = {}
        
        for doc in all_docs:
            metadata = doc.metadata
            
            chapter_key = f"{metadata.get('chapter_no', 'Unknown')} - {metadata.get('chapter_name', 'Unknown')}"
            chapters[chapter_key] = chapters.get(chapter_key, 0) + 1
            
            if metadata.get("subject"):
                subjects.add(metadata.get("subject"))
            if metadata.get("class"):
                classes.add(metadata.get("class"))
            
            content_type = metadata.get("content_type", "unknown")
            content_types[content_type] = content_types.get(content_type, 0) + 1
        
        return ChunkStatsResponse(
            success=True,
            stats={
                "total_chunks": total_chunks,
                "total_chapters": len(chapters),
                "chapters": chapters,
                "subjects": list(subjects),
                "classes": list(classes),
                "content_types": content_types
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting chunks stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.delete("/delete_book/{pdf_name}", response_model=DeleteBookResponse)
async def delete_book_from_index(
    pdf_name: str,
    user_id: str = Header(..., alias="X-User-ID")
):
    try:
        user_id = get_user_id_from_request(user_id)

        delete_chunks_by_pdf(
            client=qdrant_client,
            pdf_name=pdf_name,
            user_id=user_id
        )

        logger.info(f"Deleted book '{pdf_name}' for user {user_id}")
        return DeleteBookResponse(
            success=True,
            message=f"Deleted all chunks for PDF: {pdf_name}"
        )

    except Exception as e:
        logger.error(f"Error deleting book from index: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_semester_report/")
async def generate_semester_report_endpoint(request: SemesterReportRequest):
    """
    Generates a subject-wise semester performance report
    with insights across Unit Test, Mid Term, and Final Exam.
    """
    try:
        logger.info(f"Generating semester report for {request.student_name}")

        if not request.evaluations:
            return JSONResponse(
                content={"error": "No evaluations provided"},
                status_code=400
            )

        result = generate_semester_report(request)

        return JSONResponse(
            content={"success": True, "report": result}
        )

    except Exception as e:
        logger.error(f"Semester report endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health/")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "EDUAI Backend"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

