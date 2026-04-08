"""
Pydantic schemas (Request/Response models).
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any


class SubjectData(BaseModel):
    """Subject metadata for PDF processing."""
    subject: str
    class_name: str = Field(..., alias="class")
    pdf_name: str
    
    class Config:
        populate_by_name = True


class ProcessPDFResponse(BaseModel):
    """Response for PDF processing endpoint."""
    status: str
    message: str
    chunks: Optional[int] = None
    chapters: Optional[List[Dict]] = None


class QuestionRequest(BaseModel):
    """Question generation request."""
    type: str
    sectionName: Optional[str] = None  # Made optional to match backend
    topics: List[str]
    numQuestions: int
    marks: Optional[int] = None
    marksPerQuestion: Optional[int] = None
    llm_note: Optional[List[str]] = None
    difficulty: Optional[str] = None  # Added to match backend


class GeneratePaperRequest(BaseModel):
    """Request for question paper generation - flexible to accept backend format."""
    subject: str
    class_name: Optional[str] = Field(None, alias="class")  # Made optional, can derive from class
    className: Optional[str] = None  # Made optional, can derive from class
    maxMarks: Optional[int] = None  # Made optional, can derive from totalMarks
    totalMarks: Optional[int] = None  # Added to match backend
    timeAllowed: Optional[str] = None  # Made optional, can derive from duration
    duration: Optional[Dict[str, Any]] = None  # Added to match backend (has hours, minutes)
    instructions: Optional[List[str]] = None  # Made optional, can have defaults
    questions: List[QuestionRequest]  # sectionName is optional, so this should work
    numberOfPapers: Optional[int] = None  # Made optional
    numberofPapers: Optional[int] = None  # Added to match backend spelling
    pdf_name: str
    user_id: Optional[str] = None  # Will be set from header/form
    bookId: Optional[str] = None  # Added to match backend
    examType: Optional[str] = None  # Added to match backend
    question_type: Optional[List[str]] = None  # Added to match backend
    topics: Optional[List[str]] = None  # Added to match backend
    
    class Config:
        populate_by_name = True
        extra = "allow"  # Allow extra fields from backend


class GeneratePaperResponse(BaseModel):
    """Response for question paper generation."""
    success: bool
    question_paper: Optional[List[Dict]] = None
    message: Optional[str] = None
    token_usage: Optional[Dict[str, int]] = None  # input_tokens, output_tokens, total_tokens, api_calls


class EvaluateAnswerRequest(BaseModel):
    """Request for answer evaluation."""
    question_paper: Dict[str, Any]
    user_id: Optional[str] = None  # Will be set from header/form


class EvaluateAnswerResponse(BaseModel):
    """Response for answer evaluation."""
    Subject: str
    Class: str
    totalMarks: int
    obtainedMarks: int
    sections: List[Dict[str, Any]]
    chapter_summary: Optional[Dict[str, Any]] = None

class SubjectMark(BaseModel):
    subject: str
    marks_obtained: int
    total_marks: int
    date: str

class SemesterReportRequest(BaseModel):
    student_name: str
    class_grade: str
    semester: str
    academic_year: str
    evaluations: Dict[str, List[SubjectMark]]
    # Example keys: "Unit Test", "Mid Term", "Final Exam"

class DeleteBookRequest(BaseModel):
    """Request to delete a book from index."""
    pdf_name: str
    user_id: Optional[str] = None  # Will be set from header/form


class DeleteBookResponse(BaseModel):
    """Response for book deletion."""
    success: bool
    message: str


class ChunkFilter(BaseModel):
    """Filters for chunk retrieval."""
    chapter_no: Optional[str] = None
    subject: Optional[str] = None
    className: Optional[str] = None
    page: Optional[int] = None


class ChunkResponse(BaseModel):
    """Response for chunk retrieval."""
    success: bool
    pagination: Dict[str, Any]
    filters: Dict[str, Optional[str]]
    chunks: List[Dict[str, Any]]


class ChunkStatsResponse(BaseModel):
    """Response for chunk statistics."""
    success: bool
    stats: Dict[str, Any]
