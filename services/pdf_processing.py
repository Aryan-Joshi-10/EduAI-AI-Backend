"""
PDF parsing, OCR, and Chunking logic.
"""
import os
import json
import re
import time
import logging
import asyncio
import concurrent.futures
import io
import base64
from typing import List, Dict, Tuple, Generator, Optional, Any
from PIL import Image
from pdf2image import convert_from_path
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_core.documents import Document
from google.genai import types
from dependencies import get_gemini_client, get_openai_client
from config import OPENAI_MODEL
from utils import clean_json_string, parse_xml_to_json, parse_elements_from_xml
from services.vector_store import upload_chunks_to_qdrant

logger = logging.getLogger(__name__)

# Model name constant
MODEL_NAME = "gemini-3-flash-preview"

# Global token counter for PDF processing
_token_counter = {"input_tokens": 0, "output_tokens": 0}


def get_token_stats():
    """Get current token statistics"""
    return _token_counter.copy()


def reset_token_stats():
    """Reset token statistics"""
    _token_counter["input_tokens"] = 0
    _token_counter["output_tokens"] = 0


# =============================
# OPENAI FUNCTIONS 
# =============================

def pil_to_base64(image, max_size=(1600, 1600), quality=70):
    """Convert PIL Image to base64 string for OpenAI API with optimized compression and resizing"""
    try:
        if image is None:
            raise ValueError("Image cannot be None")
        
        if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
        image_bytes = buffer.getvalue()
        
        # Convert to base64
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        return base64_image
    except Exception as e:
        logger.error(f"Error converting PIL image to base64: {e}", exc_info=True)
        raise


def pil_to_part(image, max_size=(1600, 1600), quality=70):
    """
    Compatibility wrapper: Convert PIL Image to a format compatible with OpenAI.
    This maintains the same interface as the old Gemini pil_to_part function.
    Returns a dict with base64 image data.
    """
    base64_image = pil_to_base64(image, max_size, quality)
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}"
        }
    }


def safe_vertex_generate(parts, **kwargs):
    """
    OpenAI version: Optimized generation with retry logic and token counting.
    Maintains same interface as Gemini version for compatibility.
    Returns an object with .text attribute matching Gemini response format.
    """
    client = get_openai_client()
    retries = 5
    
    for attempt in range(retries):
        try:
            if not parts:
                raise ValueError("Parts cannot be empty")
            
            # Build messages for OpenAI API
            messages = []
            content_parts = []
            
            for part in parts:
                if isinstance(part, str):
                    content_parts.append({"type": "text", "text": part})
                elif isinstance(part, dict) and "type" in part:
                    # Already formatted for OpenAI (from pil_to_part)
                    content_parts.append(part)
                elif hasattr(part, 'inline_data'):
                    # Legacy Gemini Part object - convert to OpenAI format
                    image_data = part.inline_data.data
                    base64_image = base64.b64encode(image_data).decode('utf-8')
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{part.inline_data.mime_type};base64,{base64_image}"
                        }
                    })
                else:
                    # Fallback: try to convert to text
                    content_parts.append({"type": "text", "text": str(part)})
            
            messages.append({
                "role": "user",
                "content": content_parts
            })
            
            # Call OpenAI API
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                max_tokens=8192
            )
            
            # Update token counter
            if response.usage:
                _token_counter["input_tokens"] += response.usage.prompt_tokens
                _token_counter["output_tokens"] += response.usage.completion_tokens
            
            # Create a response object compatible with Gemini format
            class OpenAIResponse:
                def __init__(self, text_content):
                    self.text = text_content
                    self.candidates = []  # For compatibility
            
            return OpenAIResponse(response.choices[0].message.content)
            
        except Exception as e:
            error_str = str(e).lower()
            wait_time = min(2 * (2 ** attempt), 30)

            if "429" in error_str or "quota" in error_str or "rate limit" in error_str:
                logger.warning(f"Rate limit exceeded (Attempt {attempt+1}/{retries}). Waiting 30s...")
                time.sleep(30)
            elif "invalid" in error_str or "bad request" in error_str:
                logger.error(f"Invalid request (Attempt {attempt+1}/{retries}): {e}")
                if attempt == retries - 1:
                    raise
                time.sleep(wait_time)
            else:
                logger.warning(f"OpenAI error (Attempt {attempt+1}/{retries}): {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                
    logger.error(f"OpenAI generation failed after {retries} attempts.")
    return None


async def safe_vertex_generate_async(parts, **kwargs):
    """
    OpenAI async version: Optimized generation with retry logic and token counting.
    Maintains same interface as Gemini version for compatibility.
    Returns an object with .text attribute matching Gemini response format.
    """
    client = get_openai_client()
    retries = 5
    
    for attempt in range(retries):
        try:
            if not parts:
                raise ValueError("Parts cannot be empty")
            
            # Build messages for OpenAI API
            messages = []
            content_parts = []
            
            for part in parts:
                if isinstance(part, str):
                    content_parts.append({"type": "text", "text": part})
                elif isinstance(part, dict) and "type" in part:
                    # Already formatted for OpenAI (from pil_to_part)
                    content_parts.append(part)
                elif hasattr(part, 'inline_data'):
                    # Legacy Gemini Part object - convert to OpenAI format
                    image_data = part.inline_data.data
                    base64_image = base64.b64encode(image_data).decode('utf-8')
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{part.inline_data.mime_type};base64,{base64_image}"
                        }
                    })
                else:
                    # Fallback: try to convert to text
                    content_parts.append({"type": "text", "text": str(part)})
            
            messages.append({
                "role": "user",
                "content": content_parts
            })
            
            # Run blocking OpenAI call in executor to make it async
            loop = asyncio.get_event_loop()
            
            def make_openai_call():
                return client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=messages,
                    max_tokens=8192
                )
            
            response = await loop.run_in_executor(None, make_openai_call)
            
            # Update token counter
            if response.usage:
                _token_counter["input_tokens"] += response.usage.prompt_tokens
                _token_counter["output_tokens"] += response.usage.completion_tokens
            
            # Create a response object compatible with Gemini format
            class OpenAIResponse:
                def __init__(self, text_content):
                    self.text = text_content
                    self.candidates = []  # For compatibility
            
            return OpenAIResponse(response.choices[0].message.content)
            
        except Exception as e:
            error_str = str(e).lower()
            wait_time = min(2 * (2 ** attempt), 30)

            if "429" in error_str or "quota" in error_str or "rate limit" in error_str:
                logger.warning(f"Rate limit exceeded (Attempt {attempt+1}/{retries}). Waiting 30s...")
                await asyncio.sleep(30)
            elif "invalid" in error_str or "bad request" in error_str:
                logger.error(f"Invalid request (Attempt {attempt+1}/{retries}): {e}")
                if attempt == retries - 1:
                    raise
                await asyncio.sleep(wait_time)
            else:
                logger.warning(f"OpenAI error (Attempt {attempt+1}/{retries}): {e}. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
                
    logger.error(f"OpenAI generation failed after {retries} attempts.")
    return None


def clean_page_content(text: str) -> str:
    """
    A. Regex Cleaning (Before Extraction)
    Remove noise: ISBNs, copyright dates, repetitive headers/footers.
    Normalize whitespace.
    """
    if not text:
        return ""
    
    # Remove ISBN patterns (e.g., "ISBN: 978-0-123456-78-9" or "ISBN 9780123456789")
    text = re.sub(r'ISBN[:\s]*[\d\-Xx]+', '', text, flags=re.IGNORECASE)
    
    # Remove copyright and rationalised dates (e.g., "Rationalised 2023-24", "© 2023")
    text = re.sub(r'Rationalised\s+\d{4}[-–]\d{2,4}', '', text, flags=re.IGNORECASE)
    text = re.sub(r'©\s*\d{4}', '', text)
    text = re.sub(r'Copyright\s+\d{4}', '', text, flags=re.IGNORECASE)
    
    # Remove common header/footer patterns (page numbers, "Chapter X" repeated)
    # Remove standalone page numbers at start/end of lines
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    
    # Remove repetitive chapter titles that appear multiple times
    # This is a simple heuristic - can be enhanced
    lines = text.split('\n')
    seen_headers = set()
    cleaned_lines = []
    for line in lines:
        line_stripped = line.strip()
        # Skip if it's a very short line that looks like a repeated header
        if len(line_stripped) < 50 and line_stripped.lower() in seen_headers:
            continue
        if len(line_stripped) > 5 and len(line_stripped) < 50:
            seen_headers.add(line_stripped.lower())
        cleaned_lines.append(line)
    
    text = '\n'.join(cleaned_lines)
    
    # Normalize whitespace - multiple spaces to single space, but preserve newlines
    text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple newlines to double
    
    # Remove leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    
    return text.strip()


def detect_chapter_header(text: str, current_chapter: Dict) -> Dict:
    """
    B. Dynamic Chapter Detection (During Extraction)
    Check the actual text for headers like "Chapter X" or "Unit Y".
    Returns updated chapter info if a header is found.
    """
    if not text:
        return current_chapter
    
    # Patterns to detect chapter headers
    patterns = [
        r'Chapter\s+(\d+)[:\s]+(.+?)(?:\n|$)',
        r'Unit\s+(\d+)[:\s]+(.+?)(?:\n|$)',
        r'Chapter\s+(\d+)',
        r'Unit\s+(\d+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            chapter_no = match.group(1)
            chapter_title = match.group(2).strip() if len(match.groups()) > 1 else f"Chapter {chapter_no}"
            
            # Only update if we found a new chapter
            if chapter_no != current_chapter.get("chapter_no"):
                return {
                    "chapter_no": chapter_no,
                    "chapter_title": chapter_title,
                    "start_page": current_chapter.get("start_page", 0)
                }
    
    return current_chapter


async def extract_multimodal_elements_from_pdf_async(file_path: str, page_chapter_map: Dict[int, str], max_concurrent: int = 10) -> List[Dict]:
    """
    Phase 2: Async Parallel Content Extraction (The Speed Fix)
    Uses asyncio + semaphore for async OpenAI calls.
    Processes 5 pages per OpenAI request (100 pages = 20 requests).
    Injects chapter context from page_chapter_map to improve semantic accuracy.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found: {file_path}")
    
    logger.info(f"Phase 2: Extracting content from: {file_path} (Async Parallel Processing, 5 pages per request)")
    logger.info(f"Configuration: max_concurrent={max_concurrent}, pages_per_request=5, model={OPENAI_MODEL}")
    
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(file_path)
        total_pages = len(reader.pages)
        logger.info(f"Total pages to process: {total_pages}")
        estimated_requests = (total_pages // 5) + (1 if total_pages % 5 else 0)
        logger.info(f"Estimated OpenAI API calls: {estimated_requests} (5 pages each, {max_concurrent} concurrent)")
    except Exception as e:
        logger.warning(f"PyPDF2 failed, using fallback for page count: {e}")
        total_pages = None

    # Prompt for 5 pages per request
    def get_multi_page_prompt(page_numbers: List[int], chapter_names: List[str]) -> str:
        pages_info = ", ".join([f"Page {p} ({ch})" for p, ch in zip(page_numbers, chapter_names)])
        page_list = ", ".join([str(p) for p in page_numbers])
        
        # Build example XML for all pages
        example_pages = "\n".join([
            f'  <page number="{p}">\n    <element type="text">Content from page {p}</element>\n  </page>' 
            for p in page_numbers
        ])
        
        return f"""
Context: These {len(page_numbers)} pages are from the following chapters: {pages_info}

Analyze these {len(page_numbers)} textbook pages (pages {page_list}). Extract content for study material generation.

RULES:
1. TEXT: Summarize paragraphs clearly. Keep definitions and key dates exact.
2. DIAGRAMS: Describe exactly what the diagram shows (e.g., "A diagram showing the human heart with labeled aorta and ventricles").
3. FORMULAS: Write detailed mathematical formulas in LaTeX format strictly (e.g. $$ a^2 + b^2 = c^2 $$).
4. TABLES: Represent as Markdown tables.
5. **CRITICAL:** Each page must be in a separate <page> tag with the number attribute. Extract content from each page separately.

OUTPUT XML FORMAT (REQUIRED):
<pages>
{example_pages}
</pages>

IMPORTANT:
- Use <page number="X"> for each page where X is the actual page number
- Include elements inside each <page> tag
- Element types: "text", "diagram_caption", "formula_latex"
- Return ONLY XML, no other text
- Make sure to include all {len(page_numbers)} pages in your response
"""

    async def process_page_batch(page_numbers: List[int], page_images: List, semaphore: asyncio.Semaphore) -> List[Dict]:
        """Process a batch of 5 pages with chapter context injection"""
        async with semaphore:
            if not page_images or any(img is None for img in page_images):
                logger.warning(f"Skipping batch {page_numbers[0]}-{page_numbers[-1]}: Invalid images")
                return []
            
            # Get chapter names for all pages in batch
            chapter_names = [page_chapter_map.get(p_num, "General") for p_num in page_numbers]
            logger.info(f"Processing batch: pages {page_numbers[0]}-{page_numbers[-1]} (chapters: {chapter_names})")
            
            try:
                # Build prompt for multi-page request
                prompt = get_multi_page_prompt(page_numbers, chapter_names)
                
                # Build parts: prompt + all images
                parts = [prompt]
                for img in page_images:
                    parts.append(pil_to_part(img, quality=70))
                
                logger.debug(f"Calling OpenAI API for pages {page_numbers[0]}-{page_numbers[-1]}...")
                # Async OpenAI call
                resp = await safe_vertex_generate_async(parts)
                
                if not resp or not resp.text:
                    logger.warning(f"Empty response for pages {page_numbers[0]}-{page_numbers[-1]}")
                    return []
                
                raw = resp.text
                logger.debug(f"Received response for pages {page_numbers[0]}-{page_numbers[-1]}: {len(raw)} chars")
                logger.debug(f"Response preview (first 500 chars): {raw[:500]}")
                
                # Parse XML response
                data = parse_xml_to_json(raw)
                logger.debug(f"Parsed data keys: {list(data.keys()) if data else 'None'}")
                
                if not data:
                    logger.warning(f"No data parsed from response for pages {page_numbers[0]}-{page_numbers[-1]}. Raw response (first 1000 chars): {raw[:1000]}")
                    return []
                
                elements = []
                if "pages" in data:
                    logger.info(f"Found {len(data['pages'])} pages in response for batch {page_numbers[0]}-{page_numbers[-1]}")
                    # Handle multi-page response
                    for idx, p_data in enumerate(data["pages"]):
                        p_num = p_data.get("page_number") or p_data.get("number")
                        if p_num is None and idx < len(page_numbers):
                            p_num = page_numbers[idx]
                        
                        if p_num is None:
                            logger.warning(f"Could not determine page number for page {idx} in batch {page_numbers[0]}-{page_numbers[-1]}")
                            continue
                        
                        # Parse chapter_name to create detected_chapter dict
                        chapter_name = chapter_names[idx] if idx < len(chapter_names) else "General"
                        detected_chapter = {"chapter_no": "1", "chapter_title": "General", "start_page": p_num}
                        if chapter_name != "General":
                            match = re.match(r'Chapter\s+(\d+):\s*(.+)', chapter_name, re.IGNORECASE)
                            if match:
                                detected_chapter = {
                                    "chapter_no": match.group(1),
                                    "chapter_title": match.group(2).strip(),
                                    "start_page": p_num
                                }
                            else:
                                detected_chapter = {
                                    "chapter_no": "1",
                                    "chapter_title": chapter_name,
                                    "start_page": p_num
                                }
                        
                        page_elements = p_data.get("elements", [])
                        logger.debug(f"Page {p_num}: Found {len(page_elements)} elements")
                        for el in page_elements:
                            el["page"] = p_num
                            el["page_number"] = p_num
                            el["chapter_name"] = chapter_name
                            el["detected_chapter"] = detected_chapter.copy()
                            elements.append(el)
                elif "elements" in data:
                    logger.info(f"Found single-page format with {len(data['elements'])} elements for batch {page_numbers[0]}-{page_numbers[-1]}")
                    # Fallback: single page format, assign to first page
                    p_num = page_numbers[0]
                    chapter_name = chapter_names[0]
                    detected_chapter = {"chapter_no": "1", "chapter_title": "General", "start_page": p_num}
                    if chapter_name != "General":
                        match = re.match(r'Chapter\s+(\d+):\s*(.+)', chapter_name, re.IGNORECASE)
                        if match:
                            detected_chapter = {
                                "chapter_no": match.group(1),
                                "chapter_title": match.group(2).strip(),
                                "start_page": p_num
                            }
                    
                    for el in data["elements"]:
                        el["page"] = p_num
                        el["page_number"] = p_num
                        el["chapter_name"] = chapter_name
                        el["detected_chapter"] = detected_chapter.copy()
                        elements.append(el)
                else:
                    logger.warning(f"Unexpected data structure for pages {page_numbers[0]}-{page_numbers[-1]}. Keys: {list(data.keys())}")
                    logger.debug(f"Full data structure: {data}")
                
                logger.info(f"Successfully extracted {len(elements)} elements from pages {page_numbers[0]}-{page_numbers[-1]}")
                return elements
                
            except Exception as e:
                logger.error(f"Error processing pages {page_numbers[0]}-{page_numbers[-1]}: {e}", exc_info=True)
                return []

    all_elements = []
    batch_size = 150  # PDF conversion batch size
    pages_per_request = 5  # Process 5 pages per OpenAI request
    
    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(max_concurrent)
    
    current_page = 1
    while True:
        if total_pages and current_page > total_pages:
            break
        
        end_page = min(current_page + batch_size - 1, total_pages) if total_pages else current_page + batch_size - 1
        logger.info(f"Converting PDF pages {current_page} to {end_page} (DPI: 150)...")
        
        try:
            # Convert PDF pages to images (blocking, but necessary)
            loop = asyncio.get_event_loop()
            pages = await loop.run_in_executor(
                None,
                lambda: convert_from_path(file_path, first_page=current_page, last_page=end_page, dpi=150)
            )
        except Exception as e:
            if not total_pages:
                break
            logger.error(f"Conversion error: {e}")
            break
            
        if not pages:
            break

        # Group pages into batches of 5
        page_batches = []
        for i in range(0, len(pages), pages_per_request):
            batch_pages = pages[i:i + pages_per_request]
            batch_numbers = list(range(current_page + i, current_page + i + len(batch_pages)))
            page_batches.append((batch_numbers, batch_pages))
        
        logger.info(f"Processing {len(page_batches)} batches (5 pages each) for pages {current_page} to {end_page}...")
        
        # Process all batches concurrently with semaphore (10 at a time)
        logger.info(f"Submitting {len(page_batches)} batches to process concurrently (max {max_concurrent} at a time)...")
        tasks = [process_page_batch(p_nums, p_imgs, semaphore) for p_nums, p_imgs in page_batches]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect results
        successful_batches = 0
        failed_batches = 0
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch {idx+1} processing error: {result}", exc_info=True)
                failed_batches += 1
            elif result:
                all_elements.extend(result)
                successful_batches += 1
                logger.debug(f"Batch {idx+1}: Added {len(result)} elements")
            else:
                logger.warning(f"Batch {idx+1}: No elements returned")
                failed_batches += 1
        
        logger.info(f"Batch processing complete: {successful_batches} successful, {failed_batches} failed, {len(all_elements)} total elements so far")

        current_page += len(pages)
        if total_pages and current_page > total_pages:
            break
        if not total_pages and len(pages) < batch_size:
            break

    # Sort by page number to maintain document flow
    all_elements.sort(key=lambda x: x.get('page_number', x.get('page', 0)))
    
    total_requests = (total_pages // pages_per_request) + (1 if total_pages % pages_per_request else 0) if total_pages else 0
    logger.info(f"Extraction complete. Found {len(all_elements)} elements from approximately {total_requests} requests (5 pages per request).")
    if len(all_elements) == 0:
        logger.error("WARNING: No elements extracted! Check logs above for parsing errors.")
    return all_elements


def extract_multimodal_elements_from_pdf(file_path: str, page_chapter_map: Dict[int, str], max_workers: int = 10) -> List[Dict]:
    """
    Synchronous wrapper for async extraction function.
    Maintains backward compatibility.
    """
    try:
        # Check if we're already in an async context
        loop = asyncio.get_running_loop()
        # If we're here, we're in an async context - need to handle differently
        # Use nest_asyncio to allow nested event loops
        try:
            import nest_asyncio
            nest_asyncio.apply()
            # Run in a new thread to avoid blocking
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    lambda: asyncio.run(extract_multimodal_elements_from_pdf_async(file_path, page_chapter_map, max_workers))
                )
                return future.result()
        except ImportError:
            logger.warning("nest_asyncio not available. Using thread executor.")
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    lambda: asyncio.run(extract_multimodal_elements_from_pdf_async(file_path, page_chapter_map, max_workers))
                )
                return future.result()
    except RuntimeError:
        # No event loop running, safe to use asyncio.run()
        return asyncio.run(extract_multimodal_elements_from_pdf_async(file_path, page_chapter_map, max_workers))


async def extract_chapters_async(file_path: str, num_pages: int = 15, max_concurrent: int = 10) -> Tuple[List[Dict], Dict[int, str]]:
    """
    Phase 1: TOC Extraction (Parallel & Fast)
    Extracts Chapters and creates a Page-to-Chapter Map using parallel processing.
    Handles TOCs without page numbers by defaulting to 0.
    Only extracts top-level chapters, not subheadings or subsections.
    
    Returns:
        Tuple of (chapters_list, page_chapter_map) where page_chapter_map maps page_number -> chapter_name
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found: {file_path}")
        
    logger.info(f"Phase 1: Extracting Table of Contents from first {num_pages} pages (Parallel Processing, max {max_concurrent} concurrent, model={OPENAI_MODEL})...")
    
    # Get total page count for map creation
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(file_path)
        total_pages = len(reader.pages)
        logger.info(f"Total PDF pages: {total_pages}")
    except Exception as e:
        logger.warning(f"Could not get total page count: {e}. Will estimate from TOC.")
        total_pages = None

    # UPDATED PROMPT: Explicitly handle missing page numbers and exclude subheadings
    prompt = """
    Analyze this page. Is it a Table of Contents (Index)?

    Avoid confusing or extracting the following pages:
    - Title or cover page
    - Preface, Foreword, Acknowledgement pages
    - Pages with only a single image or poem
    - Pages with no list-like structure or no clear numbering
    
    If YES, extract the chapters in this JSON format:
    [
      {"chapter_no": "1", "chapter_title": "Resources", "start_page": 1},
      {"chapter_no": "2", "chapter_title": "Land and Soil", "start_page": 0} 
    ]
    
    RULES:
    1. Extract the Title and the Page Number listed against it.
    2. Extract ONLY top-level chapter entries. **DO NOT extract subheadings, subsections, or nested items** (e.g., 1.1, 1.2, A, B, bullet points, indented items).
    3. **CRITICAL:** If NO page number is visible, set "start_page": 0. Do NOT guess.
    4. Return [] if this is not a TOC page.
    """

    try:
        # Convert PDF pages to images (blocking operation, run in executor)
        loop = asyncio.get_event_loop()
        pages = await loop.run_in_executor(
            None,
            lambda: convert_from_path(file_path, first_page=1, last_page=num_pages, dpi=90)
        )
    except Exception as e:
        logger.error(f"PDF Convert Error: {e}")
        return [], {}

    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_toc_async(page_image, page_number: int):
        """Process a single TOC page asynchronously with page number tracking"""
        async with semaphore:
            try:
                # Run blocking OpenAI call in executor
                loop = asyncio.get_event_loop()
                resp = await loop.run_in_executor(
                    None,
                    lambda: safe_vertex_generate([prompt, pil_to_part(page_image)])
                )
                
                if not resp or not resp.text:
                    logger.debug(f"Page {page_number}: Empty response from OpenAI")
                    return []
                
                text = resp.text
                json_match = re.search(r"\[[\s\S]*\]", text)
                if json_match:
                    raw_json = json_match.group(0)
                    try:
                        data = json.loads(clean_json_string(raw_json))
                    except:
                        data = json.loads(raw_json)
                    
                    if isinstance(data, list):
                        # Add source_page to each chapter for tracking
                        for item in data:
                            item["source_page"] = page_number
                        logger.info(f"Page {page_number}: Extracted {len(data)} chapters")
                        return data
                    else:
                        logger.debug(f"Page {page_number}: Response is not a list")
                        return []
                else:
                    logger.debug(f"Page {page_number}: No JSON array found in response")
                    return []
            except Exception as e:
                logger.error(f"Page {page_number}: Error processing TOC: {e}", exc_info=True)
                return []
            return []

    # Process all pages in parallel
    logger.info(f"Processing {len(pages)} pages in parallel (max {max_concurrent} concurrent)...")
    tasks = [process_toc_async(page, page_num + 1) for page_num, page in enumerate(pages)]
    results_list = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Collect results and handle exceptions
    results = []
    for idx, result in enumerate(results_list):
        if isinstance(result, Exception):
            logger.error(f"Page {idx + 1}: Processing error: {result}", exc_info=True)
        elif result:
            results.extend(result)
        else:
            logger.debug(f"Page {idx + 1}: No chapters found")

    # Deduplicate - prioritize chapters from earlier pages if duplicates exist
    # Sort by source_page first, then by chapter_no to maintain order
    def sort_key(x):
        source_page = x.get("source_page", 9999)
        chapter_no_str = str(x.get("chapter_no", "0"))
        chapter_no_match = re.search(r'\d+', chapter_no_str)
        chapter_no = int(chapter_no_match.group()) if chapter_no_match else 0
        return (source_page, chapter_no)
    
    results.sort(key=sort_key)
    
    seen = set()
    final_chapters = []
    for item in results:
        key = str(item.get("chapter_no")) + item.get("chapter_title", "").lower()
        if key not in seen:
            seen.add(key)
            # Remove source_page from final output (it was just for tracking)
            item.pop("source_page", None)
            final_chapters.append(item)
    
    # Final sort by chapter number, then by start_page for proper ordering
    def sorter(x):
        try: 
            chapter_no = int(re.search(r'\d+', str(x.get("chapter_no", "0"))).group())
        except: 
            chapter_no = 0
        start_page = int(x.get("start_page", 0))
        return (chapter_no, start_page)
    
    final_chapters.sort(key=sorter)
    
    # Create Page-to-Chapter Map
    page_chapter_map = {}
    
    if final_chapters:
        # Fill in missing start_page values by searching in elements (will be done later in create_smart_chunks)
        # For now, use sequential assignment for pages with start_page=0
        last_known_page = 1
        for i, chap in enumerate(final_chapters):
            if int(chap.get("start_page", 0)) == 0:
                # Assign sequential page if no page number found
                chap["start_page"] = last_known_page + 1
            last_known_page = int(chap.get("start_page", 1))
        
        # Build the map
        for i, chap in enumerate(final_chapters):
            start = int(chap.get("start_page", 1))
            if i < len(final_chapters) - 1:
                end = int(final_chapters[i+1].get("start_page", 10000)) - 1
            else:
                # Last chapter extends to end of document
                end = total_pages if total_pages else 10000
            
            chapter_name = f"Chapter {chap.get('chapter_no', '?')}: {chap.get('chapter_title', 'Unknown')}"
            for p in range(max(1, start), end + 1):
                page_chapter_map[p] = chapter_name
        
        # Fill any remaining pages with "General" if TOC ended early
        if total_pages:
            for p in range(1, total_pages + 1):
                if p not in page_chapter_map:
                    page_chapter_map[p] = "General"
    else:
        # No TOC found - use "General" for all pages
        logger.warning("No TOC found. Using 'General' chapter for all pages.")
        if total_pages:
            for p in range(1, total_pages + 1):
                page_chapter_map[p] = "General"
        else:
            # Fallback: at least map first 1000 pages
            for p in range(1, 1001):
                page_chapter_map[p] = "General"
    
    logger.info(f"Extracted {len(final_chapters)} chapters. Created page_chapter_map with {len(page_chapter_map)} entries.")
    return final_chapters, page_chapter_map


def extract_chapters(file_path: str, num_pages: int = 15, max_workers: int = 10) -> Tuple[List[Dict], Dict[int, str]]:
    """
    Synchronous wrapper for async chapter extraction function.
    Maintains backward compatibility while enabling parallel processing.
    """
    try:
        # Check if we're already in an async context
        loop = asyncio.get_running_loop()
        # If we're here, we're in an async context - need to handle differently
        try:
            import nest_asyncio
            nest_asyncio.apply()
            # Run in a new thread to avoid blocking
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    lambda: asyncio.run(extract_chapters_async(file_path, num_pages, max_workers))
                )
                return future.result()
        except ImportError:
            logger.warning("nest_asyncio not available. Using thread executor.")
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    lambda: asyncio.run(extract_chapters_async(file_path, num_pages, max_workers))
                )
                return future.result()
    except RuntimeError:
        # No event loop running, safe to use asyncio.run()
        return asyncio.run(extract_chapters_async(file_path, num_pages, max_workers))


def create_smart_chunks(subject_data: dict, elements: List[Dict], chapters: List[Dict]) -> List[Document]:
    """
    Refined Chunking with 4-Step Pipeline:
    A. Regex Cleaning
    B. Dynamic Chapter Detection (already done during extraction)
    C. Context Injection
    D. Semantic Splitting
    """
    if not elements: return []
    if not chapters: chapters = [{"chapter_no": "1", "chapter_title": "Start", "start_page": 1}]

    logger.info("Creating chunks with improved quality pipeline...")
    
    subject = subject_data.get("subject", "Unknown")
    class_name = subject_data.get("class", "Unknown")

    for chap in chapters:
        if int(chap.get("start_page", 0)) == 0:
            target_title = re.sub(r'[^a-z0-9]', '', chap.get("chapter_title", "").lower())[:20]
            if len(target_title) < 4: continue
            
            for el in elements:
                content = re.sub(r'[^a-z0-9]', '', el.get("content", "").lower())[:100]
                if target_title in content:
                    chap["start_page"] = el.get("page", 1)
                    logger.info(f"Found Title '{chap['chapter_title']}' on Page {chap['start_page']}")
                    break

    chapters = sorted(chapters, key=lambda x: int(x.get("start_page", 0)))
    
    last_known_page = 1
    for chap in chapters:
        if int(chap.get("start_page", 0)) == 0: 
            chap["start_page"] = last_known_page + 1
        last_known_page = int(chap["start_page"])

    page_to_chap_map = {}
    for i, chap in enumerate(chapters):
        start = int(chap["start_page"])
        if i < len(chapters) - 1:
            end = int(chapters[i+1]["start_page"]) - 1
        else:
            end = 10000 
        
        logger.info(f"Mapping Chapter {chap['chapter_no']} ({chap['chapter_title']}) to Pages {start}-{end}")
        
        for p in range(start, end + 1):
            page_to_chap_map[p] = chap

    page_buffers = {} 

    for el in elements:
        pg = el.get("page", 1)
        content = el.get("content", "")
        
        # A. Regex Cleaning
        content = clean_page_content(content)
        if not content.strip():
            continue
        
        # B. Use detected chapter from extraction if available, otherwise use page mapping
        detected_chap = el.get("detected_chapter")
        if detected_chap:
            assigned_chap = detected_chap
        else:
            # Fallback to page-based mapping
            fig_match = re.search(r'(?:Figure|Fig|Table|Ex)\.?\s*(\d+)', content, re.IGNORECASE)
            forced_chap = None
            if fig_match:
                detected_num = fig_match.group(1)
                forced_chap = next((c for c in chapters if str(c.get("chapter_no")) == detected_num), None)
            
            if forced_chap:
                assigned_chap = forced_chap
            else:
                assigned_chap = page_to_chap_map.get(pg, chapters[0] if chapters else {})
        
        chapter_no = assigned_chap.get("chapter_no", "Unknown")
        chapter_title = assigned_chap.get("chapter_title", "Unknown")
        
        el_type = el.get("type", "text")
        if el_type == "formula_latex": 
            prefix = f"\n[FORMULA]: $$ {content} $$\n"
        elif el_type == "diagram_caption": 
            prefix = f"\n[DIAGRAM]: {content}\n"
        else: 
            prefix = f"{content}\n"
        
        # Detect if this looks like a header (for markdown conversion)
        is_header = False
        if el_type == "text" and len(content) < 100:
            # Check if it looks like a chapter/unit/section header
            header_patterns = [
                r'^(Chapter|Unit|Section)\s+\d+',
                r'^\d+\.\s+[A-Z]',  # Numbered section like "1. Introduction"
            ]
            for pattern in header_patterns:
                if re.match(pattern, content.strip(), re.IGNORECASE):
                    is_header = True
                    prefix = f"## {content.strip()}\n\n"  # Convert to markdown header
                    break
        
        if pg not in page_buffers:
            page_buffers[pg] = {
                "text": "", 
                "chapter_no": chapter_no,
                "chapter_title": chapter_title
            }
        page_buffers[pg]["text"] += prefix

    chunks = []
    
    # D. Semantic Splitting: Use MarkdownHeaderTextSplitter + RecursiveCharacterTextSplitter
    # First, try to split by markdown headers if present
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[
        ("##", "Section"),
        ("###", "Subsection"),
    ])
    
    # Then use RecursiveCharacterTextSplitter with tuned parameters
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,  # Optimized for 500-800 range (using 700 for balance)
        chunk_overlap=125,  # Optimized for 100-150 range (using 125)
        separators=["\n\n", "\n", ". ", " ", ""]  # Better separation
    )
    
    sorted_pages = sorted(page_buffers.keys())

    for pg in sorted_pages:
        data = page_buffers[pg]
        raw_text = data["text"]
        
        # C. Context Injection - prepend context to text before splitting
        chapter_no = str(data["chapter_no"])
        chapter_title = data["chapter_title"]
        context_prefix = f"Subject: {subject}\nClass: {class_name}\nChapter: {chapter_title} (Chapter {chapter_no})\n\n"
        final_text = context_prefix + raw_text
        
        # D. Semantic Splitting
        try:
            # First try markdown splitting if headers are present
            if "##" in final_text:
                md_docs = markdown_splitter.split_text(final_text)
                # Further split large markdown sections
                for md_doc in md_docs:
                    if len(md_doc.page_content) > 700:
                        splits = text_splitter.split_text(md_doc.page_content)
                        for split in splits:
                            chunks.append(Document(
                                page_content=split,
                                metadata={
                                    **subject_data,
                                    "chapter_no": chapter_no,
                                    "chapter_name": chapter_title,
                                    "page": int(pg),
                                    "content_type": "textbook_content",
                                    "pdf_name": subject_data.get("pdf_name", "doc"),
                                    "chunk_index": len(chunks)
                                }
                            ))
                    else:
                        chunks.append(Document(
                            page_content=md_doc.page_content,
                            metadata={
                                **subject_data,
                                "chapter_no": chapter_no,
                                "chapter_name": chapter_title,
                                "page": int(pg),
                                "content_type": "textbook_content",
                                "pdf_name": subject_data.get("pdf_name", "doc"),
                                "chunk_index": len(chunks)
                            }
                        ))
            else:
                # No markdown headers, use recursive splitter directly
                page_splits = text_splitter.split_text(final_text)
                for i, split_text in enumerate(page_splits):
                    chunks.append(Document(
                        page_content=split_text,
                        metadata={
                            **subject_data,
                            "chapter_no": chapter_no,
                            "chapter_name": chapter_title,
                            "page": int(pg),
                            "content_type": "textbook_content",
                            "pdf_name": subject_data.get("pdf_name", "doc"),
                            "chunk_index": i
                        }
                    ))
        except Exception as e:
            # Fallback to simple splitting if markdown splitting fails
            logger.warning(f"Markdown splitting failed for page {pg}, using fallback: {e}")
            page_splits = text_splitter.split_text(final_text)
            for i, split_text in enumerate(page_splits):
                chunks.append(Document(
                    page_content=split_text,
                    metadata={
                        **subject_data,
                        "chapter_no": chapter_no,
                        "chapter_name": chapter_title,
                        "page": int(pg),
                        "content_type": "textbook_content",
                        "pdf_name": subject_data.get("pdf_name", "doc"),
                        "chunk_index": i
                    }
                ))

    logger.info(f"Created {len(chunks)} chunks from {len(sorted_pages)} pages.")
    return chunks

def process_pdf(
    subject_data: dict, 
    file_path: str, 
    max_toc_pages: int = 15, 
    chapters: List[Dict] = None,
    qdrant_client: Optional[Any] = None,
    embeddings: Optional[Any] = None,
    user_id: Optional[str] = None
) -> Generator[str, None, None]:
    """
    Process PDF - extracts chapters and all content from the book.
    Generator that yields JSON progress updates at key milestones.
    
    Yields:
        JSON strings with progress updates:
        - {"progress": 33, "status": "Chapters extracted"}
        - {"progress": 66, "status": "Elements extracted"}
        - {"progress": 85, "status": "Vector storage done"} (if qdrant_client provided)
        - {"progress": 100, "status": "Success", "data": result}
    
    Note:
        This is a generator function that yields JSON strings. Consume all yields to get the final result.
    """
    if not subject_data or not isinstance(subject_data, dict):
        raise ValueError("subject_data must be a non-empty dictionary")
    
    if not file_path or not isinstance(file_path, str):
        raise ValueError("file_path must be a non-empty string")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")
    
    start_time = time.time()
    logger.info(f"Subject Data: {subject_data}")
    
    # Reset token counter at the start of processing
    reset_token_stats()
    logger.info("Token counter reset. Starting PDF processing...")
    
    page_chapter_map = {}

    try:
        # ==================================================================================
        # PHASE 1: TOC Extraction (Determine Chapters and Page Map)
        # ==================================================================================
        if chapters is None:
            logger.info("Phase 1: Extracting chapters and creating page_chapter_map...")
            chapters, page_chapter_map = extract_chapters(file_path, num_pages=max_toc_pages, max_workers=10)
            logger.info(f"Phase 1 Complete: Extracted {len(chapters)} chapters.")
        else:
            logger.info(f"Phase 1: Using provided {len(chapters)} chapters (skipping redundant extraction)")
            # Reconstruct page_chapter_map from provided chapters
            try:
                from PyPDF2 import PdfReader
                reader = PdfReader(file_path)
                total_pages = len(reader.pages)
            except Exception:
                total_pages = None
            
            # Fill in missing start_page values if 0
            last_known_page = 1
            for i, chap in enumerate(chapters):
                if int(chap.get("start_page", 0)) == 0:
                    chap["start_page"] = last_known_page + 1
                last_known_page = int(chap.get("start_page", 1))
            
            # Build the map
            for i, chap in enumerate(chapters):
                start = int(chap.get("start_page", 1))
                if i < len(chapters) - 1:
                    end = int(chapters[i+1].get("start_page", 10000)) - 1
                else:
                    end = total_pages if total_pages else 10000
                
                chapter_name = f"Chapter {chap.get('chapter_no', '?')}: {chap.get('chapter_title', 'Unknown')}"
                for p in range(max(1, start), end + 1):
                    page_chapter_map[p] = chapter_name
            
            # Fill remaining pages with "General"
            if total_pages:
                for p in range(1, total_pages + 1):
                    if p not in page_chapter_map:
                        page_chapter_map[p] = "General"
        
        # Yield progress: 33% - Chapters extracted
        yield json.dumps({"progress": 33, "status": "Chapters extracted"})

        # ==================================================================================
        # PHASE 2: Parallel Content Extraction
        # (This must be unindented so it runs AFTER the if/else block above)
        # ==================================================================================
        logger.info("Phase 2: Extracting content from ALL pages with parallel processing and chapter context...")
        logger.info(f"Configuration: 5 pages per OpenAI request, 10 concurrent requests maximum, model={OPENAI_MODEL}")
        
        elements = extract_multimodal_elements_from_pdf(file_path, page_chapter_map, max_workers=10)
        
        if not elements:
            logger.error("CRITICAL: Phase 2 returned no elements. Chunking will be skipped.")
            # Yield final result with empty chunks
            result = {
                "progress": 100,
                "status": "Success",
                "data": {
                    "chapters": chapters,
                    "chunks": 0
                }
            }
            yield json.dumps(result)
            return

        logger.info(f"Phase 2 Complete: Extracted {len(elements)} elements")
        
        # Yield progress: 66% - Elements extracted
        yield json.dumps({"progress": 66, "status": "Elements extracted"})

        # ==================================================================================
        # PHASE 3: Smart Chunking
        # ==================================================================================
        logger.info("Phase 3: Creating chunks with improved quality pipeline...")
        chunks = create_smart_chunks(subject_data, elements, chapters)
        
        elapsed_time = time.time() - start_time
        
        # Get token statistics
        token_stats = get_token_stats()
        total_tokens = token_stats["input_tokens"] + token_stats["output_tokens"]
        
        logger.info(f"Total processing time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        logger.info(f"Phase 3 Complete: Created {len(chunks)} chunks from {len(chapters)} chapters")
        logger.info(f"Token Usage - Input: {token_stats['input_tokens']:,}, Output: {token_stats['output_tokens']:,}, Total: {total_tokens:,}")
        
        # Upload to Qdrant if dependencies provided
        if qdrant_client and embeddings and user_id:
            try:
                upload_chunks_to_qdrant(
                    client=qdrant_client,
                    chunks=chunks,
                    embeddings=embeddings,
                    user_id=user_id
                )
                logger.info(f"Successfully uploaded {len(chunks)} chunks to Qdrant for user {user_id}")
                
                # Yield progress: 85% - Vector storage done
                yield json.dumps({"progress": 85, "status": "Vector storage done"})
            except Exception as e:
                logger.error(f"Error uploading chunks to Qdrant: {e}", exc_info=True)
                # Continue even if upload fails
        
        # Get final token statistics
        token_stats = get_token_stats()
        total_tokens = token_stats["input_tokens"] + token_stats["output_tokens"]
        
        # Yield final result: 100% - Success
        result = {
            "progress": 100,
            "status": "Success",
            "data": {
                "chapters": chapters,
                "chunks": len(chunks),
                "token_usage": {
                    "input_tokens": token_stats["input_tokens"],
                    "output_tokens": token_stats["output_tokens"],
                    "total_tokens": total_tokens
                }
            }
        }
        yield json.dumps(result)

    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"Error processing PDF after {elapsed_time:.2f} seconds: {e}", exc_info=True)
        # Yield error status
        error_result = {
            "progress": 100,
            "status": "Error",
            "message": str(e)
        }
        yield json.dumps(error_result)
        raise
