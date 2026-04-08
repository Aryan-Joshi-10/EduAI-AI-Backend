"""
Answer evaluation logic using OpenAI API.
"""
import os
import json
import re
import time
import logging
import concurrent.futures
from typing import Dict, List, Optional, Any
from pdf2image import convert_from_path
from dependencies import get_openai_client
from config import OPENAI_MODEL
from services.pdf_processing import pil_to_part
from utils import clean_json_string, clean_latex_content
from models import SemesterReportRequest
from fastapi.responses import JSONResponse
from fastapi import HTTPException

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GEMINI (COMMENTED OUT - REPLACED WITH OPENAI)
# ---------------------------------------------------------------------------
# from google.genai import types
# from dependencies import get_gemini_client
# import google.generativeai as genai
# MODEL_NAME = "gemini-3-flash-preview"

# Grading token and API call counter
_grading_stats = {"input_tokens": 0, "output_tokens": 0, "api_calls": 0}


def get_grading_token_stats():
    """Return current grading token usage and API call count."""
    return _grading_stats.copy()


def reset_grading_token_stats():
    """Reset grading token and call counters."""
    _grading_stats["input_tokens"] = 0
    _grading_stats["output_tokens"] = 0
    _grading_stats["api_calls"] = 0


def _grading_openai_generate(parts: List[Any], max_retries: int = 5) -> Optional[Any]:
    """
    Call OpenAI API for grading. Updates _grading_stats (input_tokens, output_tokens, api_calls).
    parts: list of str (prompt text) or dict (image part from pil_to_part).
    Returns object with .text attribute for compatibility with existing code.
    """
    client = get_openai_client()
    content_parts = []
    for part in parts:
        if isinstance(part, str):
            content_parts.append({"type": "text", "text": part})
        elif isinstance(part, dict) and part.get("type") == "image_url":
            content_parts.append(part)
        else:
            content_parts.append({"type": "text", "text": str(part)})

    messages = [{"role": "user", "content": content_parts}]

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                max_tokens=8192,
            )
            _grading_stats["api_calls"] += 1
            if response.usage:
                _grading_stats["input_tokens"] += response.usage.prompt_tokens
                _grading_stats["output_tokens"] += response.usage.completion_tokens

            class _Response:
                def __init__(self, text_content):
                    self.text = text_content
                    self.candidates = []

            return _Response(response.choices[0].message.content)
        except Exception as e:
            err = str(e).lower()
            wait = min(2 * (2 ** attempt), 30)
            if "429" in err or "rate limit" in err:
                logger.warning(f"Grading OpenAI rate limit (attempt {attempt + 1}/{max_retries}), waiting 30s...")
                time.sleep(30)
            else:
                logger.warning(f"Grading OpenAI error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
    logger.error("Grading OpenAI generation failed after %s attempts.", max_retries)
    return None


def extract_contents_from_pdf(file_path: str, max_workers: int = 5) -> str:
    """
    Extract all handwritten text from a student's handwritten answer sheet.
    Uses parallel processing to speed up extraction while maintaining quality.
    
    Args:
        file_path: Path to the PDF file
        max_workers: Maximum number of concurrent page processing workers (default: 5)
    
    Returns:
        Extracted text from all pages, sorted by page number
    """
    try:
        pages = convert_from_path(pdf_path=file_path) 
    except Exception as e:
        logger.error(f"Error converting PDF to images: {e}")
        raise RuntimeError(f"Could not convert PDF. Is Poppler installed? Error: {e}")

    prompt = """
You are an OCR and handwriting recognition expert. 
You are given a scanned page from a student's handwritten answer sheet. 
Your goal is to extract **only** the handwritten content written by the student with maximum accuracy. 

### Extraction Rules:
1. Extract everything written by hand — including Name, Roll Number, Class, Subject, Question Numbers, and all Answers.
2. Preserve the **exact words, spellings, mathematical notations, symbols, and diagrams descriptions** as visible.
3. Do **not** skip crossed-out text; mention it as: (crossed out: "<text>")
4. Do **not** interpret or summarize — just transcribe exactly what is written.
5. Maintain natural line breaks and structure (use `\n` where new lines are visible).
6. Ignore printed templates, page numbers, logos, headers, or margins.
7. If handwriting is unclear or ambiguous, mark it as `[unclear]`.

### Output format:
Return plain text, no Markdown, no code blocks.
The format should look like:

------------------------------
Page 1 Text:
<exact handwritten transcription>

If multiple pages are provided, continue as:
------------------------------
Page 2 Text:
<text>
------------------------------
    """

    def process_single_page(page_data):
        """Process a single page and return (page_num, content) tuple"""
        page_num, page = page_data
        try:
            # OpenAI: send prompt + page image for OCR
            response = _grading_openai_generate([prompt, pil_to_part(page)])
            # ---------------------------------------------------------------------------
            # OLD GEMINI CODE (COMMENTED OUT)
            # ---------------------------------------------------------------------------
            # import google.generativeai as genai
            # import os
            # genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            # model = genai.GenerativeModel("gemini-2.5-flash")
            # logger.info(f"Processing page {page_num} with Gemini 2.5 Flash...")
            # response = model.generate_content(prompt)
            # response = safe_vertex_generate([prompt, pil_to_part(page)])

            content = None
            if response and getattr(response, "text", None):
                content = response.text.strip()

            if content:
                if content.startswith("```"):
                    content = content.strip("`").replace("json", "").strip()
                return (page_num, content)
            else:
                logger.warning(f"Empty response for page {page_num}")
                return (page_num, "")
        except Exception as e:
            logger.error(f"Error processing page {page_num}: {e}")
            return (page_num, "")

    # Process pages in parallel
    logger.info(f"Processing {len(pages)} pages in parallel (max {max_workers} workers)...")
    page_data_list = [(page_num, page) for page_num, page in enumerate(pages, start=1)]
    
    # Use ThreadPoolExecutor for parallel processing
    page_results = []
    effective_workers = min(max_workers, len(pages))
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=effective_workers) as executor:
        futures = {executor.submit(process_single_page, page_data): page_data[0] 
                   for page_data in page_data_list}
        
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                if result:
                    page_results.append(result)
            except Exception as e:
                page_num = futures.get(future, "unknown")
                logger.error(f"Error in future result for page {page_num}: {e}", exc_info=True)
    
    # Sort results by page number to maintain document order
    page_results.sort(key=lambda x: x[0])
    
    # Combine all pages in order
    extracted_text = ""
    for page_num, content in page_results:
        if content:
            extracted_text += f"\n\n--- Page {page_num} ---\n{content}"
    
    logger.info(f"Extracted {len(extracted_text)} characters from answer sheet ({len(pages)} pages).")
    return extracted_text


def assign_marks(question: str, correct_answer: str, student_answer: str, max_marks: int) -> Dict:
    """Uses Vertex AI (Gemini) to evaluate the student's handwritten answer and assign marks."""
    prompt = f"""
You are a strict, rule-bound examiner evaluating a student's handwritten answer sheet.

Question: {question}
Correct Answer: {correct_answer}
Student's Handwritten Answer: {student_answer}

---

### 🔥 OVERALL EVALUATION PRINCIPLE:
Award marks **only for what is explicitly written by the student**, not for what they "might have meant".  
No assumptions. No generosity. No filling gaps.

---

## 🎯 MARKING RULES (STRICTER VERSION)

Note : if the student answer is Empty or Blank ( "" ) , award 0 marks. [ VERY VERY IMPORTANT ]

### 1. Zero-tolerance for missing required points  
- If the question demands **specific items** (e.g., "herders, farmers, merchants, kings"), the answer MUST explicitly mention them.  
- If even one required item is missing → deduct marks proportionally.

### 2. Blank / Irrelevant / Incorrect → **0 marks**
- Even partially related but off-target content = 0.
- If the student answers only half the question (e.g., only definition, no explanation) → heavy deductions.

### 3. No reward for general knowledge  
- Marks only for content aligning with the **correct answer or textbook context**.  
- Irrelevant extra information → **no marks**.

### 4. Partial marks only for:
- Clearly correct points **directly answering the question**.
- Each required point contributes a **fixed fraction** of the marks.
- Vague statements that don't show clear understanding → **0**.

### 5. Specificity Required  
- General statements like "people lived differently" or "past was different for everyone" are NOT enough for 3+ mark questions.
- Examples, categories, and named items MUST appear for credit.

### 6. Structure Matters (for long answers)
Marks deducted for:
- Missing introduction
- Missing explanation
- Missing required examples
- No logical flow

### 7. Factual accuracy required  
- Wrong facts → zero marks for that portion.
- Spelling errors that change meaning → deduct marks.
- Minor spelling mistakes that do not change meaning → do not award but do not penalize heavily.

### 8. No marks for repetition or fluff  
- Rewriting the question in different words earns **no credit**.

---

## 📌 MARKING GUIDE BY QUESTION TYPE (STRICT)

### 🔸 **MCQ / One-word / Fill-in-the-Blank**
- **Primary Check:** Does the **Option Letter** (a, b, c, d) match? If yes → Full Marks.
- **Secondary Check:** Does the **Content** match? 
  - **IGNORE formatting differences** (e.g., "$120~cm^2$" vs "120 cm²", "x" vs "×").
  - If the value/meaning is identical → Full Marks.
- Otherwise → 0 Marks.

---

### 🔸 **Short Answer (1–3 marks)**
Award marks ONLY if:
- The **key phrase/term** is exactly present.
- The explanation matches the correct answer.

Penalize:
- Missing keywords
- Vague explanation
- Off-topic examples
- Incorrect definitions

---

### 🔸 **Medium-length / Reasoning (3–4 marks)**
Expect:
- Clear definition + required explanation
- All required key points

Deductions for:
- Missing examples
- Missing second part of question
- Partial conceptual understanding
- Incomplete comparisons

---

### 🔸 **Long Answer / Analytical (5–6 marks)**
Expect:
- Intro / definition
- Explanation
- Examples or points explicitly present
- All subparts answered

If ANY required component missing:
- Deduct 1–2 marks immediately.

If multiple missing:
- Award very low marks or 0.

---

### ⭐ **MARK DISTRIBUTION RULE (very strict):**
For multi-point questions:
- Each correct explicit point = (max_marks / number_of_required_points)
- Missing point = 0 for that portion.
- Vague/generalised point = 0.

---

## ✔️ FINAL OUTPUT FORMAT
You MUST return ONLY valid JSON. No markdown, no code blocks, no explanations outside the JSON.

The maximum marks for this question is {max_marks}.

Return ONLY this JSON format (nothing else):

{{
  "awarded_marks": <number between 0 and {max_marks}>,
  "remarks": "<brief strict justification>"
}}

IMPORTANT: Return ONLY the JSON object, no other text before or after it.
"""

    try:
        # OpenAI: text-only call for mark assignment
        response = _grading_openai_generate([prompt])
        # ---------------------------------------------------------------------------
        # OLD GEMINI CODE (COMMENTED OUT)
        # ---------------------------------------------------------------------------
        # import google.generativeai as genai
        # import os
        # genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        # model = genai.GenerativeModel('gemini-2.5-flash')
        # response = model.generate_content(prompt)
        # content = [types.Part(text=prompt)]
        # response = safe_vertex_generate(content)

        if not response or not getattr(response, "text", None):
            logger.warning("No response from OpenAI for mark assignment")
            return {"awarded_marks": 0, "remarks": "Error: No response from AI"}

        raw = response.text
        if not raw:
            logger.warning("No text in OpenAI response for mark assignment")
            return {"awarded_marks": 0, "remarks": "Error: No text in response"}

        raw = raw.strip()
        logger.info(f"Raw Evaluation Output (first 500 chars): {raw[:500]}")

        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and "awarded_marks" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass
        
        cleaned = raw
        
        if "```json" in cleaned:
            cleaned = re.sub(r'```json\s*', '', cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r'```\s*$', '', cleaned, flags=re.MULTILINE)
        elif "```" in cleaned:
            cleaned = re.sub(r'```\s*', '', cleaned)
        
        cleaned = cleaned.strip()
        
        json_match = re.search(r'\{[\s\S]*\}', cleaned)
        if json_match:
            cleaned = json_match.group(0)
            try:
                parsed = json.loads(cleaned)
                if isinstance(parsed, dict) and "awarded_marks" in parsed:
                    return parsed
            except json.JSONDecodeError as e:
                logger.debug(f"JSON decode error after regex extraction: {e}")
                try:
                    cleaned_json = clean_json_string(cleaned)
                    parsed = json.loads(cleaned_json)
                    if isinstance(parsed, dict) and "awarded_marks" in parsed:
                        return parsed
                except Exception as e2:
                    logger.debug(f"clean_json_string also failed: {e2}")
        
        marks_match = re.search(r'"awarded_marks"\s*:\s*(\d+)', cleaned)
        remarks_match = re.search(r'"remarks"\s*:\s*"([^"]*)"', cleaned)
        
        if marks_match:
            try:
                awarded_marks = int(marks_match.group(1))
                remarks = remarks_match.group(1) if remarks_match else "Parsed from partial response"
                logger.warning(f"Extracted partial JSON: awarded_marks={awarded_marks}")
                return {"awarded_marks": awarded_marks, "remarks": remarks}
            except:
                pass
        
        logger.error(f"Failed to parse JSON from response. Full response (first 1000 chars): {raw[:1000]}")
        return {"awarded_marks": 0, "remarks": f"Error: Could not parse JSON from AI response. Response length: {len(raw)} chars"}
            
    except Exception as e:
        logger.error(f"Error in assign_marks: {e}", exc_info=True)
        return {"awarded_marks": 0, "remarks": f"Error: {str(e)}"}

def assign_marks_section(section_title: str, questions_payload: List) -> Dict:
    """
    Evaluates all questions in a section in ONE LLM call.
    """
    prompt = f"""
    You are a strict, rule-bound examiner evaluating a student's handwritten answer sheet.

SECTION: {section_title}

---

QUESTIONS TO EVALUATE:
{json.dumps(questions_payload, indent=2)}

---

### 🔥 OVERALL EVALUATION PRINCIPLE:
Award marks **only for what is explicitly written by the student**, not for what they "might have meant".  
No assumptions. No generosity. No filling gaps.

---

## 🎯 MARKING RULES (STRICTER VERSION)

Note : if the student answer is Empty or Blank ( "" ) , award 0 marks. [ VERY VERY IMPORTANT ]

### 1. Zero-tolerance for missing required points  
- If the question demands **specific items** (e.g., "herders, farmers, merchants, kings"), the answer MUST explicitly mention them.  
- If even one required item is missing → deduct marks proportionally.

### 2. Blank / Irrelevant / Incorrect → **0 marks**
- Even partially related but off-target content = 0.
- If the student answers only half the question (e.g., only definition, no explanation) → heavy deductions.

### 3. No reward for general knowledge  
- Marks only for content aligning with the **correct answer or textbook context**.  
- Irrelevant extra information → **no marks**.

### 4. Partial marks only for:
- Clearly correct points **directly answering the question**.
- Each required point contributes a **fixed fraction** of the marks.
- Vague statements that don't show clear understanding → **0**.

### 5. Specificity Required  
- General statements like "people lived differently" or "past was different for everyone" are NOT enough for 3+ mark questions.
- Examples, categories, and named items MUST appear for credit.

### 6. Structure Matters (for long answers)
Marks deducted for:
- Missing introduction
- Missing explanation
- Missing required examples
- No logical flow

### 7. Factual accuracy required  
- Wrong facts → zero marks for that portion.
- Spelling errors that change meaning → deduct marks.
- Minor spelling mistakes that do not change meaning → do not award but do not penalize heavily.

### 8. No marks for repetition or fluff  
- Rewriting the question in different words earns **no credit**.

---

## 📌 MARKING GUIDE BY QUESTION TYPE (STRICT)

### 🔸 **MCQ / One-word / Fill-in-the-Blank**
- **Primary Check:** Does the **Option Letter** (a, b, c, d) match? If yes → Full Marks.
- **Secondary Check:** Does the **Content** match? 
  - **IGNORE formatting differences** (e.g., "$120~cm^2$" vs "120 cm²", "x" vs "×").
  - If the value/meaning is identical → Full Marks.
- Otherwise → 0 Marks.

---

### 🔸 **Short Answer (1–3 marks)**
Award marks ONLY if:
- The **key phrase/term** is exactly present.
- The explanation matches the correct answer.

Penalize:
- Missing keywords
- Vague explanation
- Off-topic examples
- Incorrect definitions

---

### 🔸 **Medium-length / Reasoning (3–4 marks)**
Expect:
- Clear definition + required explanation
- All required key points

Deductions for:
- Missing examples
- Missing second part of question
- Partial conceptual understanding
- Incomplete comparisons

---

### 🔸 **Long Answer / Analytical (5–6 marks)**
Expect:
- Intro / definition
- Explanation
- Examples or points explicitly present
- All subparts answered

If ANY required component missing:
- Deduct 1–2 marks immediately.

If multiple missing:
- Award very low marks or 0.

---

### ⭐ **MARK DISTRIBUTION RULE (very strict):**
For multi-point questions:
- Each correct explicit point = (max_marks / number_of_required_points)
- Missing point = 0 for that portion.
- Vague/generalised point = 0.

---

## ✔️ FINAL OUTPUT FORMAT
You MUST return ONLY valid JSON. No markdown, no code blocks, no explanations outside the JSON.


Return ONLY valid JSON in this format:

{{
  "evaluations": [
    {{
      "questionNo": "",
      "awarded_marks": 0,
      "remarks": ""
    }}
  ]
}}

IMPORTANT: Return ONLY the JSON object, no other text before or after it.
    """
    try:
        response = _grading_openai_generate([prompt])

        if not response or not getattr(response, "text", None):
            return {"evaluations": []}

        raw = response.text.strip()

        match = re.search(r"\{[\s\S]*\}", raw)
        if match:
            raw = match.group(0)

        raw = clean_json_string(raw)
        parsed = json.loads(raw)

        return parsed if isinstance(parsed, dict) else {"evaluations": []}

    except Exception as e:
        logger.error(f"Error in assign_marks_section: {e}", exc_info=True)
        return {"evaluations": []}


def retrieve_section_answers(section_title: str, questions: List[Dict], answer_paper: str) -> Dict:
    """Uses Vertex AI (Gemini) once per section to extract answers for all questions."""
    q_descriptions = []
    q_question_no = []
    for q in questions:
        q_snippet = q['question'][:50] + "..." if len(q['question']) > 50 else q['question']
        q_descriptions.append(f"Q{q['questionNo']}: {q_snippet}")
        q_question_no.append(q['questionNo'])

    print(f"Section Title: {section_title}")
    questions_summary = "\n".join(q_descriptions)
    questions_list = ", ".join(q_question_no)

    prompt = f"""
You are an expert exam evaluator extracting answers from a student's handwritten script.

### CONTEXT
The student's answer sheet contains multiple sections (e.g., Section A, Section B, or Section 1, 2).
Question numbers (Q1, Q2...) **repeat** in every section.


### YOUR GOAL
- **You MUST first locate the exact section header:** (HIGH PRIORITY)
    "{section_title}"

- **The answer sheet format will typically look like:**
    Section X
    Q1. ....
    Q2. ....
    Q3. ....

You are NOT allowed to extract any answer unless you have clearly identified the correct section header.

- Follow the below steps to extract the answers:
    **STEP 1 — LOCATE SECTION (MANDATORY)**
    1. Scan the entire handwritten text.
    2. Find the exact match for:
        Section Title: "{section_title}"
    3. Extraction MUST start only AFTER this header.
    4. STOP extraction when:
    - A new section header appears
    - OR the document ends

    Anything outside this section must be completely ignored.

    -----------------------------------------------

    **STEP 2 — MAP QUESTIONS WITHIN THAT SECTION ONLY**

    Inside the identified section:

    - Question numbers (Q1, Q2, Q3...) may repeat in other sections.
    - Only match question numbers that appear AFTER the correct section header.
    - Do NOT extract Q1 from another section.
    - Do NOT guess from context outside the section.

    Strict mapping rule:
    Match:
        Q1 → Answer written under Q1 in THIS section only
        Q2 → Answer written under Q2 in THIS section only

    --------------------------------------------------

--------------------------------------------------
### INPUT DATA

Section Title:
"{section_title}"

Questions List:
{questions_list}

--------------------------------------------------

### HANDWRITTEN CONTENT
---
{answer_paper}
---
--------------------------------------------------


### EXTRACTION RULES
1. **Handle Duplicate Numbers:** Do NOT extract "Q1" from Section 2 if I am asking for "Q1" from Section 1. Use the context of the answer content to disambiguate.
2. **LaTeX Formatting:** - If the answer contains math, format it as LaTeX enclosed in `$ ... $`.
   - **CRITICAL:** Use DOUBLE backslashes for commands. Example: Write `$60 \\\\text{{ km/h}}$` (not `\\text`).
   - **Multiplication:** Write `$a \\\\times b$` (produces $\times$), NOT `$a \times b$`.
   - **Text:** Write `$60 \\\\text{{ km/h}}$`.
   - **Fractions:** Write `$\\\\frac{{1}}{{2}}$`.
   - If the student wrote "atimesb", correct it to `$a \\\\times b$` if it clearly means multiplication.
3. **Precision:** Extract exactly what is written. Do not correct spelling. If an answer is missing, return an empty string.
4. **PRESERVE NEWLINES (CRITICAL):** - If the answer is written across multiple lines (e.g., steps in a math problem, or points in a theory answer), **you MUST use `\\n`** in the JSON string to represent those line breaks.
   - **DO NOT** flatten the text into a single line.
   - Example: "Step 1: Formula\\nStep 2: Substitution\\nStep 3: Answer"

Return **only valid JSON**:
{{
  "sectionTitle": "{section_title}",
  "answers": [
    {{ "questionNo": "1", "studentAnswer": "..." }},
    {{ "questionNo": "2", "studentAnswer": "..." }}
  ]
}}

- Do NOT wrap the JSON in markdown.
- Do NOT add explanations.
- Do NOT include ```json blocks.
- Return ONLY raw JSON.
"""

    try:
        # OpenAI: text-only call for section answer extraction
        response = _grading_openai_generate([prompt])
        # ---------------------------------------------------------------------------
        # OLD GEMINI CODE (COMMENTED OUT)
        # ---------------------------------------------------------------------------
        # import google.generativeai as genai
        # import os
        # genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        # model = genai.GenerativeModel('gemini-2.5-flash')
        # response = model.generate_content(prompt)
        # # content = [types.Part(text=prompt)]
        # # response = safe_vertex_generate(content)

        text = None
        if response and getattr(response, "text", None):
            text = response.text.strip()
        if not text:
            logger.warning("Empty response from OpenAI for answer extraction")
            return {}

        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            text = match.group(0)

        text = clean_json_string(text)
        
        data = json.loads(text)
        cleaned_answers = {}
        for a in data.get("answers", []):
            raw_ans = a.get("studentAnswer", "")
            cleaned_answers[a["questionNo"]] = clean_latex_content(raw_ans)

        return cleaned_answers

    except json.JSONDecodeError as e:
        logger.error(
            f"Invalid JSON returned by LLM in retrieve_section_answers. Full response:\n"
            f"{text if 'text' in locals() else 'No response'}"
        )
        return {}
    except Exception as e:
        logger.error(f"Error retrieving section answers: {e}", exc_info=True)
        return {}
# def retrieve_section_answers(section_title: str, questions: List[Dict], answer_paper: str) -> Dict:
#     """Uses Vertex AI (Gemini) once per section to extract answers for all questions."""
#     q_descriptions = []
#     for q in questions:
#         q_snippet = q['question'][:50] + "..." if len(q['question']) > 50 else q['question']
#         q_descriptions.append(f"Q{q['questionNo']}: {q_snippet}")
    
#     questions_summary = "\n".join(q_descriptions)

#     prompt = f"""
# You are an expert exam evaluator extracting answers from a student's handwritten script.

# ### CONTEXT
# The student's answer sheet contains multiple sections (e.g., Section A, Section B, or Section 1, 2).
# Question numbers (Q1, Q2...) **repeat** in every section.

# ### YOUR GOAL
# Extract the answers ONLY for the specific section described below.
# **Section Title:** "{section_title}"
# **Questions to find:**
# {questions_summary}

# ### HANDWRITTEN CONTENT
# ---
# {answer_paper}
# ---

# ### EXTRACTION RULES
# 1. **Locate the Section:** Scan the handwritten text for headers like "Section 1", "Section A", or look for answers that match the *content* of the questions listed above.
#    - *Example:* If the questions are MCQs, look for short answers like "a) 4".
#    - *Example:* If the questions are "Solve the following", look for long steps.
# 2. **Handle Duplicate Numbers:** Do NOT extract "Q1" from Section 2 if I am asking for "Q1" from Section 1. Use the context of the answer content to disambiguate.
# 3. **LaTeX Formatting:** - If the answer contains math, format it as LaTeX enclosed in `$ ... $`.
#    - **CRITICAL:** Use DOUBLE backslashes for commands. Example: Write `$60 \\\\text{{ km/h}}$` (not `\\text`).
#    - **Multiplication:** Write `$a \\\\times b$` (produces $\times$), NOT `$a \times b$`.
#    - **Text:** Write `$60 \\\\text{{ km/h}}$`.
#    - **Fractions:** Write `$\\\\frac{{1}}{{2}}$`.
#    - If the student wrote "atimesb", correct it to `$a \\\\times b$` if it clearly means multiplication.
# 4. **Precision:** Extract exactly what is written. Do not correct spelling. If an answer is missing, return an empty string.
# 5. **PRESERVE NEWLINES (CRITICAL):** - If the answer is written across multiple lines (e.g., steps in a math problem, or points in a theory answer), **you MUST use `\\n`** in the JSON string to represent those line breaks.
#    - **DO NOT** flatten the text into a single line.
#    - Example: "Step 1: Formula\\nStep 2: Substitution\\nStep 3: Answer"

# Return **only valid JSON**:
# {{
#   "answers": [
#     {{ "questionNo": "1", "studentAnswer": "..." }},
#     {{ "questionNo": "2", "studentAnswer": "..." }}
#   ]
# }}
# """

#     try:
#         # OpenAI: text-only call for section answer extraction
#         response = _grading_openai_generate([prompt])
#         # ---------------------------------------------------------------------------
#         # OLD GEMINI CODE (COMMENTED OUT)
#         # ---------------------------------------------------------------------------
#         # import google.generativeai as genai
#         # import os
#         # genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
#         # model = genai.GenerativeModel('gemini-2.5-flash')
#         # response = model.generate_content(prompt)
#         # content = [types.Part(text=prompt)]
#         # response = safe_vertex_generate(content)

#         text = None
#         if response and getattr(response, "text", None):
#             text = response.text.strip()
#         if not text:
#             logger.warning("Empty response from OpenAI for answer extraction")
#             return {}

#         match = re.search(r"\{[\s\S]*\}", text)
#         if match:
#             text = match.group(0)

#         text = clean_json_string(text)
        
#         data = json.loads(text)
#         cleaned_answers = {}
#         for a in data.get("answers", []):
#             raw_ans = a.get("studentAnswer", "")
#             cleaned_answers[a["questionNo"]] = clean_latex_content(raw_ans)
            
#         return cleaned_answers

#     except json.JSONDecodeError as e:
#         logger.error(f"Invalid JSON returned by Vertex AI. Full response:\n{text if 'text' in locals() else 'No response'}")
#         return {}
#     except Exception as e:
#         logger.error(f"Error retrieving section answers: {e}", exc_info=True)
#         return {}


def analyze_chapters_with_llm(report: Dict) -> Dict:
    """Call LLM (Vertex AI) to compute chapter-wise totals and produce strengths/weaknesses/recommendations."""
    per_question_list = []
    for section in report.get("sections", []):
        for q in section.get("questions", []):
            per_question_list.append({
                "questionNo": q.get("questionNo"),
                "chapterNo": q.get("chapterNo", q.get("chapter", "Unknown")),
                "marks": q.get("marks", 0),
                "awarded": q.get("awarded", 0),
                "studentAnswer": q.get("studentAnswer", "")
            })

    prompt = f"""
    You are an expert exam analyst. You will be given a student's evaluated question-level data.
    
    Your task:
    1) Group questions by 'chapterNo'.
    2) For EACH chapter, calculate:
       - totalMarks: sum of 'marks'
       - obtainedMarks: sum of 'awarded'
       - percentage: (obtainedMarks / totalMarks) * 100
    3) For EACH chapter, provide:
       - strengths: list (max 3) of topics/skills the student answered well.
       - weaknesses: list (max 4) of specific gaps where marks were lost.
       - recommendations: 1-2 specific actionable study tips.
    4) Produce an 'overall_summary' with:
       - strong_chapters: list of chapterNos with high percentages.
       - weak_chapters: list of chapterNos with low percentages.
       - study_plan: list of 3 bullet points for overall improvement.

    Input Data:
    {json.dumps(per_question_list, ensure_ascii=False, indent=2)}

    Output Format (Strict JSON):
    {{
      "chapters": [
        {{
          "chapterNo": "1",
          "totalMarks": 10,
          "obtainedMarks": 8,
          "percentage": 80.0,
          "strengths": ["..."],
          "weaknesses": ["..."],
          "recommendations": "..."
        }}
      ],
      "overall_summary": {{
        "strong_chapters": ["..."],
        "weak_chapters": ["..."],
        "study_plan": ["..."]
      }}
    }}
    """

    try:
        # OpenAI: text-only call for chapter analysis
        response = _grading_openai_generate([prompt])
        # ---------------------------------------------------------------------------
        # OLD GEMINI CODE (COMMENTED OUT)
        # ---------------------------------------------------------------------------
        # import google.generativeai as genai
        # import os
        # genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        # model = genai.GenerativeModel('gemini-2.5-flash')
        # response = model.generate_content(prompt)
        # content = [types.Part(text=prompt)]
        # response = safe_vertex_generate(content)

        text = None
        if response and getattr(response, "text", None):
            text = response.text.strip()
        if not text:
            logger.warning("No response from OpenAI for chapter analysis")
            return {}
        cleaned_text = clean_json_string(text)
        return json.loads(cleaned_text)

    except Exception as e:
        logger.error(f"Chapter analysis failed: {e}", exc_info=True)
        return {
            "chapters": [],
            "overall_summary": {
                "strong_chapters": [],
                "weak_chapters": [],
                "study_plan": []
            }
        }


# def evaluate_answers(question_paper: Dict, answer_paper: str, max_workers: int = 10) -> Dict:
#     """
#     Evaluate student's handwritten answers with the official question paper.
#     Uses parallel processing to evaluate multiple questions concurrently.
    
#     Args:
#         question_paper: Dictionary containing the question paper structure
#         answer_paper: Extracted text from student's handwritten answer sheet (handles answers across different pages)
#         max_workers: Maximum number of parallel workers for evaluation (default: 10)
    
#     Returns:
#         Dictionary containing evaluation results with chapter-wise analysis and token_usage (input_tokens, output_tokens, api_calls).
#     """
#     reset_grading_token_stats()
#     result = []
#     total_marks = 0
#     obtained_marks = 0

#     def evaluate_single_question(q, student_ans):
#         """Helper function to evaluate a single question"""
#         try:
#             logger.info(f"DEBUG Q{q.get('questionNo')}: Keys -> {list(q.keys())}")
#             q_text = q['question']
#             marks = q.get('marks', 0)
#             raw_ans = q.get('correct answer') or q.get('correct_answer') or q.get('correctAnswer')
#             correct_ans = clean_latex_content(raw_ans)
#             clean_student_ans = clean_latex_content(student_ans)
            
#             evaluation = assign_marks(q_text, correct_ans, clean_student_ans, marks)
            
#             logger.debug(f"Evaluation for Q{q['questionNo']}: {evaluation}")

#             try:
#                 eval_data = json.loads(evaluation) if isinstance(evaluation, str) else evaluation
#                 score = eval_data.get("awarded_marks", 0)
#                 remarks = eval_data.get("remarks", "")
#             except Exception as e:
#                 logger.warning(f"Error parsing evaluation for Q{q['questionNo']}: {e}")
#                 score = 0
#                 remarks = "Invalid response format from AI"

#             return {
#                 "questionNo": q['questionNo'],
#                 "question": q_text,
#                 "marks": marks,
#                 "chapterNo": q.get('chapterNo', q.get('chapter', 'Unknown')),
#                 "studentAnswer": clean_student_ans,
#                 "correctAnswer": correct_ans,
#                 "awarded": score,
#                 "remarks": remarks
#             }
#         except Exception as e:
#             logger.error(f"Error evaluating question {q.get('questionNo', 'unknown')}: {e}", exc_info=True)
#             return {
#                 "questionNo": q.get('questionNo', ''),
#                 "question": q.get('question', ''),
#                 "marks": q.get('marks', 0),
#                 "chapterNo": q.get('chapterNo', 'Unknown'),
#                 "studentAnswer": student_ans,
#                 "correctAnswer": q.get('correct_answer', ''),
#                 "awarded": 0,
#                 "remarks": f"Error during evaluation: {str(e)}"
#             }

#     for section in question_paper['sections']:
#         sec_title = section.get('sectionTitle') or section.get('sectionName') or section.get('title') or "Untitled Section"
        
#         section_result = {"sectionTitle": sec_title, "questions": []}

#         answers_map = retrieve_section_answers(
#             sec_title, section.get('questions', []), answer_paper
#         )

#         logger.info(f"Answers Map for section '{sec_title}': {answers_map}")

#         questions_to_evaluate = []
#         for q in section.get('questions',[]):
#             student_ans = answers_map.get(q['questionNo'], "")
#             questions_to_evaluate.append((q, student_ans))
#             total_marks += q.get('marks', 0)

#         if questions_to_evaluate:
#             try:
#                 effective_workers = min(max_workers, len(questions_to_evaluate))
#                 with concurrent.futures.ThreadPoolExecutor(max_workers=effective_workers) as executor:
#                     futures = {
#                         executor.submit(evaluate_single_question, q, student_ans): q
#                         for q, student_ans in questions_to_evaluate
#                     }
                    
#                     evaluated_questions = []
#                     for future in concurrent.futures.as_completed(futures):
#                         try:
#                             question_result = future.result()
#                             evaluated_questions.append(question_result)
#                             obtained_marks += question_result.get("awarded", 0)
#                         except Exception as e:
#                             q = futures.get(future, {})
#                             logger.error(f"Error in future result for question {q.get('questionNo', 'unknown')}: {e}", exc_info=True)
                    
#                     evaluated_questions.sort(key=lambda x: x.get("questionNo", ""))
#                     section_result['questions'] = evaluated_questions
                    
#             except Exception as e:
#                 logger.error(f"Error in parallel evaluation for section '{section['sectionTitle']}': {e}", exc_info=True)
#                 for q, student_ans in questions_to_evaluate:
#                     question_result = evaluate_single_question(q, student_ans)
#                     section_result['questions'].append(question_result)
#                     obtained_marks += question_result.get("awarded", 0)

#         result.append(section_result)

#     stats = get_grading_token_stats()
#     final_report = {
#         "Subject": question_paper["subject"],
#         "Class": question_paper["className"],
#         "totalMarks": total_marks,
#         "obtainedMarks": int(obtained_marks),
#         "sections": result,
#         "token_usage": {
#             "input_tokens": stats["input_tokens"],
#             "output_tokens": stats["output_tokens"],
#             "total_tokens": stats["input_tokens"] + stats["output_tokens"],
#             "api_calls": stats["api_calls"],
#         },
#     }

#     logger.info("Running chapter-wise analysis...")
#     chapter_summary = analyze_chapters_with_llm(final_report)
#     final_report["chapter_summary"] = chapter_summary

#     # Refresh stats after chapter analysis
#     final_report["token_usage"] = {
#         "input_tokens": _grading_stats["input_tokens"],
#         "output_tokens": _grading_stats["output_tokens"],
#         "total_tokens": _grading_stats["input_tokens"] + _grading_stats["output_tokens"],
#         "api_calls": _grading_stats["api_calls"],
#     }
#     logger.info(
#         "Evaluation complete: %s/%s marks. OpenAI: %s calls, %s input tokens, %s output tokens.",
#         obtained_marks,
#         total_marks,
#         _grading_stats["api_calls"],
#         _grading_stats["input_tokens"],
#         _grading_stats["output_tokens"],
#     )
#     return final_report


def evaluate_answers(question_paper: Dict, answer_paper: str) -> Dict:

    reset_grading_token_stats()

    result = []
    total_marks = 0
    obtained_marks = 0

    for section in question_paper['sections']:

        sec_title = (
            section.get('sectionTitle')
            or section.get('sectionName')
            or section.get('title')
            or "Untitled Section"
        )

        section_result = {"sectionTitle": sec_title, "questions": []}

        answers_map = retrieve_section_answers(
            sec_title,
            section.get('questions', []),
            answer_paper
        )

        questions_payload = []
        question_meta_map = {}

        for q in section.get('questions', []):

            student_ans = answers_map.get(q['questionNo'], "")

            correct_ans = clean_latex_content(
                q.get('correct answer')
                or q.get('correct_answer')
                or q.get('correctAnswer')
                or ""
            )

            clean_student_ans = clean_latex_content(student_ans)

            payload_item = {
                "questionNo": str(q['questionNo']),
                "question": q['question'],
                "correctAnswer": correct_ans,
                "studentAnswer": clean_student_ans,
                "maxMarks": q.get('marks', 0)
            }

            questions_payload.append(payload_item)
            question_meta_map[str(q['questionNo'])] = q
            total_marks += q.get('marks', 0)

        # 🔥 SINGLE API CALL HERE
        if questions_payload:
            evaluation_response = assign_marks_section(
                sec_title,
                questions_payload
            )

            evaluations = evaluation_response.get("evaluations", [])

            for item in evaluations:
                q_no = str(item.get("questionNo"))
                awarded = int(item.get("awarded_marks", 0))
                remarks = item.get("remarks", "")

                original_q = question_meta_map.get(q_no, {})

                obtained_marks += awarded

                section_result["questions"].append({
                    "questionNo": q_no,
                    "question": original_q.get("question", ""),
                    "marks": original_q.get("marks", 0),
                    "chapterNo": original_q.get("chapterNo", "Unknown"),
                    "studentAnswer": answers_map.get(q_no, ""),
                    "correctAnswer": original_q.get("correct_answer", ""),
                    "awarded": awarded,
                    "remarks": remarks
                })

        result.append(section_result)

    stats = get_grading_token_stats()

    final_report = {
        "Subject": question_paper["subject"],
        "Class": question_paper["className"],
        "totalMarks": total_marks,
        "obtainedMarks": int(obtained_marks),
        "sections": result,
        "token_usage": {
            "input_tokens": stats["input_tokens"],
            "output_tokens": stats["output_tokens"],
            "total_tokens": stats["input_tokens"] + stats["output_tokens"],
            "api_calls": stats["api_calls"],
        },
    }

    logger.info("Running chapter-wise analysis...")
    chapter_summary = analyze_chapters_with_llm(final_report)
    final_report["chapter_summary"] = chapter_summary

    # Refresh stats
    final_report["token_usage"] = {
        "input_tokens": _grading_stats["input_tokens"],
        "output_tokens": _grading_stats["output_tokens"],
        "total_tokens": _grading_stats["input_tokens"] + _grading_stats["output_tokens"],
        "api_calls": _grading_stats["api_calls"],
    }

    logger.info(
        "Grading OpenAI usage: %s calls, %s input tokens, %s output tokens",
        _grading_stats["api_calls"],
        _grading_stats["input_tokens"],
        _grading_stats["output_tokens"],
    )

    return final_report


# def generate_semester_report(request: SemesterReportRequest):
#     # Prepare context string
#     try:
#         eval_context = ""
#         for idx, ev in enumerate(request.evaluations):
#             percentage = (ev.get('marks_obtained', 0) / ev.get('total_marks', 1)) * 100
#             eval_context += f"""
#             Exam {idx+1}:
#             - Subject: {ev.get('subject')}
#             - Exam Type: {ev.get('exam_type')}
#             - Date: {ev.get('date')}
#             - Score: {ev.get('marks_obtained')}/{ev.get('total_marks')} ({percentage:.1f}%)
#             """

#         prompt = f"""
#         You are an expert academic analyst generating a "Semester Performance Report" for a student named {request.student_name} (Class {request.class_grade}).
        
#         Below is the student's performance record for the {request.semester} semester ({request.academic_year}):
#         {eval_context}

#         ### YOUR TASK:
#         Analyze this data and generate a structured report similar to professional employability reports (like AMCAT), but focused on school academics.
        
#         ### OUTPUT SECTIONS REQUIRED (STRICT JSON):
#         1. **summary**: A professional paragraph (approx 50 words) summarizing the student's overall performance, consistency, and major achievements.
#         2. **subject_analysis**: A list of objects, one for each unique subject.
#            - subject: Name
#            - score: Average percentage (0-100)
#            - proficiency: "Basic", "Intermediate", "Advanced", or "Expert"
#            - insight: A short, personalized comment on their performance in this subject (e.g., "Demonstrates strong conceptual clarity...").
#         3. **skill_analysis**: Derive 4-5 core academic skills based on the subjects (e.g., "Analytical Thinking" from Math/Science, "Language Comprehension" from English/History).
#            - skill: Name
#            - score: Estimated score (0-100) based on relevant subject marks.
#         4. **recommendations**: 3-4 actionable bullet points for improvement.
#         5. **highlights**: 2-3 specific areas where the student excelled.

#         ### JSON SCHEMA:
#         {{
#           "summary": "...",
#           "subject_analysis": [
#             {{ "subject": "Math", "score": 85, "proficiency": "Advanced", "insight": "..." }}
#           ],
#           "skill_analysis": [
#             {{ "skill": "Logical Reasoning", "score": 78 }}
#           ],
#           "recommendations": ["...", "..."],
#           "highlights": ["...", "..."]
#         }}
        
#         Return ONLY valid JSON. No markdown formatting.
#         """
#         # import google.generativeai as genai

#         genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))        
#         model = genai.GenerativeModel("gemini-2.5-flash")
#         response = safe_vertex_generate([prompt])
        
#         if not response or not response.candidates:
#             raise ValueError("AI failed to generate report")

#         text = response.candidates[0].content.parts[0].text.strip()
        
#         # Clean JSON
#         import re
#         if "```" in text:
#             text = re.sub(r"```(?:json)?|```", "", text).strip()
            
#         report_data = json.loads(text)
        
#         return report_data
#         # return JSONResponse(content={"success": True, "report": report_data})

#     except Exception as e:
#         logger.error(f"Semester report error: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail=str(e))



# def generate_semester_report(request: SemesterReportRequest):
#     try:
#         subject_map = {}

#         # Build subject-wise performance across exams
#         for exam_type, records in request.evaluations.items():
#             for rec in records:
#                 subject = rec.subject
#                 percentage = (rec.marks_obtained / rec.total_marks) * 100

#                 if subject not in subject_map:
#                     subject_map[subject] = []

#                 subject_map[subject].append({
#                     "exam_type": exam_type,
#                     "score": percentage
#                 })

#         # Build context for LLM
#         subject_context = ""
#         for subject, performances in subject_map.items():
#             subject_context += f"\nSubject: {subject}\n"
#             for p in performances:
#                 subject_context += f"- {p['exam_type']}: {p['score']:.1f}%\n"

#         prompt = f"""
# You are an expert academic evaluator generating a **Subject-wise Semester Performance Report**.

# Student Details:
# - Name: {request.student_name}
# - Class: {request.class_grade}
# - Semester: {request.semester}
# - Academic Year: {request.academic_year}

# Below is the student's performance across different assessments:

# {subject_context}

# ### TASK:
# Generate **deep insights for EACH SUBJECT**, analyzing performance across:
# - Unit Test
# - Mid Term
# - Final Exam

# ### INSIGHTS MUST INCLUDE:
# For each subject:
# 1. Competency demonstrated (e.g., problem-solving, conceptual clarity)
# 2. Topics typically assessed in this subject
# 3. Performance trend (improving / consistent / declining)
# 4. Level of mastery: Basic / Intermediate / Advanced / Expert

# ### OUTPUT FORMAT (STRICT JSON):
# {{
#   "summary": "...",
#   "subject_insights": [
#     {{
#       "subject": "Data Structures and Algorithms",
#       "average_score": 85,
#       "performance_trend": "Improving",
#       "competency": "Algorithmic thinking and problem solving",
#       "topics_covered": ["Arrays", "Linked Lists", "Trees", "Time Complexity"],
#       "performance_level": "Advanced",
#       "insight": "Shows steady improvement from Unit Test to Final Exam..."
#     }}
#   ],
#   "recommendations": [
#     "...",
#     "..."
#   ],
#   "strengths": [
#     "...",
#     "..."
#   ]
# }}

# Return ONLY valid JSON. No markdown. No explanations.
# """

#         genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
#         response = safe_vertex_generate([prompt])

#         if not response or not response.candidates:
#             raise ValueError("AI failed to generate report")

#         text = response.candidates[0].content.parts[0].text.strip()

#         import re, json
#         if "```" in text:
#             text = re.sub(r"```(?:json)?|```", "", text).strip()

#         return json.loads(text)

#     except Exception as e:
#         logger.error(f"Semester report generation error: {e}", exc_info=True)
#         raise HTTPException(status_code=500, detail=str(e))

def generate_semester_report(request: SemesterReportRequest):
    try:
        subject_map = {}

        # Build subject-wise performance across exams
        for exam_type, records in request.evaluations.items():
            for rec in records:
                subject = rec.subject
                percentage = (rec.marks_obtained / rec.total_marks) * 100

                if subject not in subject_map:
                    subject_map[subject] = []

                subject_map[subject].append({
                    "exam_type": exam_type,
                    "score": percentage
                })

        # Build context for LLM
        subject_context = ""
        for subject, performances in subject_map.items():
            subject_context += f"\nSubject: {subject}\n"
            for p in performances:
                subject_context += f"- {p['exam_type']}: {p['score']:.1f}%\n"

        prompt = f"""
You are an expert academic evaluator generating a **Subject-wise Semester Performance Report**.

Student Details:
- Name: {request.student_name}
- Class: {request.class_grade}
- Semester: {request.semester}
- Academic Year: {request.academic_year}

Below is the student's performance across different assessments:

{subject_context}

### TASK:
Generate **deep insights for EACH SUBJECT**, strictly based on the scores provided.

### FOR EACH SUBJECT, INCLUDE:
1. Average score across all exams
2. Performance trend (Improving / Consistent / Declining)
3. Clear competency definition (what the student is good at in this subject)
4. Test-wise insight:
   - Unit Test
   - Mid Term
   - Final Exam
5. Performance level: Basic / Intermediate / Advanced / Expert
6. Overall subject insight (1–2 sentences)

### OUTPUT FORMAT (STRICT JSON):
{{
  "summary": "Overall academic performance summary for the semester.",
  "subject_insights": [
    {{
      "subject": "Data Structures and Algorithms",
      "average_score": 85,
      "performance_trend": "Improving",
      "competency_definition": "Ability to analyze problems, select appropriate data structures, and apply efficient algorithms",
      "test_wise_insight": {{
        "Unit Test": "Demonstrated basic conceptual understanding with scope for optimization.",
        "Mid Term": "Showed improved accuracy and better handling of structured problems.",
        "Final Exam": "Exhibited strong problem-solving ability and efficient algorithm selection."
      }},
      "performance_level": "Advanced",
      "overall_insight": "The student shows consistent improvement and strong analytical thinking across assessments."
    }}
  ],
  "strengths": [
    "Strong analytical thinking",
    "Consistent improvement across subjects"
  ],
  "recommendations": [
    "Focus on advanced problem sets to further strengthen mastery",
    "Practice time-optimized solutions for competitive scenarios"
  ]
}}

Return ONLY valid JSON.
No markdown.
No explanations.
"""

        # OpenAI: text-only call for semester report
        response = _grading_openai_generate([prompt])
        # ---------------------------------------------------------------------------
        # OLD GEMINI CODE (COMMENTED OUT)
        # ---------------------------------------------------------------------------
        # genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        # response = safe_vertex_generate([prompt])
        # if not response or not response.candidates:
        #     raise ValueError("AI failed to generate report")
        # text = response.candidates[0].content.parts[0].text.strip()

        if not response or not getattr(response, "text", None):
            raise ValueError("AI failed to generate report")

        text = response.text.strip()
        if "```" in text:
            text = re.sub(r"```(?:json)?|```", "", text).strip()

        return json.loads(text)

    except Exception as e:
        logger.error(f"Semester report generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))