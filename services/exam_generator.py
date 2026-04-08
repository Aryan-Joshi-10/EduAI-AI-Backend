"""
Question paper generation logic using Qdrant for retrieval.
"""
import json
import re
import random
import logging
from typing import List, Dict, Any
from copy import deepcopy
from langchain_core.prompts import PromptTemplate
from qdrant_client import QdrantClient
from services.vector_store import search_similar_chunks
from utils import clean_json_string
from dependencies import get_llm, get_json_parser, get_embeddings
import google.generativeai as genai
from google import genai as genai_sdk 
from google.genai.types import GenerateContentConfig
import time
import base64
import concurrent.futures

logger = logging.getLogger(__name__)

# Token and API call tracking for exam generation (same pattern as grading)
_exam_gen_stats = {"input_tokens": 0, "output_tokens": 0, "api_calls": 0}


def get_exam_generator_token_stats() -> Dict[str, int]:
    """Return current exam generator token usage and API call count."""
    return _exam_gen_stats.copy()


def reset_exam_generator_token_stats() -> None:
    """Reset exam generator token and call counters."""
    _exam_gen_stats["input_tokens"] = 0
    _exam_gen_stats["output_tokens"] = 0
    _exam_gen_stats["api_calls"] = 0


def _record_llm_usage(response: Any) -> None:
    """Extract token usage from LangChain LLM response and update _exam_gen_stats."""
    _exam_gen_stats["api_calls"] += 1
    if response is None:
        return
    meta = getattr(response, "response_metadata", None) or getattr(response, "usage_metadata", None)
    if not isinstance(meta, dict):
        return
    # LangChain / Groq: token_usage or usage_metadata
    usage = meta.get("token_usage") or meta.get("usage") or meta.get("usage_metadata")
    if isinstance(usage, dict):
        _exam_gen_stats["input_tokens"] += int(usage.get("input_tokens", usage.get("prompt_tokens", 0)))
        _exam_gen_stats["output_tokens"] += int(usage.get("output_tokens", usage.get("completion_tokens", 0)))


def safe_gemini_generate(model, parts, retries=5):
    for attempt in range(retries):
        try:
            response = model.generate_content(parts)
            _exam_gen_stats["api_calls"] += 1
            # Gemini may expose usage on response
            if response and getattr(response, "usage_metadata", None):
                um = response.usage_metadata
                if um:
                    _exam_gen_stats["input_tokens"] += int(getattr(um, "prompt_token_count", 0) or 0)
                    _exam_gen_stats["output_tokens"] += int(getattr(um, "candidates_token_count", 0) or getattr(um, "completion_token_count", 0) or 0)
            return response
        except Exception as e:
            wait = min(2 ** attempt, 20)
            time.sleep(wait)
    return None

def generate_svg_visual(prompt: str) -> Dict[str, str]:
    """
    Generates SVG code using a Gemini Text Model.
    Best for: Geometry, Graphs, Charts, Simple Diagrams.
    """
    try:
        model = genai.GenerativeModel("gemini-2.0-flash-exp") # Or gemini-1.5-flash
        
        svg_prompt = f"""
        You are a coding assistant. Write raw SVG code for the following educational diagram.
        
        Request: {prompt}
        
        Constraints:
        1. Output ONLY the <svg>...</svg> code. No markdown, no comments.
        2. Use a white background (fill="white" on a rect).
        3. Make lines black (stroke="black") and clearly visible.
        4. Add labels (text) if implied by the prompt.
        5. Keep it simple and clean. Size: 300x300.
        """
        
        response = safe_gemini_generate(model, [svg_prompt])
        if not response or not response.candidates:
            return None
            
        raw_text = response.candidates[0].content.parts[0].text
        # Clean markdown
        clean_svg = re.sub(r"```svg|```xml|```", "", raw_text).strip()
        print(f"Response of SVG_Visual: {clean_svg}")
        
        # Verify it looks like SVG
        if "<svg" in clean_svg:
            return {"type": "svg", "content": clean_svg}
        return None

    except Exception as e:
        logger.error(f"SVG Generation failed: {e}")
        return None

def generate_realistic_image(prompt: str) -> Dict[str, str]:
    try:
        PROJECT_ID = "gen-lang-client-0238295665"
        LOCATION = "global"

        client = genai_sdk.Client(
            vertexai=True,
            project=PROJECT_ID,
            location=LOCATION,
        )

        logger.info(f"🎨 Generating image with Gemini for prompt: '{prompt}'")

        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=f"Draw an educational diagram, white background, clear visibility: {prompt}",
            config=GenerateContentConfig(
                response_modalities=["IMAGE"],
                candidate_count=1,
            ),
        )

        _exam_gen_stats["api_calls"] += 1
        if response and getattr(response, "usage_metadata", None):
            um = response.usage_metadata
            if um:
                _exam_gen_stats["input_tokens"] += int(getattr(um, "prompt_token_count", 0) or 0)
                _exam_gen_stats["output_tokens"] += int(getattr(um, "candidates_token_count", 0) or getattr(um, "completion_token_count", 0) or 0)

        # ---------- DEFENSIVE CHECKS ----------
        if not response or not response.candidates:
            raise ValueError("No candidates returned by Gemini")
        
        logger.debug(f"Full Gemini response: {response}")

        candidate = response.candidates[0]

        if not candidate.content or not candidate.content.parts:
            raise ValueError("Candidate has no content parts")

        for part in candidate.content.parts:
            if getattr(part, "inline_data", None):
                img_bytes = part.inline_data.data
                mime_type = part.inline_data.mime_type or "image/png"

                b64_data = base64.b64encode(img_bytes).decode("utf-8")

                logger.info("✅ Image generated successfully via Google GenAI SDK")
                return {
                    "type": "image",
                    "content": f"data:{mime_type};base64,{b64_data}",
                }

        # If no image part was found
        raise ValueError("No inline image data found in response")

    except Exception as e:
        logger.error(f"❌ Image Generation failed: {e}")
        logger.info("🔁 Falling back to SVG generation")
        return generate_svg_visual(prompt)

def process_paper_visuals(exam_paper: Dict) -> Dict:
    """
    Iterates through the generated exam paper.
    If a question requires a visual, calls the 'Artist' to generate it.
    """
    logger.info("Processing Visuals for Exam Paper...")
    
    sections = exam_paper.get("sections", [])
    if not sections: return exam_paper

    total_visuals = 0
    
    # We can use ThreadPool to generate images in parallel for speed
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_map = {}
        
        for sec_idx, section in enumerate(sections):
            for q_idx, question in enumerate(section.get("questions", [])):
                vis_annot = question.get("visual_annotation")
                correct_answers = question.get("correct_answer")
                
                # Check if visual is required
                if vis_annot and vis_annot.get("required") is True:
                    prompt = vis_annot.get("prompt")
                    v_type = vis_annot.get("type", "svg")

                    # print(f"AFTER Prompt for {question} : {prompt}")
                    print(f"Type for {question} : {v_type}")
                    
                    if prompt:
                        # Submit task
                        if v_type == "svg":
                            future = executor.submit(generate_svg_visual, prompt)
                        else:
                            future = executor.submit(generate_realistic_image, prompt)
                        
                        future_map[future] = (sec_idx, q_idx)
                        total_visuals += 1

        # Collect results
        for future in concurrent.futures.as_completed(future_map):
            sec_i, q_i = future_map[future]
            try:
                result = future.result()
                if result:
                    # Inject the generated visual into the question
                    # We create a clean "image_data" field for the frontend
                    sections[sec_i]["questions"][q_i]["image_data"] = result
                    logger.info(f"Visual generated for Q{q_i+1} (Section {sec_i+1})")
                else:
                    logger.warning(f"Visual generation failed for Q{q_i+1}")
            except Exception as e:
                logger.error(f"Error in visual worker: {e}")

    logger.info(f"Visual processing complete. Generated {total_visuals} visuals.")
    return exam_paper


def get_prompt_template(subject: str, requested_type: str):
    """Get prompt template for question generation."""
    subject = subject.lower()
    requested_type = requested_type.lower()

    latex_instruction = """
    - *LATEX REQUIREMENT:* If the answer involves numbers, equations, or units, the correct_answer field MUST be formatted in LaTeX (enclosed in $...$).
      Example: "$x = 5$" instead of "x = 5".
    """

    if any(x in requested_type for x in ['mcq', 'multiple choice', 'objective']):
        format_rules = f"""
        **FORMAT: MULTIPLE CHOICE (STRICT)**
        - Exactly 4 options: (a), (b), (c), (d).
        - Only ONE correct option.
        - No vague or overlapping options.
        - Options must be age-appropriate for the given class.
        - `correct_answer` must EXACTLY match one option.
        - + {latex_instruction}  
        """
    elif any(x in requested_type for x in ['true', 'false']):
        format_rules = """
        **FORMAT: TRUE / FALSE**
        - Only two options: True, False.
        - Statement must be clearly verifiable.
        - Avoid trick or ambiguous language.
        """
    elif any(x in requested_type for x in ['fill', 'blank']):
        format_rules = """
        **FORMAT: FILL IN THE BLANKS**
        - Only ONE blank per question.
        - No options.
        - Blank must test a key concept or calculation.
        """
    else:
        format_rules = """
        **FORMAT: SUBJECTIVE / DESCRIPTIVE**
        - No options.
        - Questions must clearly specify what is expected.
        - `correct_answer` should include key points or final answer.
        """

    if subject in ['maths', 'mathematics']:
        subject_rules = """
        ### MATHEMATICS RULES (NON-NEGOTIABLE)
        1. NO theory-based questions.
        2. Every question must be solvable numerically or logically.
        3. Do NOT ask definitions or formulas directly.
        4. Questions must be self-contained.
        5. Difficulty must match the given class level.
        6. Use $$LaTeX$$ for all mathematical expressions.
        """
    elif subject in ['science', 'physics', 'chemistry', 'biology']:
        subject_rules = """
        ### SCIENCE RULES
        1. Follow NCERT / State Board phrasing.
        2. Avoid experimental ambiguity.
        3. Diagrams should be implied, not referenced.
        4. Higher classes may include reasoning-based questions.
        """
    elif subject in ['english', 'hindi']:
        subject_rules = """
        ### LANGUAGE RULES
        - Strictly follow the textbook context provided.
        - Ensure questions test specific language skills (Reading, Writing, Grammar).
        - Do NOT hallucinate poems or stories not in the context.
        """
        if any(x in requested_type for x in ['read the extract','prose', 'comprehension', 'passage', 'extract']):
            format_rules = """
            **FORMAT: READING COMPREHENSION (BOARD PATTERN)**
            1. **MANDATORY:** You MUST extract a full paragraph (15-20 lines) from the Context and put it in the `description` field of the JSON output.
            2. **QUESTIONS:** Generate 'Activities' based *only* on that passage:
               - **A1 (Simple Factual):** True/False, Fill in blanks, or Complete sentences.
               - **A2 (Complex Factual):** Web diagram, Complete the table, or Arrange in sequence. *Describe the diagram structure in text.*
               - **A3 (Vocabulary):** Find synonyms/antonyms/phrases from passage.
               - **A4 (Grammar):** Do as directed based on sentences in the passage.
               - **A5 (Personal Response):** Open-ended question related to the passage theme.
            3. **NO MCQs.**
            """
            subject_rules = """
            - The passage MUST be verbatim from the provided context.
            - Questions must follow the standard 10th Board Activity Sheet pattern.
            """
        elif any(x in requested_type for x in ['poem', 'poetry', 'appreciation']):
            format_rules = """
            **FORMAT: POETRY COMPREHENSION**
            1. **MANDATORY:** Extract the full poem (or stanzas) from the Context and put it in the `description` field.
            2. **GENERATE ACTIVITIES:**
               - A1: Simple factual (pick out lines, true/false).
               - A2: Explain lines / Poetic Device / Rhyme Scheme.
               - A3: Appreciation / Theme of the poem.
            3. **NO MCQs.**
            """
            subject_rules += "\n- If the poem text is not in the context, return an error in the description rather than inventing one."
        elif any(x in requested_type for x in ['grammar', 'do as directed', 'language study']):
            format_rules = """
            **FORMAT: LANGUAGE STUDY (GRAMMAR)**
            - Generate standalone grammar questions (Voice, Speech, Transformation, Spot error).
            - For each question, provide the sentence and the specific instruction (e.g., "Change to Passive Voice").
            - `correct_answer` must contain the rewritten correct sentence.
            - **NO MCQs.**
            """
        elif any(x in requested_type for x in ['writing', 'letter', 'report', 'speech', 'expansion']):
            format_rules = """
            **FORMAT: WRITING SKILLS**
            - Generate Topics based on the themes in the context (e.g., "Write a letter to...", "Draft a speech on...").
            - In the `question` field, provide the prompt and any hints/points to use.
            - `correct_answer` should be a "Key Points" checklist for evaluation.
            """
    else:
        subject_rules = """
        ### GENERAL ACADEMIC RULES
        - Stick strictly to textbook facts.
        - Avoid dates/numbers not present in context.
        """

    visual_instructions = """
### VISUALS: STRICT TEXTBOOK SOURCE ONLY

**CRITICAL RULE:** You may ONLY request an image if a corresponding diagram description exists explicitly in the `TEXTBOOK CONTEXT` below (look for text tagged as `[DIAGRAM]` or explicit descriptions of figures).

--------------------------------------------------
1. NO HALLUCINATION
--------------------------------------------------
- Do NOT invent diagrams.
- If the textbook context does NOT describe a diagram for the specific topic, do NOT generate any image or visual.

--------------------------------------------------
2. MATCH TEXTBOOK CONTEXT EXACTLY
--------------------------------------------------
- The image `prompt` must be derived DIRECTLY from the `[DIAGRAM]` description.
- Do NOT add, remove, or modify elements beyond what is stated.
- Do NOT infer diagram details not explicitly mentioned.

--------------------------------------------------
3. LEVEL APPROPRIATENESS
--------------------------------------------------
- Diagram complexity, style, and labeling must match the provided `Class` level.
- Use simple, clean, exam-oriented textbook visuals.

--------------------------------------------------
4. IMAGE GENERATION IS QUESTION-TYPE DEPENDENT
--------------------------------------------------
- DO NOT generate images for:
  - MCQ questions
  - True/False questions
  - Assertion–Reason questions
  - One-line or very short answer questions

- ONLY generate images when:
  - The question explicitly requires a diagram, such as:
    - "Label the parts of the given diagram"
    - "Identify the parts marked A, B, C in the diagram"
    - "Observe the following diagram and answer"
  - AND a matching `[DIAGRAM]` exists in the textbook context.

👉 If the question can be answered without visual reference, DO NOT generate an image.

--------------------------------------------------
5. STRICT RULES FOR "LABEL THE DIAGRAM" QUESTIONS
--------------------------------------------------
When generating diagrams for labeling:

- The diagram MUST contain ONLY neutral labels:
  - Uppercase letters: A, B, C, D, etc.

- The diagram MUST NOT:
  - Display part names
  - Contain hints
  - Reveal answers directly or indirectly

- Do NOT mention which label corresponds to which part in:
  - The image
  - The image prompt
  - The question text

- The learner must be asked to identify the labels AFTER the image is shown.

--------------------------------------------------
6. LABEL PLACEMENT ACCURACY
--------------------------------------------------
- Labels must be positioned clearly and correctly near the relevant structure.
- Do NOT guess label positions.
- If accurate placement cannot be determined from textbook context:
  - DO NOT generate the image.

--------------------------------------------------
7. LABELING FORMAT STANDARD (MANDATORY)
--------------------------------------------------
- Use ONLY uppercase alphabetical labels: A, B, C, D
- No numbering, no arrows with text, no legends
- Black-and-white or grayscale only
- White background
- Clean line drawing, exam-style textbook appearance

--------------------------------------------------
8. NO ANSWER LEAKAGE (ZERO TOLERANCE)
--------------------------------------------------
- Do NOT reveal answers in:
  - The image
  - The image description
  - The question text
- Answers, if required, must appear ONLY in a separate answer key.

--------------------------------------------------
9. DECISION LOGIC (REFERENCE)
--------------------------------------------------
Scenario A:
Context: "[DIAGRAM]: A cross-section of a flower showing stamen and pistil."
Action:
- Generate diagram ONLY if question requires labeling.
Visual Annotation:
{{ "required": true, "type": "image",
   "prompt": "Educational line drawing of a flower cross-section showing internal parts. Label parts as A, B, C. Class {class_level}. White background." }}

Scenario B:
Context: Textual explanation only, no diagram mentioned.
Action:
- DO NOT generate image.
- Ask text-only question.

Scenario C (Mathematics):
Context: Geometry described.
Action:
- Create SVG visual ONLY if required by question.
Visual Annotation:
{{ "required": true, "type": "svg",
   "prompt": "Triangle PQR with angle P = 60 degrees. Clean textbook diagram." }}

--------------------------------------------------
10. FINAL SAFETY CHECK (MANDATORY)
--------------------------------------------------
Before generating ANY visual, confirm ALL:
- A `[DIAGRAM]` exists in textbook context
- Question type requires a visual
- Labels are neutral (A, B, C…)
- No answers or hints are shown
- Diagram matches class level
- Label placement is unambiguous

❌ If ANY condition fails → DO NOT generate an image
"""

# 2. Difficulty must strictly match **Class {class_level}**.
# 2. Generate Strictly Difficult Questions. Extreme Difficult Irrespective of Class level.
    return PromptTemplate(
        input_variables=[
            "subject",
            "context",
            "request_data",
            "question_type",
            "num_of_questions",
            "class_level",
            "marks_per_question",
            "difficulty",
            "format_instructions",
        ],
        template="""
You are an experienced board-level examiner preparing a **final examination question paper**.

### EXAM DETAILS
- Subject: {subject}
- Class: {class_level}
- Question Type: {question_type}
- Number of Questions: {num_of_questions}
- Marks per Question: {marks_per_question}
- Difficulty of Question: {difficulty}

### STRICT INSTRUCTIONS
1. Questions must be **100% syllabus-aligned**.
2. Difficulty must strictly match **Class {class_level}**.
3. No repetition of questions or patterns.
4. No references to textbook pages, figures, or examples.
5. No meta explanations.
6. Language must be formal and exam-oriented.

### Visual Instructions
""" + visual_instructions + """
Make the prompts enhanced for generating images.
Example:
Educational line diagram of a rectangular plant cell.
Include:
- A thick outer boundary representing the cell wall
- A thin inner boundary representing the cell membrane
- A large central vacuole occupying most of the cell
- A circular nucleus near the periphery
- Several oval organelle shapes distributed in the cytoplasm

Label exactly four distinct structures using ONLY:
A, B, C, D

*********STRICT INSTRUCTIONS***********:
- Each label must appear exactly once
- DO NOT SHOW/DISPLAY ANSWERS ON THE GENERATED IMAGE
- Do NOT include legends or captions
- Black lines on white background
- Simple NCERT-style textbook diagram
- Class {class_level} appropriate


### QUESTION DESIGN
- Mix direct, application, and reasoning questions.
- For Multiple Choice Questions Or Single Correct Questions:
   - Create exactly 4 distinct and plausible options which can be derived from the TEXTBOOK CONTENT for every question.
   - Ensure only one correct answer.
   - The "correct answer" field must contain the full correct option text (e.g., "a) ..."). 
- FOR ENGLISH & HINDI:
    - visual_annotation.required MUST ALWAYS be false
    - UNLESS explicitly requested AND [DIAGRAM] exists
- Clearly mention the chapter number from which each question is generated or selected.. 
- Avoid unnecessary complexity for lower classes.
- Ensure fairness and clarity.

### FORMAT RULES
""" + format_rules + """

### SUBJECT RULES
""" + subject_rules + """

### INPUT FROM USER
{request_data}

### TEXTBOOK CONTEXT (AUTHORITATIVE)
{context}

### OUTPUT FORMAT (MANDATORY)
{format_instructions}

SCHEMA TO FOLLOW:
{{
    "sectionTitle": "", // question_type
    "description": "",  // Description about question_type
    "questions": [
      {{
        "questionNo": "",
        "question": "",
        "options": [], // Only for MCQs
        "marks": 0,
        "correct_answer": "",
        "chapterNo": 0,
        "visual_annotation": {{
          "required": boolean,     // TRUE only if context contains [DIAGRAM] for this topic.
          "type": "svg" | "image",
          "prompt": "string"       // Must match the [DIAGRAM] description from context.
        }}
      }}
    ]
}}

Return ONLY valid JSON.
"""
    )


def get_context_from_request(
    client: QdrantClient,
    request_data: dict,
    user_id: str,
    k: int = 15
) -> Dict:
    """Generate exam paper using Qdrant retrieval with user_id filtering."""
    if not request_data or "questions" not in request_data:
        raise ValueError("Invalid request data")
    
    logger.info(f"Generating exam for user {user_id} using Qdrant retrieval...")
    
    exam_paper = {
        "subject": request_data.get("subject", ""),
        "className": request_data.get("class", ""),
        "maxMarks": request_data.get("maxMarks", 0),
        "timeAllowed": request_data.get("timeAllowed", ""),
        "instructions": request_data.get("instructions", []),
        "sections": []
    }

    embeddings = get_embeddings()
    llm = get_llm()
    parser = get_json_parser()

    subject = request_data.get("subject", "").lower()
    is_language = subject in ['english', 'hindi', 'marathi', 'sanskrit']

    used_chunk_ids = set()

    for q_idx, q in enumerate(request_data["questions"]):
        q_type = q.get("type", "Unknown")
        topics = q.get("topics", [])
        
        target_chapters = [] 
        for t in topics:
            match = re.search(r"(\d+)", str(t))
            if match:
                chap_id = match.group(1)
                chap_name = str(t)
                target_chapters.append((chap_id, chap_name))
            else:
                target_chapters.append((str(t), str(t)))

        if not target_chapters: 
            logger.warning(f"Could not parse any chapter numbers from topics: {topics}")
            continue

        all_contexts = []
        
        base_k = 30 if is_language else 15
        chunks_per_chapter = max(5, base_k // len(target_chapters))

        for chap_id, chap_name in target_chapters:
            try:
                # Build additional filters for PDF name and chapter
                additional_filters = {}
                if request_data.get("pdf_name"):
                    additional_filters["pdf_name"] = request_data.get("pdf_name")
                
                # Match by chapter_no if we have a number
                # If no number, rely on semantic search (query includes chapter name)
                if chap_id.isdigit():
                    additional_filters["chapter_no"] = str(chap_id).strip()
                # Note: We don't filter by chapter_name when no number because it requires an index
                # Semantic search with chapter name in query will find relevant chunks
                
                # Search query - includes chapter name for semantic matching
                # query = f"{chap_name} {q.get('llm_note', '')}"
                difficulty = q.get("difficulty", "").lower()
                
                query_generation_prompt = f"""
                Generate 5 different semantic search queries 
                for retrieving diverse and non-overlapping content 
                from chapter '{chap_name}' 
                for generating '{q_type}' questions 
                of difficulty '{difficulty}'.

                Make sure:
                - Queries focus on different subtopics
                - Cover conceptual, theoretical, and application angles
                - Return ONLY a Python list of strings
                """

                query_response = llm.invoke(query_generation_prompt)
                generated_queries = json.loads(clean_json_string(query_response.content))
                
                chapter_docs = []

                for query in generated_queries:
                    docs = search_similar_chunks(
                        client = client,
                        query_text=query,
                        embeddings=embeddings,
                        user_id=user_id,
                        k=max(4, chunks_per_chapter // len(generated_queries)),
                        additional_filters=additional_filters
                    )

                    if docs:
                        chapter_docs.extend(docs)

                
                if chapter_docs:
                    logger.info(f"Retrieved {len(chapter_docs)} chunks for Chapter {chap_name} (id: {chap_id})")
                    for d in chapter_docs:
                        print(f"[Source: Chapter {chap_name}]\n{d.page_content}")
                        all_contexts.append(f"[Source: Chapter {chap_name}]\n{d.page_content}")
                else:
                    logger.warning(f"Zero docs for Chapter {chap_name} (id: {chap_id}). Check metadata or extraction.")
                
            except Exception as e:
                logger.error(f"Retrieval error for chapter {chap_id}: {e}")

        if not all_contexts:
            logger.error(f"CRITICAL: No valid content found for {q_type} in selected chapters.")
            continue

        random.shuffle(all_contexts)
        print(len("\n\n".join(all_contexts)))
        combined_context = "\n\n".join(all_contexts)[:25000] 
        # combined_context = "\n\n".join(all_contexts[:8])

        try:
            logger.info(f"Generating Section: {q_type}")
            
            selected_prompt = get_prompt_template(subject, q_type)
            
            marks_val = q.get("marks") or q.get("marksPerQuestion") or 1

            difficulty = q.get("difficulty", "").lower()
            print("\n" + difficulty + "\n")
            
            final_prompt = selected_prompt.format(
                subject=subject,
                context=combined_context,
                request_data=json.dumps(q, indent=2),
                question_type=q_type,
                num_of_questions=q.get("numQuestions", 0),
                class_level=request_data.get("class", "Unknown"),
                marks_per_question=marks_val,
                difficulty=difficulty,
                format_instructions=parser.get_format_instructions()
            )
            
            response = llm.invoke(final_prompt)
            _record_llm_usage(response)
            cleaned_text = clean_json_string(response.content)
            parsed = json.loads(cleaned_text)
            
            
            if isinstance(parsed, dict):
                exam_paper["sections"].append({
                    "sectionName": q.get("sectionName", ""),
                    "sectionTitle": parsed.get("sectionTitle", ""),
                    "description": parsed.get("description", ""),
                    "questions": parsed.get("questions", [])
                })
        except Exception as e:
            logger.error(f"Error generating section {q_type}: {e}")

    return exam_paper


def summarize_questions(llm, questions: List[str]) -> List[str]:
    """Summarize questions into conceptual summaries."""
    if not questions:
        return []
    
    if not llm:
        logger.error("LLM instance is None")
        return []

    try:
        prompt = f"""
        Summarize the following questions into short one-line conceptual summaries 
        that capture what each question is testing (without copying exact phrasing).

        Questions:
        {json.dumps(questions, indent=2)}

        Return as a bullet list, no numbering, no extra text.
        """

        resp = llm.invoke(prompt)
        _record_llm_usage(resp)
        if not resp or not hasattr(resp, 'content'):
            return []
        
        lines = [line.strip("-• ").strip() for line in resp.content.split("\n") if line.strip()]
        logger.info(f"Generated {len(lines)} question summaries")
        return lines
    except Exception as e:
        logger.error(f"Summary generation failed: {e}", exc_info=True)
        return []


def generate_multiple_papers_with_summaries(
    client: QdrantClient,
    request_data: dict,
    user_id: str,
    num_papers: int = 1,
    k: int = 10
) -> List[Dict]:
    """Generate multiple diverse question papers."""
    if not request_data or not isinstance(request_data, dict):
        raise ValueError("request_data must be a non-empty dictionary")
    
    if num_papers <= 0:
        raise ValueError("num_papers must be greater than 0")
    
    logger.info(f"Generating {num_papers} diverse question papers for user {user_id}")
    reset_exam_generator_token_stats()

    generated_papers = []
    previous_concept_summaries = []
    llm = get_llm()

    for paper_no in range(num_papers):
        try:
            logger.info(f"Generating Paper {paper_no + 1}/{num_papers}")

            if previous_concept_summaries:
                diversity_hint = (
                    "Avoid creating questions similar to these concepts:\n"
                    + "\n".join(previous_concept_summaries[-40:])
                )
            else:
                diversity_hint = "Create unique and diverse questions covering all given topics."

            try:
                modified_request = deepcopy(request_data)
            except Exception as e:
                logger.error(f"Error cloning request data: {e}", exc_info=True)
                raise
            
            if "questions" not in modified_request:
                raise ValueError("request_data must contain 'questions' key")
            
            for q in modified_request["questions"]:
                if not isinstance(q, dict):
                    continue
                if "llm_note" in q and isinstance(q["llm_note"], list):
                    q["llm_note"].append(diversity_hint)
                else:
                    q["llm_note"] = [diversity_hint]

            try:
                paper = get_context_from_request(client, modified_request, user_id, k=k)
                if not paper or not isinstance(paper, dict):
                    logger.warning(f"Invalid paper generated for paper {paper_no + 1}")
                    continue
                generated_papers.append(paper)
            except Exception as e:
                logger.error(f"Error generating paper {paper_no + 1}: {e}", exc_info=True)
                continue

            all_questions = []
            for sec in paper.get("sections", []):
                if not isinstance(sec, dict):
                    continue
                for ques in sec.get("questions", []):
                    if isinstance(ques, dict):
                        question_text = ques.get("question", "")
                        if question_text:
                            all_questions.append(question_text)

            if all_questions:
                new_summaries = summarize_questions(llm, all_questions)
                previous_concept_summaries.extend(new_summaries)
                previous_concept_summaries = previous_concept_summaries[-100:]
        except Exception as e:
            logger.error(f"Unexpected error generating paper {paper_no + 1}: {e}", exc_info=True)
            continue

    logger.info(f"Successfully generated {len(generated_papers)}/{num_papers} papers")
    return generated_papers
