"""
Question paper generation logic using Qdrant for retrieval.
"""
import json
import re
import random
import logging
from typing import List, Dict
from copy import deepcopy
from langchain_core.prompts import PromptTemplate
from qdrant_client import QdrantClient
from services.vector_store import search_similar_chunks
from utils import clean_json_string
from dependencies import get_llm, get_json_parser, get_embeddings

logger = logging.getLogger(__name__)


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

    return PromptTemplate(
        input_variables=[
            "subject",
            "context",
            "request_data",
            "question_type",
            "num_of_questions",
            "marks_per_question",
            "class_level",
            "format_instructions"
        ],
        template="""
You are an experienced board-level examiner preparing a **final examination question paper**.

### EXAM DETAILS
- Subject: {subject}
- Class: {class_level}
- Question Type: {question_type}
- Number of Questions: {num_of_questions}
- Marks per Question: {marks_per_question}

### STRICT INSTRUCTIONS
1. Questions must be **100% syllabus-aligned**.
2. Difficulty must strictly match **Class {class_level}**.
3. No repetition of questions or patterns.
4. No references to textbook pages, figures, or examples.
5. No meta explanations.
6. Language must be formal and exam-oriented.

### QUESTION DESIGN
- Mix direct, application, and reasoning questions.
- For Multiple Choice Questions Or Single Correct Questions:
   - Create exactly 4 distinct and plausible options which can be derived from the TEXTBOOK CONTENT for every question.
   - Ensure only one correct answer.
   - The "correct answer" field must contain the full correct option text (e.g., "a) ..."). 
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
        "chapterNo": 0
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
                query = f"{chap_name} {q.get('llm_note', '')}"
                
                # Use Qdrant search with user_id filtering
                docs = search_similar_chunks(
                    client=client,
                    query_text=query,
                    embeddings=embeddings,
                    user_id=user_id,
                    k=chunks_per_chapter,
                    additional_filters=additional_filters
                )
                
                if docs:
                    logger.info(f"Retrieved {len(docs)} chunks for Chapter {chap_name} (id: {chap_id})")
                    for d in docs:
                        all_contexts.append(f"[Source: Chapter {chap_name}]\n{d.page_content}")
                else:
                    logger.warning(f"Zero docs for Chapter {chap_name} (id: {chap_id}). Check metadata or extraction.")

            except Exception as e:
                logger.error(f"Retrieval error for chapter {chap_id}: {e}")

        if not all_contexts:
            logger.error(f"CRITICAL: No valid content found for {q_type} in selected chapters.")
            continue

        random.shuffle(all_contexts)
        combined_context = "\n\n".join(all_contexts)[:25000] 

        try:
            logger.info(f"Generating Section: {q_type}")
            
            selected_prompt = get_prompt_template(subject, q_type)
            
            marks_val = q.get("marks") or q.get("marksPerQuestion") or 1
            
            final_prompt = selected_prompt.format(
                subject=subject,
                context=combined_context,
                request_data=json.dumps(q, indent=2),
                question_type=q_type,
                num_of_questions=q.get("numQuestions", 0),
                class_level=request_data.get("class", "Unknown"),
                marks_per_question=marks_val,
                format_instructions=parser.get_format_instructions()
            )
            
            response = llm.invoke(final_prompt)
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
