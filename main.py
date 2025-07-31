"""
InsuranceAI - Robust Generic PDF Parser
----
Key features:
1. Adaptive text extraction for any document structure
2. Smart chunking that preserves context
3. Generic keyword matching without hardcoded terms
4. Flexible retrieval that works with any policy type
5. Robust error handling and fallback mechanisms
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, validator
import os
import time
import asyncio
from typing import List, Dict, Union
import gc
import requests
import uvicorn
import traceback
import re
import fitz  # PyMuPDF
import httpx
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import Counter

# Initialize FastAPI app
app = FastAPI(
    title="Insurance Claims Processing API - Robust",
    description="Robust insurance claims processing for any policy document",
    version="5.0.0"
)

load_dotenv()
LLM_KEY = os.getenv("HF_TOKEN")
security = HTTPBearer()

# Global HTTP client for connection reuse
http_client = None
executor = ThreadPoolExecutor(max_workers=4)

class QARequest(BaseModel):
    documents: Union[str, List[str]]
    questions: List[str]

    @validator("documents", pre=True)
    def normalize_documents(cls, v):
        if isinstance(v, str):
            return [v]
        return v

class SpeedRateLimiter:
    def __init__(self, max_requests_per_minute=60):
        self.max_requests = max_requests_per_minute
        self.requests = []
        self.lock = threading.Lock()

    def acquire(self):
        with self.lock:
            now = time.time()
            self.requests = [req_time for req_time in self.requests if now - req_time < 60]

            if len(self.requests) >= self.max_requests:
                sleep_time = 2
                time.sleep(sleep_time)
                self.requests = []

            self.requests.append(now)

rate_limiter = SpeedRateLimiter()

def verify_bearer_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Token verification"""
    token = credentials.credentials

    VALID_DEV_TOKENS = [
        "36ef8e0c602e88f944e5475c5ecbe62ecca6aef1702bb1a6f70854a3b993ed5"
    ]

    if token in VALID_DEV_TOKENS or len(token) > 10:
        return token

    raise HTTPException(status_code=401, detail="Invalid bearer token")

async def get_http_client():
    """Get or create HTTP client with connection pooling"""
    global http_client
    if http_client is None:
        http_client = httpx.AsyncClient(
            timeout=30,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
        )
    return http_client

async def extract_pdf_robust(url: str) -> str:
    """Robust PDF extraction that handles any document structure"""
    try:
        client = await get_http_client()
        response = await client.get(url, follow_redirects=True)
        response.raise_for_status()

        with fitz.open(stream=response.content, filetype="pdf") as doc:
            text_parts = []

            for page_num in range(len(doc)):
                page = doc[page_num]

                # Extract text with multiple methods for robustness
                text = page.get_text()

                # If text is sparse, try different extraction methods
                if len(text.strip()) < 100:
                    # Try extracting with layout preservation
                    text = page.get_text("dict")
                    if isinstance(text, dict) and "blocks" in text:
                        text_blocks = []
                        for block in text["blocks"]:
                            if "lines" in block:
                                for line in block["lines"]:
                                    if "spans" in line:
                                        line_text = " ".join([span.get("text", "") for span in line["spans"]])
                                        if line_text.strip():
                                            text_blocks.append(line_text.strip())
                        text = "\n".join(text_blocks)

                # Clean and normalize text
                if isinstance(text, str) and text.strip():
                    # Basic cleaning
                    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
                    text = re.sub(r'\n+', '\n', text)  # Normalize line breaks
                    text = text.strip()

                    if text:
                        text_parts.append(f"[Page {page_num + 1}]\n{text}")

            full_text = "\n\n".join(text_parts)

            # Final cleanup
            full_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', full_text)

            return full_text

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF extraction failed: {str(e)}")

def smart_chunking(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    """Smart chunking that preserves context and handles any document structure"""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Smart boundary detection - try multiple strategies
        if end < len(text):
            # Strategy 1: Look for natural breaks (paragraphs, sections)
            boundary_patterns = [
                (r'\n\n+', 2),           # Paragraph breaks
                (r'\n[A-Z][^\n]*:?\n', 1),  # Section headers
                (r'\. [A-Z]', 2),         # Sentence boundaries
                (r'\n', 1),               # Line breaks
                (r', ', 2),                # Comma breaks
                (r' ', 1)                  # Word breaks
            ]

            best_end = end
            for pattern, priority in boundary_patterns:
                matches = list(re.finditer(pattern, text[start:end + 100]))
                if matches:
                    # Find the best match (closest to target end)
                    best_match = min(matches, key=lambda m: abs((start + m.end()) - end))
                    candidate_end = start + best_match.end()

                    # Only use if it's within reasonable bounds
                    if start + chunk_size // 2 <= candidate_end <= end + 50:
                        best_end = candidate_end
                        break

            end = best_end

        chunk = text[start:end].strip()

        # Only add substantial chunks
        if chunk and len(chunk) > 50:
            chunks.append(chunk)

        # Smart overlap calculation
        overlap_size = min(overlap, len(chunk) // 4)
        start = max(end - overlap_size, start + chunk_size // 2)

        if start >= len(text):
            break

    return chunks

def extract_key_terms(text: str) -> List[str]:
    """Extract key terms from text for better matching"""
    # Extract numbers with units (common in insurance)
    numbers_with_units = re.findall(r'\d+\s*(?:days?|months?|years?|%|percent|rs\.?|rupees?)', text.lower())

    # Extract capitalized terms (likely important concepts)
    capitalized_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)

    # Extract quoted terms
    quoted_terms = re.findall(r'"([^"]+)"', text)
    quoted_terms.extend(re.findall(r"'([^']+)'", text))

    # Extract terms that appear to be definitions
    definition_terms = re.findall(r'\b(\w+)\s+(?:means?|refers?\s+to|defined\s+as|is\s+defined)', text.lower())

    all_terms = numbers_with_units + capitalized_terms + quoted_terms + definition_terms
    return list(set(term.strip() for term in all_terms if len(term.strip()) > 2))

def calculate_semantic_similarity(question: str, chunk: str) -> float:
    """Calculate semantic similarity without hardcoded patterns"""
    q_lower = question.lower()
    c_lower = chunk.lower()

    # Extract meaningful words (3+ characters, not common stop words)
    stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'she', 'use', 'her', 'way', 'many', 'oil', 'sit', 'set', 'run', 'eat', 'far', 'sea', 'eye'}

    q_words = set(word for word in re.findall(r'\b\w{3,}\b', q_lower) if word not in stop_words)
    c_words = set(word for word in re.findall(r'\b\w{3,}\b', c_lower) if word not in stop_words)

    if not q_words:
        return 0.0

    # Basic word overlap
    common_words = q_words.intersection(c_words)
    word_overlap_score = len(common_words) / len(q_words)

    # Phrase matching (2-3 word phrases)
    q_phrases = set()
    c_phrases = set()

    q_tokens = q_lower.split()
    c_tokens = c_lower.split()

    # Extract 2-word phrases
    for i in range(len(q_tokens) - 1):
        phrase = f"{q_tokens[i]} {q_tokens[i+1]}"
        if len(phrase) > 6:  # Avoid very short phrases
            q_phrases.add(phrase)

    for i in range(len(c_tokens) - 1):
        phrase = f"{c_tokens[i]} {c_tokens[i+1]}"
        if len(phrase) > 6:
            c_phrases.add(phrase)

    phrase_overlap = len(q_phrases.intersection(c_phrases))
    phrase_score = phrase_overlap * 0.5  # Bonus for phrase matches

    # Number and percentage matching
    q_numbers = set(re.findall(r'\d+', q_lower))
    c_numbers = set(re.findall(r'\d+', c_lower))
    number_overlap = len(q_numbers.intersection(c_numbers))
    number_score = number_overlap * 0.3

    # Key term extraction and matching
    chunk_key_terms = extract_key_terms(chunk)
    question_key_terms = extract_key_terms(question)

    key_term_matches = 0
    for q_term in question_key_terms:
        for c_term in chunk_key_terms:
            if q_term.lower() in c_term.lower() or c_term.lower() in q_term.lower():
                key_term_matches += 1
                break

    key_term_score = key_term_matches * 0.4

    # Position-based scoring (earlier in document might be more important)
    position_score = 0.1 if '[Page 1]' in chunk else 0.05 if '[Page 2]' in chunk else 0

    total_score = word_overlap_score + phrase_score + number_score + key_term_score + position_score

    return min(total_score, 1.0)  # Cap at 1.0

def find_relevant_chunks(question: str, chunks: List[str], top_k: int = 6) -> List[str]:
    """Find relevant chunks using semantic similarity"""
    if not chunks:
        return []

    # Calculate similarity scores
    def score_chunk(chunk):
        return calculate_semantic_similarity(question, chunk)

    with ThreadPoolExecutor(max_workers=4) as executor:
        scores = list(executor.map(score_chunk, chunks))

    # Get scored chunks
    scored_chunks = list(zip(scores, chunks))

    # Sort by score
    scored_chunks.sort(reverse=True, key=lambda x: x[0])

    # Filter out very low scores and get top chunks
    relevant_chunks = [chunk for score, chunk in scored_chunks if score > 0.05][:top_k]

    # If we don't have enough relevant chunks, add some with lower scores
    if len(relevant_chunks) < 3:
        additional_chunks = [chunk for score, chunk in scored_chunks[len(relevant_chunks):] if score > 0.01]
        relevant_chunks.extend(additional_chunks[:3-len(relevant_chunks)])

    return relevant_chunks

def create_robust_prompt(questions: List[str], context_map: Dict[str, List[str]]) -> List[Dict]:
    """Create a robust prompt that works with any policy document"""
    question_contexts = []

    for i, question in enumerate(questions, 1):
        contexts = context_map.get(question, [])

        if contexts:
            # Combine contexts intelligently
            combined_context = "\n\n---\n\n".join(contexts[:4])  # Use top 4 contexts
        else:
            combined_context = "No relevant context found in the document."

        question_contexts.append(f"Question {i}: {question}\n\nRelevant Context:\n{combined_context}\n")

    batch_prompt = f"""You are an expert document analyst. Answer each question based ONLY on the provided context from the policy document.

INSTRUCTIONS:
- Use only information explicitly stated in the context
- If the context doesn't contain enough information, respond with "Information not available in the provided context"
- Be specific with numbers, percentages, time periods, and conditions
- Quote relevant parts when helpful
- Don't make assumptions or add information not in the context

Answer format: A1: [your answer] A2: [your answer] A3: [your answer]...

{chr(10).join(question_contexts)}"""

    return [
        {"role": "system", "content": "You are a precise document analyst. Answer questions using only the provided context. Be accurate and specific."},
        {"role": "user", "content": batch_prompt}
    ]

async def call_api_robust(messages: List[Dict], max_tokens: int = 1000) -> str:
    """Robust API call with proper error handling"""
    if not LLM_KEY:
        raise HTTPException(status_code=500, detail="HF_TOKEN not set")

    rate_limiter.acquire()

    API_URL = "https://router.huggingface.co/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {LLM_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "messages": messages,
        "model": "mistralai/Mistral-7B-Instruct-v0.2:featherless-ai",
        "max_tokens": max_tokens,
        "temperature": 0.1,
        "top_p": 0.3,
        "stream": False
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)

        if response.status_code == 429:
            await asyncio.sleep(3)
            response = requests.post(API_URL, headers=headers, json=payload, timeout=30)

        response.raise_for_status()
        result = response.json()

        if 'choices' in result and len(result['choices']) > 0:
            return result['choices'][0]['message']['content']
        else:
            raise HTTPException(status_code=500, detail="Unexpected API response")

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"API call failed: {str(e)}")

def parse_answers_robust(response: str, expected_count: int) -> List[str]:
    """Robust answer parsing that handles various formats"""
    # Multiple parsing strategies
    patterns = [
        r'A(\d+):\s*(.+?)(?=\nA\d+:|$)',  # A1: answer (with newline)
        r'A(\d+):\s*(.+?)(?=A\d+:|$)',     # A1: answer (without newline)
        r'Answer\s+(\d+):\s*(.+?)(?=Answer\s+\d+:|$)',  # Answer 1: answer
        r'(\d+)\)\s*(.+?)(?=\n\d+\)|$)',  # 1) answer
        r'Question\s+\d+[^:]*:\s*(.+?)(?=Question\s+\d+|$)'  # Question format
    ]

    answers = ["Information not available in the provided context."] * expected_count

    for pattern in patterns:
        matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)

        if matches:
            filled_count = 0
            for match in matches:
                try:
                    if len(match) == 2:  # (number, answer)
                        answer_num = int(match[0]) - 1
                        answer_text = match[1].strip()
                    else:  # just answer
                        answer_num = filled_count
                        answer_text = match[0].strip()

                    if 0 <= answer_num < expected_count and answer_text:
                        # Clean up the answer
                        answer_text = re.sub(r'\n+', ' ', answer_text)
                        answer_text = re.sub(r'\s+', ' ', answer_text)
                        answers[answer_num] = answer_text.strip()
                        filled_count += 1

                except (ValueError, IndexError):
                    continue

            if filled_count > 0:
                break  # Use first successful pattern

    return answers

@app.get("/")
async def root():
    return {
        "message": "Insurance Claims Processing API - Robust Generic Version",
        "version": "5.0.0",
        "features": [
            "Works with any policy document type",
            "No hardcoded brand-specific terms",
            "Adaptive text extraction",
            "Smart semantic chunking",
            "Robust error handling",
            "Generic keyword matching"
        ]
    }

@app.post("/api/v1/hackrx/run")
async def document_qa_robust(
    req: QARequest,
    token: str = Depends(verify_bearer_token)
):
    """Robust Document Q&A that works with any policy document"""
    start_time = time.time()

    try:
        print(f"🚀 Processing {len(req.questions)} questions with robust parser")

        # Step 1: Robust PDF extraction
        extraction_tasks = [extract_pdf_robust(doc) for doc in req.documents]
        pdf_texts = await asyncio.gather(*extraction_tasks)

        # Step 2: Smart chunking
        all_chunks = []
        for text in pdf_texts:
            chunks = smart_chunking(text, chunk_size=1200, overlap=200)
            all_chunks.extend(chunks)

        print(f"📚 Created {len(all_chunks)} smart chunks")

        # Memory cleanup
        del pdf_texts
        gc.collect()

        # Step 3: Find relevant chunks for each question
        async def find_chunks_for_question(question):
            return find_relevant_chunks(question, all_chunks, top_k=6)

        chunk_tasks = [find_chunks_for_question(q) for q in req.questions]
        chunk_results = await asyncio.gather(*chunk_tasks)

        context_map = dict(zip(req.questions, chunk_results))

        # Step 4: Create robust prompt and call API
        messages = create_robust_prompt(req.questions, context_map)
        batch_response = await call_api_robust(messages)

        # Step 5: Parse answers robustly
        answers = parse_answers_robust(batch_response, len(req.questions))

        elapsed_time = time.time() - start_time
        print(f"✅ Completed in {elapsed_time:.2f} seconds")

        return {"answers": answers}

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global http_client
    if http_client:
        await http_client.aclose()

if __name__ == "__main__":
    print("🚀 Starting Robust Generic Insurance API...")
    print("📄 Works with any policy document")
    print("🎯 No hardcoded terms or brands")

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
