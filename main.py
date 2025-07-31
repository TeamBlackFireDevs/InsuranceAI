"""
InsuranceAI - Accuracy Optimized with Gemini API (Target: 70%+ accuracy, <30s response)
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
import fitz
import httpx
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import threading
import json
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(
    title="Insurance Claims Processing API - Gemini Powered",
    description="High-accuracy insurance claims processing with Gemini API <30s response time",
    version="4.1.0"
)

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Changed from HF_TOKEN
security = HTTPBearer()

# Initialize embedding model (cached globally)
embedding_model = None
http_client = None
executor = ThreadPoolExecutor(max_workers=6)

def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        # Fast, accurate embedding model
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return embedding_model

class QARequest(BaseModel):
    documents: Union[str, List[str]]
    questions: List[str]

    @validator("documents", pre=True)
    def normalize_documents(cls, v):
        if isinstance(v, str):
            return [v]
        return v

class GeminiRateLimiter:
    def __init__(self, max_requests_per_minute=15):  # Gemini has stricter limits
        self.max_requests = max_requests_per_minute
        self.requests = []
        self.lock = threading.Lock()

    async def acquire(self):
        with self.lock:
            now = time.time()
            self.requests = [req_time for req_time in self.requests if now - req_time < 60]
            
            if len(self.requests) >= self.max_requests:
                sleep_time = 4  # Longer wait for Gemini
                await asyncio.sleep(sleep_time)
                self.requests = []
            
            self.requests.append(now)

rate_limiter = GeminiRateLimiter()

def verify_bearer_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    token = credentials.credentials
    VALID_DEV_TOKENS = [
        "36ef8e0c602e88f944e5475c5ecbe62ecca6aef1702bb1a6f70854a3b993ed5"
    ]
    
    if token in VALID_DEV_TOKENS or len(token) > 10:
        return token
    
    raise HTTPException(status_code=401, detail="Invalid bearer token")

async def get_http_client():
    global http_client
    if http_client is None:
        http_client = httpx.AsyncClient(
            timeout=25,
            limits=httpx.Limits(max_connections=8, max_keepalive_connections=4)
        )
    return http_client

async def extract_pdf_enhanced(url: str) -> str:
    """Enhanced PDF extraction with better text cleaning"""
    try:
        client = await get_http_client()
        response = await client.get(url, follow_redirects=True)
        response.raise_for_status()

        with fitz.open(stream=response.content, filetype="pdf") as doc:
            # Process more pages for better coverage
            max_pages = min(30, len(doc))
            text_parts = []

            for page_num in range(max_pages):
                page = doc[page_num]
                text = page.get_text()
                
                # Better text cleaning
                text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
                text = re.sub(r'[^\w\s\.\,\:\;\-\%\$\(\)]', ' ', text)  # Keep important chars
                text_parts.append(text)

            full_text = "\n".join(text_parts)
            return full_text

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF extraction failed: {str(e)}")

def smart_chunking(text: str, chunk_size: int = 1500, overlap: int = 300) -> List[str]:
    """Improved chunking with semantic boundaries"""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Better boundary detection
        if end < len(text):
            # Look for sentence boundaries first
            for delimiter in [". ", ".\n", "\n\n", ": ", ";\n"]:
                last_pos = text.rfind(delimiter, start + chunk_size - 400, end)
                if last_pos > start + chunk_size // 3:
                    end = last_pos + len(delimiter)
                    break

        chunk = text[start:end].strip()
        if len(chunk) > 100:  # Filter very short chunks
            chunks.append(chunk)

        start = max(end - overlap, end)
        if start >= len(text):
            break

    return chunks

def enhanced_keyword_scoring(question: str, chunk: str) -> float:
    """Enhanced keyword scoring with insurance domain knowledge"""
    q_lower = question.lower()
    c_lower = chunk.lower()

    # Extract meaningful keywords (3+ chars, not stopwords)
    stopwords = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use'}
    
    q_words = set(word for word in re.findall(r'\b\w{3,}\b', q_lower) if word not in stopwords)
    if not q_words:
        return 0.0

    # Basic keyword matching
    score = sum(1 for word in q_words if word in c_lower) / len(q_words)

    # Insurance-specific term boosting
    insurance_terms = {
        'grace period': 2.0,
        'waiting period': 2.0,
        'maternity': 1.5,
        'cataract': 1.5,
        'discount': 1.5,
        'ayush': 1.5,
        'premium': 1.3,
        'coverage': 1.3,
        'deductible': 1.3,
        'copay': 1.3,
        'claim': 1.3,
        'policy': 1.2,
        'benefit': 1.2
    }

    for term, boost in insurance_terms.items():
        if term in q_lower and term in c_lower:
            score *= boost

    # Numerical information bonus
    if re.search(r'\d+\s*(days|months|years|%|\$)', c_lower):
        score *= 1.3

    # Exact phrase matching bonus
    q_phrases = re.findall(r'\b\w+\s+\w+\b', q_lower)
    for phrase in q_phrases:
        if phrase in c_lower:
            score *= 1.4

    return min(score, 3.0)  # Cap the score

def semantic_similarity_scoring(question: str, chunk: str, model) -> float:
    """Semantic similarity using embeddings"""
    try:
        q_embedding = model.encode([question])
        c_embedding = model.encode([chunk])
        similarity = cosine_similarity(q_embedding, c_embedding)[0][0]
        return float(similarity)
    except:
        return 0.0

def hybrid_chunk_retrieval(question: str, chunks: List[str], top_k: int = 5) -> List[str]:
    """Hybrid retrieval combining keyword and semantic similarity"""
    if not chunks:
        return []

    model = get_embedding_model()
    
    def score_chunk(chunk):
        keyword_score = enhanced_keyword_scoring(question, chunk)
        semantic_score = semantic_similarity_scoring(question, chunk, model)
        # Weighted combination
        return 0.6 * keyword_score + 0.4 * semantic_score

    # Score chunks in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        scores = list(executor.map(score_chunk, chunks))

    # Get top chunks with minimum threshold
    scored_chunks = [(score, chunk) for score, chunk in zip(scores, chunks) if score > 0.1]
    scored_chunks.sort(reverse=True, key=lambda x: x[0])

    return [chunk for _, chunk in scored_chunks[:top_k]]

def create_gemini_prompt(questions: List[str], context_map: Dict[str, List[str]]) -> str:
    """Create optimized prompt for Gemini API"""
    question_contexts = []

    for i, question in enumerate(questions, 1):
        # Use top 4 chunks for better context
        context = "\n---\n".join(context_map.get(question, [])[:4])
        question_contexts.append(f"Question {i}: {question}\nRelevant Context:\n{context}\n")

    # Optimized prompt for Gemini
    prompt = f"""You are an expert insurance analyst. Answer each question accurately using ONLY the provided context.

CRITICAL INSTRUCTIONS:
- Use specific details, numbers, percentages, and timeframes from the context
- If information is not in the context, respond "Information not available in provided context"
- Be precise and factual
- Do not make assumptions or add external knowledge

REQUIRED FORMAT: A1: [detailed answer] A2: [detailed answer] A3: [detailed answer]...

{chr(10).join(question_contexts)}

Remember: Answer in the exact format A1:, A2:, A3:, etc. Use only the provided context."""

    return prompt

async def call_gemini_api(prompt: str) -> str:
    """Call Google Gemini API with enhanced error handling"""
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY environment variable not set")

    await rate_limiter.acquire()

    API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={GEMINI_API_KEY}"
    headers = {
        "Content-Type": "application/json"
    }

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,  # Low temperature for consistency
            "topP": 0.8,
            "maxOutputTokens": 1500,
            "candidateCount": 1
        },
        "safetySettings": [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            }
        ]
    }

    try:
        client = await get_http_client()
        response = await client.post(API_URL, headers=headers, json=payload)

        if response.status_code == 429:
            print("Rate limited, waiting...")
            await asyncio.sleep(5)
            response = await client.post(API_URL, headers=headers, json=payload)

        response.raise_for_status()
        result = response.json()

        if 'candidates' in result and len(result['candidates']) > 0:
            if 'content' in result['candidates'][0]:
                return result['candidates'][0]['content']['parts'][0]['text']
            else:
                raise HTTPException(status_code=500, detail="Content blocked by safety filters")
        else:
            raise HTTPException(status_code=500, detail="No response from Gemini API")

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        else:
            raise HTTPException(status_code=500, detail=f"Gemini API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"API call failed: {str(e)}")

def enhanced_answer_parsing(response: str, expected_count: int) -> List[str]:
    """Enhanced answer parsing with fallback strategies"""
    # Primary pattern
    pattern = r'A(\d+):\s*(.+?)(?=A\d+:|$)'
    matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)

    answers = ["Information not available in provided context."] * expected_count

    for match in matches:
        try:
            answer_num = int(match[0]) - 1
            if 0 <= answer_num < expected_count:
                answer = match[1].strip()
                # Clean up the answer
                answer = re.sub(r'\n+', ' ', answer)
                answer = re.sub(r'\s+', ' ', answer)
                answers[answer_num] = answer
        except (ValueError, IndexError):
            continue

    # Fallback: if no matches, try splitting by numbers
    if all(ans == "Information not available in provided context." for ans in answers):
        lines = response.split('\n')
        for i, line in enumerate(lines[:expected_count]):
            if line.strip():
                answers[i] = line.strip()

    return answers

@app.get("/")
async def root():
    return {
        "message": "Insurance Claims Processing API - Gemini Powered",
        "version": "4.1.0",
        "model": "Gemini 2.0 Flash",
        "target_accuracy": "70%+",
        "target_response_time": "<30 seconds",
        "improvements": [
            "Gemini 2.0 Flash API integration",
            "Hybrid retrieval (keyword + semantic)",
            "Better chunking strategy",
            "Enhanced prompt engineering",
            "Domain-specific scoring",
            "Better text preprocessing"
        ]
    }

@app.post("/api/v1/hackrx/run")
async def document_qa_gemini(
    req: QARequest,
    token: str = Depends(verify_bearer_token)
):
    """Gemini-powered Document Q&A optimized for accuracy"""
    start_time = time.time()

    try:
        print(f"🤖 Gemini processing of {len(req.questions)} questions")

        # Step 1: Enhanced PDF extraction
        extraction_tasks = [extract_pdf_enhanced(doc) for doc in req.documents]
        pdf_texts = await asyncio.gather(*extraction_tasks)

        # Step 2: Smart chunking
        all_chunks = []
        for text in pdf_texts:
            chunks = smart_chunking(text, chunk_size=1500, overlap=300)
            all_chunks.extend(chunks)

        print(f"📚 Created {len(all_chunks)} enhanced chunks")

        # Memory cleanup
        del pdf_texts
        gc.collect()

        # Step 3: Hybrid retrieval
        async def find_chunks_for_question(question):
            return hybrid_chunk_retrieval(question, all_chunks, top_k=5)

        chunk_tasks = [find_chunks_for_question(q) for q in req.questions]
        chunk_results = await asyncio.gather(*chunk_tasks)

        context_map = dict(zip(req.questions, chunk_results))

        # Step 4: Gemini API call
        prompt = create_gemini_prompt(req.questions, context_map)
        batch_response = await call_gemini_api(prompt)

        # Step 5: Enhanced answer parsing
        answers = enhanced_answer_parsing(batch_response, len(req.questions))

        elapsed_time = time.time() - start_time
        print(f"✅ Completed in {elapsed_time:.2f} seconds")

        return {"answers": answers}

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    global http_client
    if http_client:
        await http_client.aclose()

if __name__ == "__main__":
    print("🤖 Starting Gemini-Powered Insurance API...")
    print("📈 Target: 70%+ accuracy in <30 seconds")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )