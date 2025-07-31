"""
InsuranceAI - Accuracy Optimized with Gemini API (No sentence-transformers)
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
from collections import Counter
import math

app = FastAPI(
    title="Insurance Claims Processing API - Gemini Powered",
    description="High-accuracy insurance claims processing with Gemini API <30s response time",
    version="4.2.0"
)

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
security = HTTPBearer()

http_client = None
executor = ThreadPoolExecutor(max_workers=6)

class QARequest(BaseModel):
    documents: Union[str, List[str]]
    questions: List[str]

    @validator("documents", pre=True)
    def normalize_documents(cls, v):
        if isinstance(v, str):
            return [v]
        return v

class GeminiRateLimiter:
    def __init__(self, max_requests_per_minute=15):
        self.max_requests = max_requests_per_minute
        self.requests = []
        self.lock = threading.Lock()

    async def acquire(self):
        with self.lock:
            now = time.time()
            self.requests = [req_time for req_time in self.requests if now - req_time < 60]
            
            if len(self.requests) >= self.max_requests:
                sleep_time = 4
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
            max_pages = min(30, len(doc))
            text_parts = []

            for page_num in range(max_pages):
                page = doc[page_num]
                text = page.get_text()
                
                # Better text cleaning
                text = re.sub(r'\s+', ' ', text)
                text = re.sub(r'[^\w\s\.\,\:\;\-\%\$\(\)]', ' ', text)
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

        if end < len(text):
            for delimiter in [". ", ".\n", "\n\n", ": ", ";\n"]:
                last_pos = text.rfind(delimiter, start + chunk_size - 400, end)
                if last_pos > start + chunk_size // 3:
                    end = last_pos + len(delimiter)
                    break

        chunk = text[start:end].strip()
        if len(chunk) > 100:
            chunks.append(chunk)

        start = max(end - overlap, end)
        if start >= len(text):
            break

    return chunks

def advanced_keyword_scoring(question: str, chunk: str) -> float:
    """Advanced keyword scoring with TF-IDF-like approach and domain knowledge"""
    q_lower = question.lower()
    c_lower = chunk.lower()

    # Enhanced stopwords list
    stopwords = {
        'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 
        'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 
        'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use', 'will',
        'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'about', 'after', 'before',
        'during', 'under', 'over', 'through', 'between', 'among', 'within', 'without', 'against'
    }
    
    # Extract meaningful keywords
    q_words = [word for word in re.findall(r'\b\w{3,}\b', q_lower) if word not in stopwords]
    c_words = [word for word in re.findall(r'\b\w{3,}\b', c_lower) if word not in stopwords]
    
    if not q_words:
        return 0.0

    # Calculate word frequencies in chunk
    c_word_freq = Counter(c_words)
    total_c_words = len(c_words)

    # TF-IDF-like scoring
    score = 0.0
    matched_words = 0
    
    for q_word in q_words:
        if q_word in c_word_freq:
            # Term frequency in chunk
            tf = c_word_freq[q_word] / total_c_words if total_c_words > 0 else 0
            # Simple IDF approximation (longer words are more important)
            idf = math.log(len(q_word)) if len(q_word) > 3 else 1
            score += tf * idf
            matched_words += 1

    # Base score
    base_score = (matched_words / len(q_words)) * (score / len(q_words)) if q_words else 0

    # Insurance domain-specific boosting
    insurance_terms = {
        'grace period': 3.0,
        'waiting period': 3.0,
        'maternity benefit': 2.5,
        'maternity coverage': 2.5,
        'cataract surgery': 2.5,
        'cataract treatment': 2.5,
        'discount rate': 2.0,
        'discount percentage': 2.0,
        'ayush treatment': 2.0,
        'ayush benefit': 2.0,
        'premium amount': 1.8,
        'premium cost': 1.8,
        'coverage limit': 1.8,
        'coverage amount': 1.8,
        'deductible amount': 1.8,
        'copay amount': 1.8,
        'claim amount': 1.8,
        'claim limit': 1.8,
        'policy term': 1.5,
        'policy period': 1.5,
        'benefit amount': 1.5,
        'benefit limit': 1.5,
        'waiting': 1.3,
        'grace': 1.3,
        'maternity': 1.3,
        'cataract': 1.3,
        'discount': 1.3,
        'ayush': 1.3,
        'premium': 1.2,
        'coverage': 1.2,
        'deductible': 1.2,
        'copay': 1.2,
        'claim': 1.2,
        'policy': 1.1,
        'benefit': 1.1
    }

    # Apply domain boosting
    domain_boost = 1.0
    for term, boost in insurance_terms.items():
        if term in q_lower and term in c_lower:
            domain_boost *= boost
            break  # Apply only the first matching boost

    # Numerical information bonus
    numerical_bonus = 1.0
    if re.search(r'\d+\s*(days|months|years|%|\$|rupees|rs)', c_lower):
        numerical_bonus = 1.4

    # Exact phrase matching bonus
    phrase_bonus = 1.0
    q_phrases = re.findall(r'\b\w+\s+\w+(?:\s+\w+)?\b', q_lower)
    for phrase in q_phrases:
        if len(phrase.split()) >= 2 and phrase in c_lower:
            phrase_bonus *= 1.5

    # Question type specific boosting
    question_type_bonus = 1.0
    if any(word in q_lower for word in ['what', 'how much', 'how many', 'when', 'where']):
        if any(word in c_lower for word in ['amount', 'cost', 'price', 'fee', 'charge', 'rate', 'percentage']):
            question_type_bonus = 1.3

    # Final score calculation
    final_score = base_score * domain_boost * numerical_bonus * phrase_bonus * question_type_bonus
    
    return min(final_score, 5.0)  # Cap the score

def enhanced_chunk_retrieval(question: str, chunks: List[str], top_k: int = 6) -> List[str]:
    """Enhanced keyword-based retrieval with advanced scoring"""
    if not chunks:
        return []
    
    def score_chunk(chunk):
        return advanced_keyword_scoring(question, chunk)

    # Score chunks in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        scores = list(executor.map(score_chunk, chunks))

    # Get top chunks with minimum threshold
    scored_chunks = [(score, chunk) for score, chunk in zip(scores, chunks) if score > 0.05]
    scored_chunks.sort(reverse=True, key=lambda x: x[0])

    # Return top chunks
    top_chunks = [chunk for _, chunk in scored_chunks[:top_k]]
    
    # If we don't have enough high-scoring chunks, add some medium-scoring ones
    if len(top_chunks) < 3:
        medium_chunks = [(score, chunk) for score, chunk in zip(scores, chunks) if 0.01 <= score <= 0.05]
        medium_chunks.sort(reverse=True, key=lambda x: x[0])
        additional_chunks = [chunk for _, chunk in medium_chunks[:3-len(top_chunks)]]
        top_chunks.extend(additional_chunks)

    return top_chunks

def create_gemini_prompt(questions: List[str], context_map: Dict[str, List[str]]) -> str:
    """Create optimized prompt for Gemini API"""
    question_contexts = []

    for i, question in enumerate(questions, 1):
        # Use top 4 chunks for better context
        relevant_chunks = context_map.get(question, [])[:4]
        if relevant_chunks:
            context = "\n---\n".join(relevant_chunks)
            question_contexts.append(f"Question {i}: {question}\nRelevant Context:\n{context}\n")
        else:
            question_contexts.append(f"Question {i}: {question}\nRelevant Context:\nNo relevant context found.\n")

    # Optimized prompt for Gemini
    prompt = f"""You are an expert insurance analyst. Answer each question accurately using ONLY the provided context.

CRITICAL INSTRUCTIONS:
- Use specific details, numbers, percentages, and timeframes from the context
- If information is not in the context, respond "Information not available in provided context"
- Be precise and factual
- Do not make assumptions or add external knowledge
- Focus on extracting exact values and specific details

REQUIRED FORMAT: A1: [detailed answer] A2: [detailed answer] A3: [detailed answer]...

{chr(10).join(question_contexts)}

Remember: Answer in the exact format A1:, A2:, A3:, etc. Use only the provided context. Be specific with numbers and details."""

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
            "temperature": 0.1,
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
        "message": "Insurance Claims Processing API - Gemini Powered (No ML Dependencies)",
        "version": "4.2.0",
        "model": "Gemini 2.0 Flash",
        "target_accuracy": "65%+",
        "target_response_time": "<30 seconds",
        "improvements": [
            "Advanced keyword-based retrieval",
            "TF-IDF-like scoring",
            "Enhanced domain knowledge",
            "Better chunking strategy",
            "Optimized prompt engineering",
            "No ML dependencies for Vercel compatibility"
        ]
    }

@app.post("/api/v1/hackrx/run")
async def document_qa_gemini(
    req: QARequest,
    token: str = Depends(verify_bearer_token)
):
    """Gemini-powered Document Q&A optimized for accuracy without ML dependencies"""
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

        # Step 3: Enhanced keyword-based retrieval
        async def find_chunks_for_question(question):
            return enhanced_chunk_retrieval(question, all_chunks, top_k=6)

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
    print("🤖 Starting Gemini-Powered Insurance API (No ML Dependencies)...")
    print("📈 Target: 65%+ accuracy in <30 seconds")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )