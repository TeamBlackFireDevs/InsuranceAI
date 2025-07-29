"""
InsuranceAI - Fast, accurate policy QA (Vercel-optimized)
--------------------------------------------------------

Lightweight version without sentence-transformers:
1. 100% async I/O with httpx
2. PyMuPDF text extraction (10x faster)
3. Simple keyword-based chunk ranking
4. Single batched LLM call
5. Token bucket rate limiting
6. Memory-efficient for Vercel free tier
"""

import os
import re
import time
import asyncio
from typing import List, Dict, Union

import fitz  # PyMuPDF
import httpx
import uvicorn
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, validator
from dotenv import load_dotenv

# --------------------------------------------------------------------------- #
#                              FastAPI scaffolding                            #
# --------------------------------------------------------------------------- #

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN") or ""

app = FastAPI(
    title="InsuranceAI – Lightning QA",
    description="Fast insurance policy Q&A for Vercel",
    version="2.0.0",
)

security = HTTPBearer()

class QARequest(BaseModel):
    documents: Union[str, List[str]]
    questions: List[str]

    @validator("documents", pre=True)
    def _ensure_list(cls, v):
        return [v] if isinstance(v, str) else v

# --------------------------------------------------------------------------- #
#                           Simple rate limiter                               #
# --------------------------------------------------------------------------- #

class AsyncTokenBucket:
    def __init__(self, rate: int, per_seconds: int = 60):
        self.capacity = rate
        self.tokens = rate
        self.per_seconds = per_seconds
        self.timestamp = time.perf_counter()
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            now = time.perf_counter()
            elapsed = now - self.timestamp
            self.timestamp = now
            self.tokens = min(self.capacity, self.tokens + elapsed * (self.capacity / self.per_seconds))
            if self.tokens < 1:
                sleep_for = (1 - self.tokens) * (self.per_seconds / self.capacity)
                await asyncio.sleep(sleep_for)
                self.tokens += 1
            else:
                self.tokens -= 1

bucket = AsyncTokenBucket(rate=30, per_seconds=60)

# --------------------------------------------------------------------------- #
#                       PDF downloading & text extraction                     #
# --------------------------------------------------------------------------- #

async def extract_pdf_text(url: str) -> str:
    """Download PDF and extract text with PyMuPDF."""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(url, follow_redirects=True)
            r.raise_for_status()
        
        with fitz.open(stream=r.content, filetype="pdf") as doc:
            pages = [page.get_text() for page in doc]
        return "\f".join(pages)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF extraction failed: {e}")

# --------------------------------------------------------------------------- #
#                    Lightweight keyword-based chunking                       #
# --------------------------------------------------------------------------- #

def chunk_text(text: str, size: int = 1200, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks."""
    if len(text) <= size:
        return [text]
    
    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        # Try to break at sentence or paragraph
        slice_ = text[start:end]
        for delim in (".\n", ". ", "\n\n", "\n"):
            idx = slice_.rfind(delim)
            if idx > size * 0.6:  # Don't break too early
                end = start + idx + len(delim)
                break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = max(end - overlap, end)
    return chunks

def score_chunk_relevance(question: str, chunk: str) -> float:
    """Simple keyword-based relevance scoring."""
    q_words = set(re.findall(r'\b\w+\b', question.lower()))
    c_words = set(re.findall(r'\b\w+\b', chunk.lower()))
    
    if not q_words:
        return 0.0
    
    # Exact matches get high score
    exact_matches = len(q_words & c_words)
    
    # Partial matches (stemming-like)
    partial_matches = 0
    for qw in q_words:
        if len(qw) > 4:  # Only for longer words
            for cw in c_words:
                if qw.startswith(cw[:4]) or cw.startswith(qw[:4]):
                    partial_matches += 0.5
                    break
    
    return (exact_matches + partial_matches) / len(q_words)

def top_k_chunks(question: str, chunks: List[str], k: int = 5) -> List[str]:
    """Get top-k most relevant chunks using keyword scoring."""
    if not chunks:
        return []
    
    scored = [(score_chunk_relevance(question, chunk), chunk) for chunk in chunks]
    scored.sort(key=lambda x: x[0], reverse=True)
    
    return [chunk for _, chunk in scored[:k]]

# --------------------------------------------------------------------------- #
#                         Hugging Face chat completion                        #
# --------------------------------------------------------------------------- #

HF_CHAT_URL = "https://api.endpoints.huggingface.cloud/v1/chat/completions"
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

async def call_hf_chat(messages: List[Dict], max_tokens: int = 800) -> str:
    """Call HF chat completion API with rate limiting."""
    if not HF_TOKEN:
        raise HTTPException(status_code=500, detail="HF_TOKEN env var missing")

    await bucket.acquire()
    
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 0.1,
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(HF_CHAT_URL, headers=headers, json=payload)
    
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=f"HF API error: {r.text}")
    
    data = r.json()
    return data["choices"][0]["message"]["content"]

# --------------------------------------------------------------------------- #
#                               Prompt building                               #
# --------------------------------------------------------------------------- #

def build_batch_prompt(questions: List[str], context_map: Dict[str, List[str]]) -> List[Dict]:
    """Build single prompt with all questions and their contexts."""
    segments = []
    for idx, question in enumerate(questions, 1):
        context = "\n".join(context_map[question][:3])  # Use top 3 chunks
        segments.append(f"Q{idx}: {question}\nContext:\n{context}\n")
    
    user_prompt = (
        "You are a professional insurance policy analyst. Answer each question "
        "using ONLY the information provided in its context. If the context "
        "doesn't contain the answer, respond exactly: "
        "\"The provided context does not contain this information.\"\n\n"
        "Format your response as:\nA1: [answer to Q1]\nA2: [answer to Q2]\n"
        "and so on for all questions.\n\n" +
        "\n---\n".join(segments)
    )
    
    return [
        {"role": "system", "content": "You are an expert insurance policy analyst."},
        {"role": "user", "content": user_prompt}
    ]

def parse_numbered_answers(response: str, expected_count: int) -> List[str]:
    """Extract numbered answers from LLM response."""
    pattern = r"A(\d+)[:\-\.]?\s*(.+?)(?=\nA\d+[:\-\.]|\Z)"
    matches = re.findall(pattern, response, flags=re.DOTALL | re.IGNORECASE)
    
    # Initialize answers array
    answers = ["The provided context does not contain this information."] * expected_count
    
    # Fill in found answers
    for num_str, answer in matches:
        try:
            idx = int(num_str) - 1
            if 0 <= idx < expected_count:
                answers[idx] = answer.strip()
        except ValueError:
            continue
    
    return answers

# --------------------------------------------------------------------------- #
#                                 Security                                    #
# --------------------------------------------------------------------------- #

def auth(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    token = credentials.credentials
    if len(token) < 10:
        raise HTTPException(status_code=401, detail="Invalid bearer token")
    return token

# --------------------------------------------------------------------------- #
#                                    Routes                                   #
# --------------------------------------------------------------------------- #

@app.post("/api/v1/hackrx/run")
async def document_qa(req: QARequest, _: str = Depends(auth)):
    """Main endpoint for document Q&A."""
    start_time = time.perf_counter()
    
    try:
        # 1. Extract text from all PDFs concurrently
        texts = await asyncio.gather(*[extract_pdf_text(url) for url in req.documents])
        
        # 2. Chunk all text
        all_chunks = []
        for text in texts:
            all_chunks.extend(chunk_text(text))
        
        # 3. Find relevant chunks for each question
        context_map = {}
        for question in req.questions:
            context_map[question] = top_k_chunks(question, all_chunks, k=4)
        
        # 4. Build single prompt and get response
        messages = build_batch_prompt(req.questions, context_map)
        raw_response = await call_hf_chat(messages)
        
        # 5. Parse answers
        answers = parse_numbered_answers(raw_response, len(req.questions))
        
        elapsed = time.perf_counter() - start_time
        print(f"✅ Completed in {elapsed:.2f}s")
        
        return {"answers": answers}
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {
        "service": "InsuranceAI",
        "version": "2.0.0",
        "status": "optimized for Vercel free tier",
        "model": MODEL_NAME,
        "hf_token_configured": bool(HF_TOKEN)
    }

# --------------------------------------------------------------------------- #
#                                 Dev server                                  #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    if not HF_TOKEN:
        print("⚠️  Set HF_TOKEN environment variable")
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
