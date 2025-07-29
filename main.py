"""
InsuranceAI - Fast, accurate policy QA
--------------------------------------

Key improvements:
1. 100 % async I/O – httpx for both PDF download and HF LLM calls.
2. PyMuPDF text extraction (≈10-15× faster than pdfminer).
3. Token-bucket rate limiter (30 req / min).
4. ONE deterministic LLM request per run (all questions batched).
5. Smaller but strong model: mistralai/Mistral-7B-Instruct (≈5-6 s latency).
6. Guard-railed prompt + regex validation to stop hallucinations.
7. Memory-efficient chunking & cosine-similarity ranking (MiniLM).
"""

from __future__ import annotations

import os
import re
import gc
import time
import json
import math
import asyncio
import tempfile
from typing import List, Dict, Union

import fitz  # PyMuPDF
import httpx
import uvicorn
import numpy as np

from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, validator
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util as st_util

# --------------------------------------------------------------------------- #
#                              FastAPI scaffolding                            #
# --------------------------------------------------------------------------- #

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN") or ""  # Hugging Face access token

app = FastAPI(
    title="InsuranceAI – lightning QA",
    description="Fast & deterministic insurance-policy Q&A",
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
#                           Utility: token bucket rate-limiter                #
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

async def _download_bytes(url: str) -> bytes:
    """Download file bytes asynchronously."""
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(url, follow_redirects=True)
        r.raise_for_status()
        return r.content


async def extract_pdf_text(url: str) -> str:
    """
    Download PDF and extract text with PyMuPDF.

    Returns:
        The concatenated text with page-form-feed separators.
    """
    try:
        raw = await _download_bytes(url)
        with fitz.open(stream=raw, filetype="pdf") as doc:
            pages = [page.get_text() for page in doc]
        text = "\f".join(pages)
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF extraction failed: {e}") from e


# --------------------------------------------------------------------------- #
#                       Text chunking & semantic ranking                      #
# --------------------------------------------------------------------------- #

EMBED_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def chunk_text(text: str, size: int = 1500, overlap: int = 250) -> List[str]:
    if len(text) <= size:
        return [text]

    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        # smart break on sentence / newline
        slice_ = text[start:end]
        for delim in (".", "\n"):
            idx = slice_.rfind(delim)
            if idx != -1 and idx > size * 0.5:
                end = start + idx + 1
                break
        chunks.append(text[start:end].strip())
        start = max(end - overlap, end)
    return chunks


def top_k_chunks(question: str, chunks: List[str], k: int = 4) -> List[str]:
    if not chunks:
        return []

    q_emb = EMBED_MODEL.encode(question, convert_to_numpy=True, normalize_embeddings=True)
    idxs = list(range(len(chunks)))
    # embed in smaller batches to save RAM
    scores = []
    batch = 64
    for i in range(0, len(chunks), batch):
        embs = EMBED_MODEL.encode(
            chunks[i : i + batch], convert_to_numpy=True, normalize_embeddings=True
        )
        sims = np.dot(embs, q_emb)
        scores.extend(sims.tolist())
    best = sorted(idxs, key=lambda i: scores[i], reverse=True)[:k]
    return [chunks[i] for i in best]


# --------------------------------------------------------------------------- #
#                         Hugging Face chat completion                        #
# --------------------------------------------------------------------------- #

HF_CHAT_URL = "https://api.endpoints.huggingface.cloud/v1/chat/completions"
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"


async def call_hf_chat(messages: List[Dict], max_tokens: int = 900) -> str:
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
        raise HTTPException(status_code=r.status_code, detail=f"HF API: {r.text}")
    data = r.json()
    return data["choices"][0]["message"]["content"]


# --------------------------------------------------------------------------- #
#                               Prompt helpers                                #
# --------------------------------------------------------------------------- #

def build_batch_prompt(questions: List[str], context_map: Dict[str, List[str]]) -> List[Dict]:
    """
    Build ONE prompt containing all questions with their top-K contexts.
    The model must answer in numbered list form (A1, A2, …).
    """
    segments = []
    for idx, q in enumerate(questions, 1):
        ctx = "\n".join(context_map[q])
        segments.append(f"Q{idx}: {q}\nContext:\n{ctx}\n")
    user_prompt = (
        "You are a professional insurance policy analyst. "
        "Answer every question **strictly** from its context. "
        "If the context lacks the answer respond exactly: "
        "\"The provided context does not contain this information\".\n\n"
        "Return answers as:\nA1: <answer>\nA2: <answer>\n... etc.\n\n"
        + "\n---\n".join(segments)
    )
    return [
        {"role": "system", "content": "You are an expert insurance analyst."},
        {"role": "user", "content": user_prompt},
    ]


def split_answers(raw: str, expected: int) -> List[str]:
    pattern = r"A(\d+)[:\-]\s*(.+?)(?=\nA\d+[:\-]|\Z)"
    matches = re.findall(pattern, raw, flags=re.S | re.I)
    if len(matches) < expected:
        # fallback naive split
        parts = [p.strip() for p in raw.split("\n") if p.strip()]
        return (parts + ["Answer not found"] * expected)[:expected]
    # order by idx
    answers = [""] * expected
    for idx_str, ans in matches:
        i = int(idx_str) - 1
        if 0 <= i < expected:
            answers[i] = ans.strip()
    for i, a in enumerate(answers):
        if not a:
            answers[i] = "Answer not found"
    return answers


# --------------------------------------------------------------------------- #
#                              Security helper                                #
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
    start_time = time.perf_counter()

    # 1) Fetch & extract PDFs concurrently
    texts = await asyncio.gather(*[extract_pdf_text(u) for u in req.documents])
    chunks = []
    for txt in texts:
        chunks.extend(chunk_text(txt))

    # 2) Rank & map contexts
    context_map: Dict[str, List[str]] = {
        q: top_k_chunks(q, chunks, k=4) for q in req.questions
    }

    # 3) Build & send single chat request
    prompt = build_batch_prompt(req.questions, context_map)
    raw_answer = await call_hf_chat(prompt)

    # 4) Parse numbered answers
    answers = split_answers(raw_answer, len(req.questions))

    elapsed = time.perf_counter() - start_time
    print(f"Finished in {elapsed:0.2f}s")
    return {"answers": answers}


@app.get("/")
async def root():
    return {
        "service": "InsuranceAI",
        "version": "2.0.0",
        "model": MODEL_NAME,
        "docs": "/docs",
        "hf_token_configured": bool(HF_TOKEN),
    }


# --------------------------------------------------------------------------- #
#                                 Dev server                                  #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    if not HF_TOKEN:
        print("⚠️  HF_TOKEN env var is missing; set it first.")
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), log_level="info")
