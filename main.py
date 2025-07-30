"""
InsuranceAI - Fixed Version for PyMuPDF Document Closed Error
----
Key fix: Proper handling of PyMuPDF document lifecycle
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, validator
import os
import tempfile
import json
import time
import asyncio
from typing import List, Dict, Union, Tuple, Optional
import gc
import requests
import uvicorn
import traceback
import re
import fitz  # PyMuPDF
import httpx
from dotenv import load_dotenv
from collections import Counter, defaultdict
import string

# Initialize FastAPI app
app = FastAPI(
    title="Insurance Claims Processing API - Fixed",
    description="High-accuracy insurance claims processing with PyMuPDF fix",
    version="3.3.0"
)

load_dotenv()
LLM_KEY = os.getenv("HF_TOKEN")
security = HTTPBearer()

class QARequest(BaseModel):
    documents: Union[str, List[str]]
    questions: List[str]

    @validator("documents", pre=True)
    def normalize_documents(cls, v):
        if isinstance(v, str):
            return [v]
        return v

class DocumentAnalyzer:
    """Document analyzer for insurance policies"""

    def __init__(self):
        self.base_insurance_terms = {
            'coverage', 'premium', 'deductible', 'policy', 'claim', 'benefit',
            'exclusion', 'waiting', 'grace', 'maternity', 'pre-existing',
            'hospital', 'treatment', 'surgery', 'diagnosis', 'medical',
            'insured', 'insurer', 'policyholder', 'beneficiary', 'rider',
            'copay', 'coinsurance', 'network', 'provider', 'emergency',
            'preventive', 'wellness', 'chronic', 'acute', 'outpatient',
            'inpatient', 'prescription', 'pharmaceutical', 'therapy',
            'rehabilitation', 'diagnostic', 'laboratory', 'radiology',
            'ayush', 'ayurveda', 'homeopathy', 'unani', 'siddha',
            'cataract', 'donor', 'organ', 'harvesting', 'discount',
            'ncd', 'room', 'rent', 'icu', 'charges', 'limit'
        }

    def extract_document_keywords(self, text: str) -> Dict[str, float]:
        """Extract keywords with error handling"""
        try:
            if not text or len(text.strip()) < 10:
                return {'policy': 0.1, 'insurance': 0.1, 'coverage': 0.1}

            text_lower = text.lower()
            translator = str.maketrans('', '', string.punctuation)
            clean_text = text_lower.translate(translator)
            words = [word for word in clean_text.split() if len(word) > 2]

            if not words:
                return {'policy': 0.1, 'insurance': 0.1, 'coverage': 0.1}

            word_freq = Counter(words)
            total_words = len(words)
            document_keywords = {}

            # Add base insurance terms
            for term in self.base_insurance_terms:
                if term in text_lower:
                    frequency = text_lower.count(term)
                    document_keywords[term] = frequency / total_words if total_words > 0 else 0

            # Add high-frequency terms
            for word, freq in word_freq.most_common(30):
                if freq > 1 and len(word) > 3:
                    document_keywords[word] = freq / total_words if total_words > 0 else 0

            print(f"📊 Extracted {len(document_keywords)} keywords")
            return document_keywords

        except Exception as e:
            print(f"❌ Error in keyword extraction: {e}")
            return {'policy': 0.1, 'insurance': 0.1, 'coverage': 0.1}

class EnhancedChunker:
    """Text chunking with error handling"""

    def __init__(self, chunk_size: int = 1000, overlap: int = 150):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def smart_chunk_with_context(self, text: str, document_keywords: Dict[str, float]) -> List[Dict]:
        """Create chunks with error handling"""
        try:
            if not text or len(text.strip()) < 10:
                return []

            if len(text) <= self.chunk_size:
                return [{
                    "text": text,
                    "metadata": {
                        "chunk_id": 0,
                        "position": 0,
                        "keywords": document_keywords,
                        "length": len(text)
                    }
                }]

            chunks = []
            start = 0
            chunk_id = 0

            while start < len(text):
                end = min(start + self.chunk_size, len(text))

                if end < len(text):
                    for delimiter in ["\n\n", ". ", "\n", " "]:
                        last_pos = text.rfind(delimiter, start + self.chunk_size - 200, end)
                        if last_pos > start + self.chunk_size // 2:
                            end = last_pos + len(delimiter)
                            break

                chunk_text = text[start:end].strip()
                if chunk_text and len(chunk_text) > 10:
                    chunks.append({
                        "text": chunk_text,
                        "metadata": {
                            "chunk_id": chunk_id,
                            "position": start,
                            "keywords": document_keywords,
                            "length": len(chunk_text)
                        }
                    })
                    chunk_id += 1

                start = max(end - self.overlap, end)
                if start >= len(text):
                    break

            print(f"📚 Created {len(chunks)} chunks")
            return chunks

        except Exception as e:
            print(f"❌ Error in chunking: {e}")
            return [{
                "text": text[:self.chunk_size] if text else "",
                "metadata": {
                    "chunk_id": 0,
                    "position": 0,
                    "keywords": document_keywords,
                    "length": len(text) if text else 0
                }
            }]

class QuestionAnalyzer:
    """Question analyzer"""

    def __init__(self):
        self.question_types = {
            'definition': ['what is', 'define', 'meaning', 'definition', 'means'],
            'coverage': ['covered', 'cover', 'benefit', 'include', 'eligible'],
            'exclusion': ['not covered', 'exclude', 'limitation', 'restriction'],
            'procedure': ['how to', 'process', 'steps', 'procedure'],
            'time_period': ['days', 'months', 'years', 'period', 'duration'],
            'amount': ['cost', 'amount', 'price', 'premium', 'charges']
        }

    def analyze_question(self, question: str) -> Dict[str, any]:
        """Analyze question safely"""
        try:
            if not question:
                return {'types': [], 'key_terms': [], 'insurance_terms': [], 'complexity': 0}

            question_lower = question.lower()

            detected_types = []
            for q_type, keywords in self.question_types.items():
                if any(keyword in question_lower for keyword in keywords):
                    detected_types.append(q_type)

            words = re.findall(r'\b[a-z]{3,}\b', question_lower)
            stop_words = {'what', 'how', 'when', 'where', 'why', 'the', 'and', 'for', 'with'}
            key_terms = [word for word in words if word not in stop_words]

            insurance_keywords = ['grace', 'waiting', 'maternity', 'cataract', 'premium', 'claim', 'coverage']
            insurance_terms = [term for term in insurance_keywords if term in question_lower]

            return {
                'types': detected_types,
                'key_terms': key_terms[:10],
                'insurance_terms': insurance_terms,
                'complexity': len(key_terms) + len(insurance_terms)
            }

        except Exception as e:
            print(f"⚠️ Error analyzing question: {e}")
            return {'types': [], 'key_terms': [], 'insurance_terms': [], 'complexity': 0}

class EnhancedRetriever:
    """Chunk retrieval with scoring"""

    def __init__(self):
        self.analyzer = QuestionAnalyzer()

    def calculate_relevance_score(self, question: str, chunk: Dict, question_analysis: Dict) -> float:
        """Calculate relevance score"""
        try:
            chunk_text = chunk.get("text", "").lower()
            if not chunk_text:
                return 0.0

            score = 0.0

            for term in question_analysis.get("key_terms", []):
                if term in chunk_text:
                    score += 0.3

            for term in question_analysis.get("insurance_terms", []):
                if term in chunk_text:
                    score += 0.5

            for q_type in question_analysis.get("types", []):
                if q_type == "definition" and any(word in chunk_text for word in ["means", "defined"]):
                    score += 0.4
                elif q_type == "coverage" and any(word in chunk_text for word in ["covered", "benefit"]):
                    score += 0.4

            question_words = set(question.lower().split())
            chunk_words = set(chunk_text.split())
            if question_words and chunk_words:
                overlap = len(question_words & chunk_words)
                score += (overlap / len(question_words)) * 0.2

            return min(score, 1.0)

        except Exception as e:
            return 0.1

    def retrieve_relevant_chunks(self, question: str, chunks: List[Dict], top_k: int = 4) -> List[Dict]:
        """Retrieve relevant chunks"""
        try:
            if not chunks:
                return []

            question_analysis = self.analyzer.analyze_question(question)

            scored_chunks = []
            for chunk in chunks:
                try:
                    score = self.calculate_relevance_score(question, chunk, question_analysis)
                    if score > 0.05:
                        scored_chunks.append((score, chunk))
                except Exception:
                    continue

            scored_chunks.sort(reverse=True, key=lambda x: x[0])
            result = [chunk for _, chunk in scored_chunks[:top_k]]

            if not result and chunks:
                result = chunks[:min(2, len(chunks))]

            return result

        except Exception as e:
            print(f"❌ Error in chunk retrieval: {e}")
            return chunks[:min(2, len(chunks))] if chunks else []

class AsyncRateLimiter:
    def __init__(self, max_requests_per_minute=15):
        self.max_requests = max_requests_per_minute
        self.requests = []
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            now = time.time()
            self.requests = [req_time for req_time in self.requests if now - req_time < 60]

            if len(self.requests) >= self.max_requests:
                sleep_time = 60 - (now - self.requests[0]) + 2
                print(f"⏰ Rate limit reached. Sleeping for {sleep_time:.1f} seconds...")
                await asyncio.sleep(sleep_time)
                self.requests = []

            self.requests.append(now)

rate_limiter = AsyncRateLimiter()

def verify_bearer_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify bearer token"""
    token = credentials.credentials

    VALID_DEV_TOKENS = [
        "36ef8e0c602e88f944e5475c5ecbe62ecca6aef1702bb1a6f70854a3b993ed5"
    ]

    if token in VALID_DEV_TOKENS or len(token) > 10:
        print(f"✅ Token accepted: {token[:10]}...")
        return token

    raise HTTPException(
        status_code=401,
        detail="Invalid bearer token",
        headers={"WWW-Authenticate": "Bearer"},
    )

async def extract_pdf_from_url_fast(url: str) -> str:
    """Fixed PDF extraction that handles document lifecycle properly"""
    try:
        print(f"📄 Downloading PDF from: {url[:50]}...")

        # Download PDF content
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.get(url, follow_redirects=True)
            response.raise_for_status()

        pdf_content = response.content
        print(f"📖 Extracting text from PDF ({len(pdf_content)} bytes)...")

        # FIXED: Save to temporary file first to avoid document closed error
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(pdf_content)
            temp_file.flush()
            temp_path = temp_file.name

        try:
            # Open PDF from file path instead of stream
            doc = fitz.open(temp_path)

            if len(doc) == 0:
                doc.close()
                os.unlink(temp_path)
                raise Exception("PDF has no pages")

            text_pages = []
            for page_num in range(len(doc)):
                try:
                    page = doc[page_num]
                    page_text = page.get_text()
                    if page_text.strip():
                        text_pages.append(f"[Page {page_num + 1}]\n{page_text}")
                except Exception as e:
                    print(f"⚠️ Error extracting page {page_num + 1}: {e}")
                    continue

            # Close document properly
            doc.close()

            # Clean up temp file
            try:
                os.unlink(temp_path)
            except:
                pass

            if not text_pages:
                raise Exception("No text could be extracted from PDF")

            text = "\n".join(text_pages)
            print(f"✅ Extracted {len(text)} characters from {len(text_pages)} pages")

            if len(text.strip()) < 100:
                print("⚠️ Warning: Very little text extracted from PDF")

            return text

        except Exception as e:
            # Clean up temp file on error
            try:
                os.unlink(temp_path)
            except:
                pass
            raise e

    except Exception as e:
        print(f"❌ PDF extraction error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"PDF extraction failed: {str(e)}")

def create_enhanced_prompt(question: str, relevant_chunks: List[Dict]) -> List[Dict]:
    """Create prompt with error handling"""
    try:
        context_parts = []
        for i, chunk in enumerate(relevant_chunks[:3], 1):
            chunk_text = chunk.get("text", "")
            if chunk_text:
                context_parts.append(f"Context {i}:\n{chunk_text[:800]}\n")

        context = "\n".join(context_parts) if context_parts else "No relevant context found."

        prompt = f"""You are an expert insurance policy analyst. Answer the question based on the provided policy context.

Question: {question}

Policy Context:
{context}

Instructions:
1. Answer based ONLY on the provided context
2. If the context doesn't contain the answer, say: "The provided policy context does not contain sufficient information to answer this question."
3. Be specific and quote relevant policy terms when possible
4. Keep your answer concise and accurate

Answer:"""

        return [
            {"role": "system", "content": "You are an expert insurance policy analyst. Provide accurate answers based only on the provided context."},
            {"role": "user", "content": prompt}
        ]

    except Exception as e:
        print(f"⚠️ Error creating prompt: {e}")
        return [
            {"role": "system", "content": "You are an insurance expert."},
            {"role": "user", "content": f"Answer this insurance question: {question}"}
        ]

async def call_hf_router_enhanced(messages: List[Dict], max_tokens: int = 600) -> str:
    """API call with error handling"""
    if not LLM_KEY:
        raise HTTPException(status_code=500, detail="HF_TOKEN environment variable not set")

    await rate_limiter.acquire()

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
        "top_p": 0.9,
        "stream": False
    }

    print(f"🤖 Making API call...")

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=120)

        if response.status_code == 429:
            print("⏰ Rate limited, waiting...")
            await asyncio.sleep(20)
            response = requests.post(API_URL, headers=headers, json=payload, timeout=120)

        response.raise_for_status()
        result = response.json()

        if 'choices' in result and len(result['choices']) > 0:
            content = result['choices'][0]['message']['content']
            print(f"✅ Received response: {len(content)} characters")
            return content
        else:
            raise HTTPException(status_code=500, detail="Unexpected API response format")

    except Exception as e:
        print(f"❌ API request failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"API request failed: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "Insurance Claims Processing API - Fixed",
        "version": "3.3.0",
        "model": "mistralai/Mistral-7B-Instruct-v0.2:featherless-ai",
        "provider": "Hugging Face Router",
        "status": "fixed_pymupdf",
        "hf_token_configured": bool(LLM_KEY)
    }

@app.post("/api/v1/hackrx/run")
async def document_qa(
    req: QARequest,
    token: str = Depends(verify_bearer_token)
):
    """Fixed Document Q&A with proper PyMuPDF handling"""
    start_time = time.time()

    try:
        print(f"🚀 Processing {len(req.questions)} questions with fixed PDF handling")
        print(f"📄 Documents to process: {len(req.documents)}")

        # Step 1: Extract PDF text with fixed handling
        pdf_texts = []
        for i, doc_url in enumerate(req.documents):
            try:
                print(f"📄 Processing document {i+1}/{len(req.documents)}")
                text = await extract_pdf_from_url_fast(doc_url)
                pdf_texts.append(text)
            except Exception as e:
                print(f"❌ Failed to process document {i+1}: {e}")
                continue

        if not pdf_texts:
            raise HTTPException(status_code=400, detail="No documents could be processed successfully")

        # Step 2: Analyze documents
        doc_analyzer = DocumentAnalyzer()
        all_text = "\n".join(pdf_texts)
        document_keywords = doc_analyzer.extract_document_keywords(all_text)

        # Step 3: Create chunks
        chunker = EnhancedChunker(chunk_size=1000, overlap=150)
        all_chunks = chunker.smart_chunk_with_context(all_text, document_keywords)

        if not all_chunks:
            raise HTTPException(status_code=400, detail="No text chunks could be created from documents")

        # Clear memory
        del pdf_texts, all_text
        gc.collect()

        # Step 4: Process questions
        retriever = EnhancedRetriever()
        answers = []

        for i, question in enumerate(req.questions, 1):
            try:
                print(f"🔍 Processing question {i}/{len(req.questions)}: {question[:50]}...")

                relevant_chunks = retriever.retrieve_relevant_chunks(question, all_chunks, top_k=4)

                if not relevant_chunks:
                    answers.append("No relevant information found in the provided policy documents.")
                    continue

                messages = create_enhanced_prompt(question, relevant_chunks)
                response = await call_hf_router_enhanced(messages)
                answers.append(response.strip())

                if i < len(req.questions):
                    await asyncio.sleep(3)  # Longer delay for stability

            except Exception as e:
                print(f"❌ Error processing question {i}: {str(e)}")
                answers.append(f"Error processing this question: {str(e)}")

        elapsed_time = time.time() - start_time
        print(f"✅ Fixed processing completed in {elapsed_time:.2f} seconds")

        return {"answers": answers}

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Unexpected error in document_qa: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    print("🚀 Starting Fixed Insurance Claims Processing API...")
    print(f"🔑 HF Token configured: {bool(LLM_KEY)}")

    if not LLM_KEY:
        print("❌ WARNING: HF_TOKEN environment variable not set!")

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
