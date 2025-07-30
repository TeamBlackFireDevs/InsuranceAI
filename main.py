"""
InsuranceAI - Improved Gemini Version with Better Retrieval
----
Enhanced chunk retrieval and keyword matching for better accuracy
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
    title="Insurance Claims Processing API - Improved Gemini",
    description="Enhanced insurance claims processing with improved retrieval",
    version="4.1.0"
)

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
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
    """Enhanced document analyzer for insurance policies"""

    def __init__(self):
        # Expanded insurance terms with variations
        self.base_insurance_terms = {
            'coverage', 'premium', 'deductible', 'policy', 'claim', 'benefit',
            'exclusion', 'waiting', 'grace', 'maternity', 'pre-existing', 'ped',
            'hospital', 'treatment', 'surgery', 'diagnosis', 'medical',
            'insured', 'insurer', 'policyholder', 'beneficiary', 'rider',
            'copay', 'coinsurance', 'network', 'provider', 'emergency',
            'preventive', 'wellness', 'chronic', 'acute', 'outpatient',
            'inpatient', 'prescription', 'pharmaceutical', 'therapy',
            'rehabilitation', 'diagnostic', 'laboratory', 'radiology',
            'ayush', 'ayurveda', 'homeopathy', 'unani', 'siddha',
            'cataract', 'donor', 'organ', 'harvesting', 'discount',
            'ncd', 'room', 'rent', 'icu', 'charges', 'limit', 'period',
            'days', 'months', 'years', 'continuous', 'renewal', 'break',
            'floater', 'sum', 'amount', 'payable', 'reimbursed', 'expenses'
        }

        # Specific question-answer mappings for better retrieval
        self.question_keywords = {
            'grace period': ['grace', 'period', 'premium', 'payment', 'days'],
            'waiting period': ['waiting', 'period', 'pre-existing', 'ped', 'diseases', 'months'],
            'maternity': ['maternity', 'pregnancy', 'childbirth', 'delivery', 'expenses'],
            'cataract': ['cataract', 'surgery', 'waiting', 'period', 'eye'],
            'organ donor': ['organ', 'donor', 'medical', 'expenses', 'covered'],
            'ncd': ['ncd', 'no', 'claim', 'discount', 'bonus', 'percentage'],
            'health check': ['health', 'check', 'preventive', 'checkup', 'reimbursed'],
            'hospital': ['hospital', 'definition', 'qualified', 'beds', 'medical'],
            'ayush': ['ayush', 'ayurveda', 'homeopathy', 'unani', 'siddha', 'treatment'],
            'room rent': ['room', 'rent', 'icu', 'charges', 'sub-limit', 'plan']
        }

    def extract_document_keywords(self, text: str) -> Dict[str, float]:
        """Enhanced keyword extraction"""
        try:
            if not text or len(text.strip()) < 10:
                return {'policy': 0.1, 'insurance': 0.1, 'coverage': 0.1}

            text_lower = text.lower()

            # Clean text more thoroughly
            # Remove extra whitespace and normalize
            text_lower = re.sub(r'\s+', ' ', text_lower)

            # Extract words (including hyphenated terms)
            words = re.findall(r'\b[a-z](?:[a-z-]*[a-z])?\b', text_lower)
            words = [word for word in words if len(word) > 2]

            if not words:
                return {'policy': 0.1, 'insurance': 0.1, 'coverage': 0.1}

            word_freq = Counter(words)
            total_words = len(words)
            document_keywords = {}

            # Add base insurance terms with higher weights
            for term in self.base_insurance_terms:
                count = text_lower.count(term)
                if count > 0:
                    document_keywords[term] = (count / total_words) * 2  # Higher weight

            # Add high-frequency domain-specific terms
            for word, freq in word_freq.most_common(50):
                if freq > 2 and len(word) > 3:
                    # Boost insurance-related terms
                    weight = 2 if word in self.base_insurance_terms else 1
                    document_keywords[word] = (freq / total_words) * weight

            print(f"📊 Extracted {len(document_keywords)} keywords")
            return document_keywords

        except Exception as e:
            print(f"❌ Error in keyword extraction: {e}")
            return {'policy': 0.1, 'insurance': 0.1, 'coverage': 0.1}

class EnhancedChunker:
    """Improved text chunking with better boundary detection"""

    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def smart_chunk_with_context(self, text: str, document_keywords: Dict[str, float]) -> List[Dict]:
        """Enhanced chunking with better context preservation"""
        try:
            if not text or len(text.strip()) < 10:
                return []

            # Clean and normalize text
            text = re.sub(r'\n+', '\n', text)  # Normalize line breaks
            text = re.sub(r'\s+', ' ', text)    # Normalize spaces

            if len(text) <= self.chunk_size:
                return [{
                    "text": text,
                    "metadata": {
                        "chunk_id": 0,
                        "position": 0,
                        "keywords": document_keywords,
                        "length": len(text),
                        "keyword_density": self._calculate_keyword_density(text, document_keywords)
                    }
                }]

            chunks = []
            start = 0
            chunk_id = 0

            while start < len(text):
                end = min(start + self.chunk_size, len(text))

                # Better boundary detection
                if end < len(text):
                    # Look for natural breaks in order of preference
                    break_points = [
                        (r'\n\n', 2),      # Paragraph breaks
                        (r'\. [A-Z]', 2),   # Sentence endings
                        (r'\n', 1),         # Line breaks
                        (r', ', 2),          # Comma breaks
                        (r' ', 1)            # Word breaks
                    ]

                    best_break = end
                    for pattern, offset in break_points:
                        matches = list(re.finditer(pattern, text[start + self.chunk_size - 300:end]))
                        if matches:
                            last_match = matches[-1]
                            break_pos = start + self.chunk_size - 300 + last_match.end()
                            if break_pos > start + self.chunk_size // 2:
                                best_break = break_pos
                                break

                    end = best_break

                chunk_text = text[start:end].strip()
                if chunk_text and len(chunk_text) > 20:  # Minimum meaningful chunk size
                    keyword_density = self._calculate_keyword_density(chunk_text, document_keywords)

                    chunks.append({
                        "text": chunk_text,
                        "metadata": {
                            "chunk_id": chunk_id,
                            "position": start,
                            "keywords": document_keywords,
                            "length": len(chunk_text),
                            "keyword_density": keyword_density
                        }
                    })
                    chunk_id += 1

                # Move start position with overlap
                start = max(end - self.overlap, end)
                if start >= len(text):
                    break

            print(f"📚 Created {len(chunks)} enhanced chunks")
            return chunks

        except Exception as e:
            print(f"❌ Error in chunking: {e}")
            return [{
                "text": text[:self.chunk_size] if text else "",
                "metadata": {
                    "chunk_id": 0,
                    "position": 0,
                    "keywords": document_keywords,
                    "length": len(text) if text else 0,
                    "keyword_density": 0
                }
            }]

    def _calculate_keyword_density(self, text: str, keywords: Dict[str, float]) -> float:
        """Calculate keyword density for a chunk"""
        if not text or not keywords:
            return 0.0

        text_lower = text.lower()
        total_score = 0.0

        for keyword, weight in keywords.items():
            if keyword in text_lower:
                count = text_lower.count(keyword)
                total_score += count * weight

        return total_score / len(text.split()) if text.split() else 0.0

class ImprovedQuestionAnalyzer:
    """Enhanced question analyzer with better pattern matching"""

    def __init__(self):
        self.question_patterns = {
            'grace_period': {
                'keywords': ['grace', 'period', 'premium', 'payment'],
                'patterns': [r'grace\s+period', r'premium\s+payment', r'grace\s+time']
            },
            'waiting_period': {
                'keywords': ['waiting', 'period', 'pre-existing', 'ped', 'diseases'],
                'patterns': [r'waiting\s+period', r'pre-existing', r'ped']
            },
            'maternity': {
                'keywords': ['maternity', 'pregnancy', 'childbirth', 'delivery'],
                'patterns': [r'maternity', r'pregnancy', r'childbirth']
            },
            'cataract': {
                'keywords': ['cataract', 'surgery', 'eye', 'waiting'],
                'patterns': [r'cataract', r'eye\s+surgery']
            },
            'organ_donor': {
                'keywords': ['organ', 'donor', 'medical', 'expenses'],
                'patterns': [r'organ\s+donor', r'donor\s+expenses']
            },
            'ncd': {
                'keywords': ['ncd', 'no', 'claim', 'discount', 'bonus'],
                'patterns': [r'no\s+claim\s+discount', r'ncd', r'claim\s+bonus']
            },
            'health_checkup': {
                'keywords': ['health', 'check', 'preventive', 'checkup'],
                'patterns': [r'health\s+check', r'preventive', r'checkup']
            },
            'hospital_definition': {
                'keywords': ['hospital', 'definition', 'define', 'qualified'],
                'patterns': [r'define.*hospital', r'hospital.*definition']
            },
            'ayush': {
                'keywords': ['ayush', 'ayurveda', 'homeopathy', 'unani', 'siddha'],
                'patterns': [r'ayush', r'ayurveda', r'homeopathy']
            },
            'room_rent': {
                'keywords': ['room', 'rent', 'icu', 'charges', 'sub-limit'],
                'patterns': [r'room\s+rent', r'icu\s+charges', r'sub-limit']
            }
        }

    def analyze_question(self, question: str) -> Dict[str, any]:
        """Enhanced question analysis"""
        try:
            if not question:
                return {'types': [], 'key_terms': [], 'insurance_terms': [], 'complexity': 0, 'question_type': None}

            question_lower = question.lower()

            # Detect specific question type
            question_type = None
            max_score = 0

            for q_type, config in self.question_patterns.items():
                score = 0

                # Check keywords
                for keyword in config['keywords']:
                    if keyword in question_lower:
                        score += 1

                # Check patterns
                for pattern in config['patterns']:
                    if re.search(pattern, question_lower):
                        score += 2

                if score > max_score:
                    max_score = score
                    question_type = q_type

            # Extract key terms
            words = re.findall(r'\b[a-z]{3,}\b', question_lower)
            stop_words = {'what', 'how', 'when', 'where', 'why', 'the', 'and', 'for', 'with', 'are', 'does', 'this'}
            key_terms = [word for word in words if word not in stop_words]

            # Insurance-specific terms
            insurance_keywords = [
                'grace', 'waiting', 'maternity', 'cataract', 'premium', 'claim', 
                'coverage', 'ncd', 'discount', 'donor', 'organ', 'ayush', 'hospital',
                'room', 'rent', 'icu', 'preventive', 'health', 'check'
            ]
            insurance_terms = [term for term in insurance_keywords if term in question_lower]

            return {
                'types': [question_type] if question_type else [],
                'key_terms': key_terms[:15],
                'insurance_terms': insurance_terms,
                'complexity': len(key_terms) + len(insurance_terms),
                'question_type': question_type,
                'confidence': max_score
            }

        except Exception as e:
            print(f"⚠️ Error analyzing question: {e}")
            return {'types': [], 'key_terms': [], 'insurance_terms': [], 'complexity': 0, 'question_type': None}

class ImprovedRetriever:
    """Enhanced chunk retrieval with better scoring"""

    def __init__(self):
        self.analyzer = ImprovedQuestionAnalyzer()

    def calculate_relevance_score(self, question: str, chunk: Dict, question_analysis: Dict) -> float:
        """Enhanced relevance scoring"""
        try:
            chunk_text = chunk.get("text", "").lower()
            if not chunk_text:
                return 0.0

            score = 0.0
            question_lower = question.lower()

            # Base keyword matching (improved)
            for term in question_analysis.get("key_terms", []):
                if term in chunk_text:
                    # Exact word boundary matching
                    if re.search(r'\b' + re.escape(term) + r'\b', chunk_text):
                        score += 0.4
                    else:
                        score += 0.2

            # Insurance terms (higher weight)
            for term in question_analysis.get("insurance_terms", []):
                if term in chunk_text:
                    if re.search(r'\b' + re.escape(term) + r'\b', chunk_text):
                        score += 0.6
                    else:
                        score += 0.3

            # Question type specific scoring
            question_type = question_analysis.get("question_type")
            if question_type:
                type_patterns = self.analyzer.question_patterns.get(question_type, {})

                # Check for specific patterns in chunk
                for pattern in type_patterns.get('patterns', []):
                    if re.search(pattern, chunk_text):
                        score += 0.8

                # Check for type-specific keywords
                for keyword in type_patterns.get('keywords', []):
                    if re.search(r'\b' + re.escape(keyword) + r'\b', chunk_text):
                        score += 0.5

            # Phrase matching (exact phrases from question)
            question_phrases = self._extract_phrases(question_lower)
            for phrase in question_phrases:
                if phrase in chunk_text:
                    score += 0.7

            # Keyword density bonus
            keyword_density = chunk.get("metadata", {}).get("keyword_density", 0)
            score += keyword_density * 0.1

            # Penalize very short chunks
            chunk_length = len(chunk_text.split())
            if chunk_length < 20:
                score *= 0.5

            return min(score, 2.0)  # Cap at 2.0

        except Exception as e:
            print(f"⚠️ Error calculating relevance score: {e}")
            return 0.1

    def _extract_phrases(self, text: str) -> List[str]:
        """Extract meaningful phrases from text"""
        phrases = []

        # Extract 2-3 word phrases
        words = text.split()
        for i in range(len(words) - 1):
            if len(words[i]) > 2 and len(words[i+1]) > 2:
                phrases.append(f"{words[i]} {words[i+1]}")

        for i in range(len(words) - 2):
            if all(len(word) > 2 for word in words[i:i+3]):
                phrases.append(f"{words[i]} {words[i+1]} {words[i+2]}")

        return phrases

    def retrieve_relevant_chunks(self, question: str, chunks: List[Dict], top_k: int = 5) -> List[Dict]:
        """Enhanced chunk retrieval"""
        try:
            if not chunks:
                return []

            question_analysis = self.analyzer.analyze_question(question)
            print(f"🔍 Question type detected: {question_analysis.get('question_type', 'unknown')}")
            print(f"🔍 Key terms: {question_analysis.get('key_terms', [])[:5]}")

            scored_chunks = []
            for chunk in chunks:
                try:
                    score = self.calculate_relevance_score(question, chunk, question_analysis)
                    if score > 0.1:  # Lower threshold
                        scored_chunks.append((score, chunk))
                except Exception as e:
                    print(f"⚠️ Error scoring chunk: {e}")
                    continue

            # Sort by score
            scored_chunks.sort(reverse=True, key=lambda x: x[0])

            # Debug: Print top scores
            print(f"🎯 Top chunk scores: {[f'{score:.2f}' for score, _ in scored_chunks[:5]]}")

            result = [chunk for _, chunk in scored_chunks[:top_k]]

            # If no good chunks found, return top chunks anyway
            if not result and chunks:
                result = chunks[:min(3, len(chunks))]
                print("⚠️ Using fallback chunks")

            return result

        except Exception as e:
            print(f"❌ Error in chunk retrieval: {e}")
            return chunks[:min(3, len(chunks))] if chunks else []

class AsyncRateLimiter:
    def __init__(self, max_requests_per_minute=25):
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
    """PDF extraction with proper handling"""
    try:
        print(f"📄 Downloading PDF from: {url[:50]}...")

        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.get(url, follow_redirects=True)
            response.raise_for_status()

        pdf_content = response.content
        print(f"📖 Extracting text from PDF ({len(pdf_content)} bytes)...")

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(pdf_content)
            temp_file.flush()
            temp_path = temp_file.name

        try:
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

            doc.close()

            try:
                os.unlink(temp_path)
            except:
                pass

            if not text_pages:
                raise Exception("No text could be extracted from PDF")

            text = "\n".join(text_pages)
            print(f"✅ Extracted {len(text)} characters from {len(text_pages)} pages")

            return text

        except Exception as e:
            try:
                os.unlink(temp_path)
            except:
                pass
            raise e

    except Exception as e:
        print(f"❌ PDF extraction error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"PDF extraction failed: {str(e)}")

def create_enhanced_prompt(question: str, relevant_chunks: List[Dict]) -> str:
    """Enhanced prompt creation"""
    try:
        context_parts = []
        for i, chunk in enumerate(relevant_chunks[:4], 1):  # Use top 4 chunks
            chunk_text = chunk.get("text", "")
            if chunk_text:
                # Include more context per chunk
                context_parts.append(f"Context {i}:\n{chunk_text[:1000]}\n")

        context = "\n".join(context_parts) if context_parts else "No relevant context found."

        prompt = f"""You are an expert insurance policy analyst. Answer the question based ONLY on the provided policy context.

Question: {question}

Policy Context:
{context}

Instructions:
1. Answer based STRICTLY on the provided context above
2. If the context contains the answer, provide specific details and quote relevant sections
3. If the context doesn't contain sufficient information, say: "The provided policy context does not contain sufficient information to answer this question."
4. Be precise and include specific terms, periods, amounts, or conditions mentioned in the policy
5. Do not make assumptions or add information not present in the context

Answer:"""

        return prompt

    except Exception as e:
        print(f"⚠️ Error creating prompt: {e}")
        return f"Answer this insurance question based on the policy context: {question}"

async def call_gemini_api(prompt: str) -> str:
    """Call Google Gemini API"""
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY environment variable not set")

    await rate_limiter.acquire()

    API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": GEMINI_API_KEY
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
            "maxOutputTokens": 1000
        }
    }

    print(f"🤖 Making Gemini API call...")

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(API_URL, headers=headers, json=payload)

            if response.status_code == 429:
                print("⏰ Rate limited, waiting...")
                await asyncio.sleep(20)
                response = await client.post(API_URL, headers=headers, json=payload)

            response.raise_for_status()
            result = response.json()

            if 'candidates' in result and len(result['candidates']) > 0:
                content = result['candidates'][0]['content']['parts'][0]['text']
                print(f"✅ Received Gemini response: {len(content)} characters")
                return content
            else:
                print(f"❌ Unexpected Gemini API response: {result}")
                raise HTTPException(status_code=500, detail="Unexpected Gemini API response format")

    except Exception as e:
        print(f"❌ Gemini API request failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Gemini API request failed: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "Insurance Claims Processing API - Improved Gemini",
        "version": "4.1.0",
        "model": "gemini-2.0-flash",
        "provider": "Google Gemini API",
        "status": "improved_retrieval",
        "gemini_api_key_configured": bool(GEMINI_API_KEY)
    }

@app.post("/api/v1/hackrx/run")
async def document_qa(
    req: QARequest,
    token: str = Depends(verify_bearer_token)
):
    """Improved Document Q&A with enhanced retrieval"""
    start_time = time.time()

    try:
        print(f"🚀 Processing {len(req.questions)} questions with improved retrieval")
        print(f"📄 Documents to process: {len(req.documents)}")

        # Step 1: Extract PDF text
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

        # Step 2: Enhanced document analysis
        doc_analyzer = DocumentAnalyzer()
        all_text = "\n".join(pdf_texts)
        document_keywords = doc_analyzer.extract_document_keywords(all_text)

        # Step 3: Enhanced chunking
        chunker = EnhancedChunker(chunk_size=1000, overlap=200)
        all_chunks = chunker.smart_chunk_with_context(all_text, document_keywords)

        if not all_chunks:
            raise HTTPException(status_code=400, detail="No text chunks could be created from documents")

        # Clear memory
        del pdf_texts, all_text
        gc.collect()

        # Step 4: Process questions with improved retrieval
        retriever = ImprovedRetriever()
        answers = []

        for i, question in enumerate(req.questions, 1):
            try:
                print(f"🔍 Processing question {i}/{len(req.questions)}: {question[:50]}...")

                relevant_chunks = retriever.retrieve_relevant_chunks(question, all_chunks, top_k=5)

                if not relevant_chunks:
                    answers.append("No relevant information found in the provided policy documents.")
                    continue

                prompt = create_enhanced_prompt(question, relevant_chunks)
                response = await call_gemini_api(prompt)
                answers.append(response.strip())

                if i < len(req.questions):
                    await asyncio.sleep(2)

            except Exception as e:
                print(f"❌ Error processing question {i}: {str(e)}")
                answers.append(f"Error processing this question: {str(e)}")

        elapsed_time = time.time() - start_time
        print(f"✅ Improved processing completed in {elapsed_time:.2f} seconds")

        return {"answers": answers}

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Unexpected error in document_qa: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    print("🚀 Starting Improved Gemini Insurance Claims Processing API...")
    print(f"🔑 Gemini API Key configured: {bool(GEMINI_API_KEY)}")

    if not GEMINI_API_KEY:
        print("❌ WARNING: GEMINI_API_KEY environment variable not set!")

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
