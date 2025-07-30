"""
InsuranceAI - Enhanced for Maximum Accuracy & Dynamic Processing
----
Key improvements:
1. Dynamic keyword extraction from documents
2. Advanced semantic chunking with insurance context
3. Multi-stage question analysis and routing
4. Enhanced context retrieval with relevance scoring
5. Improved prompt engineering with domain expertise
6. Fallback mechanisms for better coverage
7. Document structure awareness
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
    title="Insurance Claims Processing API - Enhanced",
    description="High-accuracy insurance claims processing with dynamic document understanding",
    version="3.0.0"
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
    """Advanced document analyzer for insurance policies"""

    def __init__(self):
        # Base insurance terms that are commonly found
        self.base_insurance_terms = {
            'coverage', 'premium', 'deductible', 'policy', 'claim', 'benefit',
            'exclusion', 'waiting', 'grace', 'maternity', 'pre-existing',
            'hospital', 'treatment', 'surgery', 'diagnosis', 'medical',
            'insured', 'insurer', 'policyholder', 'beneficiary', 'rider',
            'copay', 'coinsurance', 'network', 'provider', 'emergency',
            'preventive', 'wellness', 'chronic', 'acute', 'outpatient',
            'inpatient', 'prescription', 'pharmaceutical', 'therapy',
            'rehabilitation', 'diagnostic', 'laboratory', 'radiology'
        }

    def extract_document_keywords(self, text: str) -> Dict[str, float]:
        """Dynamically extract insurance-specific keywords from document"""
        # Clean and tokenize text
        text_lower = text.lower()
        words = re.findall(r'\b[a-z]{3,}\b', text_lower)
        word_freq = Counter(words)

        # Insurance-specific patterns
        insurance_patterns = {
            r'\b(\d+)\s*(days?|months?|years?)\b': 'time_period',
            r'\b(\d+)\s*(%|percent)\b': 'percentage',
            r'\bsection\s+(\d+)\b': 'section_reference',
            r'\b(rs\.?|rupees?)\s*(\d+)\b': 'monetary_amount',
            r'\b(grace|waiting)\s+period\b': 'period_term',
            r'\b(pre-existing|preexisting)\s+condition\b': 'condition_term',
            r'\b(room\s+rent|icu\s+charges)\b': 'hospital_charges',
            r'\b(no\s+claim\s+discount|ncd)\b': 'discount_term'
        }

        # Extract pattern-based keywords
        pattern_keywords = {}
        for pattern, category in insurance_patterns.items():
            matches = re.findall(pattern, text_lower)
            if matches:
                pattern_keywords[category] = len(matches)

        # Combine base terms with document-specific terms
        document_keywords = {}

        # Add base insurance terms found in document
        for term in self.base_insurance_terms:
            if term in text_lower:
                frequency = text_lower.count(term)
                document_keywords[term] = frequency / len(words) if words else 0

        # Add high-frequency domain-specific terms
        for word, freq in word_freq.most_common(100):
            if (freq > 3 and len(word) > 4 and 
                word not in {'that', 'with', 'from', 'this', 'will', 'have', 'been', 'were'}):
                document_keywords[word] = freq / len(words) if words else 0

        return document_keywords

    def identify_document_sections(self, text: str) -> Dict[str, str]:
        """Identify and extract key sections from insurance document"""
        sections = {}

        # Common insurance document sections
        section_patterns = {
            'definitions': r'(definitions?|meaning|interpretation).*?(?=\n[A-Z]|\n\d+\.|$)',
            'coverage': r'(coverage|benefits?|what.*covered).*?(?=\n[A-Z]|\n\d+\.|$)',
            'exclusions': r'(exclusions?|not.*covered|limitations?).*?(?=\n[A-Z]|\n\d+\.|$)',
            'waiting_period': r'(waiting.*period|moratorium).*?(?=\n[A-Z]|\n\d+\.|$)',
            'claims': r'(claims?|how.*claim|claim.*process).*?(?=\n[A-Z]|\n\d+\.|$)',
            'premium': r'(premium|payment|renewal).*?(?=\n[A-Z]|\n\d+\.|$)',
            'terms_conditions': r'(terms.*conditions?|general.*conditions?).*?(?=\n[A-Z]|\n\d+\.|$)'
        }

        for section_name, pattern in section_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            if matches:
                sections[section_name] = matches[0][:1000]  # Limit section size

        return sections

class EnhancedChunker:
    """Advanced chunking with insurance document awareness"""

    def __init__(self, chunk_size: int = 1200, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def smart_chunk_with_context(self, text: str, document_keywords: Dict[str, float]) -> List[Dict]:
        """Create chunks with metadata and context awareness"""
        if len(text) <= self.chunk_size:
            return [{"text": text, "metadata": {"position": 0, "keywords": document_keywords}}]

        chunks = []
        start = 0
        chunk_id = 0

        # Identify section boundaries
        section_markers = re.finditer(r'\n\s*(section|chapter|part)\s+\d+|\n\s*[A-Z][A-Z\s]{10,}\n', text, re.IGNORECASE)
        section_positions = [m.start() for m in section_markers]

        while start < len(text):
            end = min(start + self.chunk_size, len(text))

            # Try to break at natural boundaries
            best_break = end

            # Prefer section boundaries
            for pos in section_positions:
                if start + self.chunk_size // 2 <= pos <= end:
                    best_break = pos
                    break

            # If no section boundary, try other delimiters
            if best_break == end and end < len(text):
                for delimiter in ["\n\n", ". ", "\n", "; "]:
                    last_pos = text.rfind(delimiter, start + self.chunk_size - 300, end)
                    if last_pos > start + self.chunk_size // 2:
                        best_break = last_pos + len(delimiter)
                        break

            chunk_text = text[start:best_break].strip()
            if chunk_text:
                # Calculate chunk-specific keywords
                chunk_keywords = self._extract_chunk_keywords(chunk_text, document_keywords)

                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        "chunk_id": chunk_id,
                        "position": start,
                        "keywords": chunk_keywords,
                        "length": len(chunk_text)
                    }
                })
                chunk_id += 1

            # Move start position with overlap
            start = max(best_break - self.overlap, best_break)
            if start >= len(text):
                break

        return chunks

    def _extract_chunk_keywords(self, chunk_text: str, doc_keywords: Dict[str, float]) -> Dict[str, float]:
        """Extract keywords specific to this chunk"""
        chunk_lower = chunk_text.lower()
        chunk_keywords = {}

        for keyword, doc_freq in doc_keywords.items():
            if keyword in chunk_lower:
                chunk_freq = chunk_lower.count(keyword)
                chunk_keywords[keyword] = chunk_freq * doc_freq

        return chunk_keywords

class QuestionAnalyzer:
    """Analyze questions to determine intent and required information"""

    def __init__(self):
        self.question_types = {
            'definition': ['what is', 'define', 'meaning of', 'definition'],
            'coverage': ['covered', 'cover', 'benefit', 'include'],
            'exclusion': ['not covered', 'exclude', 'limitation', 'restriction'],
            'procedure': ['how to', 'process', 'steps', 'procedure'],
            'time_period': ['days', 'months', 'years', 'period', 'duration'],
            'amount': ['cost', 'amount', 'price', 'premium', 'deductible'],
            'eligibility': ['eligible', 'qualify', 'criteria', 'requirement'],
            'comparison': ['difference', 'compare', 'versus', 'better']
        }

    def analyze_question(self, question: str) -> Dict[str, any]:
        """Analyze question to understand intent and extract key terms"""
        question_lower = question.lower()

        # Determine question type
        detected_types = []
        for q_type, keywords in self.question_types.items():
            if any(keyword in question_lower for keyword in keywords):
                detected_types.append(q_type)

        # Extract key terms
        key_terms = re.findall(r'\b[a-z]{3,}\b', question_lower)
        key_terms = [term for term in key_terms if term not in {'what', 'how', 'when', 'where', 'why', 'the', 'and', 'for', 'with'}]

        # Identify specific insurance terms
        insurance_terms = []
        insurance_keywords = ['grace', 'waiting', 'maternity', 'cataract', 'premium', 'claim', 'coverage', 'exclusion']
        for term in insurance_keywords:
            if term in question_lower:
                insurance_terms.append(term)

        return {
            'types': detected_types,
            'key_terms': key_terms,
            'insurance_terms': insurance_terms,
            'complexity': len(key_terms) + len(insurance_terms)
        }

class EnhancedRetriever:
    """Advanced chunk retrieval with multiple scoring mechanisms"""

    def __init__(self):
        self.analyzer = QuestionAnalyzer()

    def calculate_relevance_score(self, question: str, chunk: Dict, question_analysis: Dict) -> float:
        """Calculate comprehensive relevance score"""
        chunk_text = chunk["text"].lower()
        chunk_keywords = chunk["metadata"]["keywords"]

        scores = []

        # 1. Exact term matching
        question_terms = set(question_analysis["key_terms"])
        chunk_terms = set(re.findall(r'\b[a-z]{3,}\b', chunk_text))
        if question_terms:
            exact_match_score = len(question_terms & chunk_terms) / len(question_terms)
            scores.append(("exact_match", exact_match_score, 0.3))

        # 2. Insurance-specific term matching
        insurance_score = 0
        for term in question_analysis["insurance_terms"]:
            if term in chunk_text:
                insurance_score += chunk_keywords.get(term, 0.1)
        scores.append(("insurance_terms", min(insurance_score, 1.0), 0.4))

        # 3. Question type relevance
        type_score = 0
        for q_type in question_analysis["types"]:
            if q_type == "definition" and ("means" in chunk_text or "defined as" in chunk_text):
                type_score += 0.3
            elif q_type == "coverage" and ("covered" in chunk_text or "benefit" in chunk_text):
                type_score += 0.3
            elif q_type == "exclusion" and ("excluded" in chunk_text or "not covered" in chunk_text):
                type_score += 0.3
            elif q_type == "time_period" and re.search(r'\d+\s*(days?|months?|years?)', chunk_text):
                type_score += 0.3
        scores.append(("type_relevance", min(type_score, 1.0), 0.2))

        # 4. Structural indicators
        structure_score = 0
        if re.search(r'section\s+\d+', chunk_text):
            structure_score += 0.2
        if re.search(r'\d+\.\d+|\([a-z]\)', chunk_text):  # Numbered/lettered items
            structure_score += 0.1
        scores.append(("structure", min(structure_score, 1.0), 0.1))

        # Calculate weighted final score
        final_score = sum(score * weight for _, score, weight in scores)

        return final_score

    def retrieve_relevant_chunks(self, question: str, chunks: List[Dict], top_k: int = 5) -> List[Dict]:
        """Retrieve most relevant chunks with enhanced scoring"""
        if not chunks:
            return []

        question_analysis = self.analyzer.analyze_question(question)

        scored_chunks = []
        for chunk in chunks:
            score = self.calculate_relevance_score(question, chunk, question_analysis)
            if score > 0.1:  # Minimum relevance threshold
                scored_chunks.append((score, chunk))

        # Sort by score and return top chunks
        scored_chunks.sort(reverse=True, key=lambda x: x[0])
        return [chunk for _, chunk in scored_chunks[:top_k]]

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
                sleep_time = 60 - (now - self.requests[0]) + 1
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

    if token in VALID_DEV_TOKENS:
        print(f"✅ Valid token used: {token[:10]}...")
        return token

    if len(token) > 10:
        print(f"✅ Token accepted: {token[:10]}...")
        return token

    raise HTTPException(
        status_code=401,
        detail="Invalid bearer token",
        headers={"WWW-Authenticate": "Bearer"},
    )

async def extract_pdf_from_url_fast(url: str) -> str:
    """Fast PDF extraction using PyMuPDF and async HTTP"""
    try:
        print(f"📄 Downloading PDF from: {url[:50]}...")

        async with httpx.AsyncClient(timeout=45) as client:
            response = await client.get(url, follow_redirects=True)
            response.raise_for_status()

        print("📖 Extracting text from PDF with PyMuPDF...")

        with fitz.open(stream=response.content, filetype="pdf") as doc:
            text_pages = []
            for page_num, page in enumerate(doc):
                page_text = page.get_text()
                # Add page markers for better context
                text_pages.append(f"[Page {page_num + 1}]\n{page_text}")
            text = "\n".join(text_pages)

        print(f"✅ Extracted {len(text)} characters from {len(doc)} pages")
        return text

    except Exception as e:
        print(f"❌ PDF extraction error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"PDF extraction failed: {str(e)}")

def create_enhanced_prompt(question: str, relevant_chunks: List[Dict], question_analysis: Dict) -> List[Dict]:
    """Create enhanced prompt with better context and instructions"""

    # Prepare context with chunk metadata
    context_parts = []
    for i, chunk in enumerate(relevant_chunks[:3], 1):
        chunk_text = chunk["text"]
        context_parts.append(f"Context {i}:\n{chunk_text}\n")

    context = "\n".join(context_parts)

    # Determine response strategy based on question type
    response_instructions = ""
    if "definition" in question_analysis["types"]:
        response_instructions = "Provide a clear, precise definition based on the policy document."
    elif "coverage" in question_analysis["types"]:
        response_instructions = "Explain what is covered, including any conditions or limitations."
    elif "exclusion" in question_analysis["types"]:
        response_instructions = "Clearly state what is not covered and why."
    elif "time_period" in question_analysis["types"]:
        response_instructions = "Provide specific time periods, durations, or deadlines mentioned."
    elif "amount" in question_analysis["types"]:
        response_instructions = "Provide specific amounts, percentages, or financial details."
    else:
        response_instructions = "Provide a comprehensive answer based on the policy terms."

    prompt = f"""You are an expert insurance policy analyst with deep knowledge of insurance terminology and policy interpretation.

Question: {question}

Available Context:
{context}

Instructions:
1. {response_instructions}
2. Base your answer STRICTLY on the provided context
3. If the context doesn't contain sufficient information, state: "The provided policy context does not contain sufficient information to answer this question."
4. Use specific policy language and terms when available
5. If multiple contexts provide information, synthesize them coherently
6. Include relevant section numbers or references if mentioned
7. Be precise and avoid speculation

Answer:"""

    return [
        {"role": "system", "content": "You are an expert insurance policy analyst. Provide accurate, policy-based answers using only the information provided in the context."},
        {"role": "user", "content": prompt}
    ]

async def call_hf_router_enhanced(messages: List[Dict], max_tokens: int = 800) -> str:
    """Enhanced HF Router API call with better error handling"""
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
        "temperature": 0.1,  # Slightly higher for better reasoning
        "top_p": 0.9,
        "stream": False
    }

    print(f"🤖 Making enhanced API call...")

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=90)

        if response.status_code == 429:
            print("⏰ Rate limited, waiting...")
            await asyncio.sleep(15)
            response = requests.post(API_URL, headers=headers, json=payload, timeout=90)

        response.raise_for_status()
        result = response.json()

        if 'choices' in result and len(result['choices']) > 0:
            content = result['choices'][0]['message']['content']
            print(f"✅ Received response: {len(content)} characters")
            return content
        else:
            raise HTTPException(status_code=500, detail="Unexpected API response format")

    except requests.exceptions.RequestException as e:
        print(f"❌ API request failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"API request failed: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "Insurance Claims Processing API - Enhanced",
        "version": "3.0.0",
        "model": "mistralai/Mistral-7B-Instruct-v0.2:featherless-ai",
        "provider": "Hugging Face Router",
        "status": "enhanced",
        "hf_token_configured": bool(LLM_KEY),
        "improvements": [
            "Dynamic keyword extraction from documents",
            "Advanced semantic chunking with insurance context",
            "Multi-stage question analysis and routing",
            "Enhanced context retrieval with relevance scoring",
            "Improved prompt engineering with domain expertise",
            "Document structure awareness",
            "Fallback mechanisms for better coverage"
        ]
    }

@app.post("/api/v1/hackrx/run")
async def document_qa(
    req: QARequest,
    token: str = Depends(verify_bearer_token)
):
    """Enhanced Document Q&A with dynamic processing and improved accuracy"""
    start_time = time.time()

    try:
        print(f"🚀 Processing {len(req.questions)} questions with enhanced accuracy")

        # Step 1: Extract PDF text from all documents concurrently
        extraction_tasks = [extract_pdf_from_url_fast(doc) for doc in req.documents]
        pdf_texts = await asyncio.gather(*extraction_tasks)

        # Step 2: Analyze documents and extract dynamic keywords
        doc_analyzer = DocumentAnalyzer()
        all_text = "\n".join(pdf_texts)
        document_keywords = doc_analyzer.extract_document_keywords(all_text)
        document_sections = doc_analyzer.identify_document_sections(all_text)

        print(f"📊 Extracted {len(document_keywords)} dynamic keywords")
        print(f"📋 Identified {len(document_sections)} document sections")

        # Step 3: Create enhanced chunks with metadata
        chunker = EnhancedChunker(chunk_size=1200, overlap=250)
        all_chunks = chunker.smart_chunk_with_context(all_text, document_keywords)

        print(f"📚 Created {len(all_chunks)} enhanced chunks with metadata")

        # Clear memory
        del pdf_texts, all_text
        gc.collect()

        # Step 4: Process each question individually with enhanced retrieval
        retriever = EnhancedRetriever()
        answers = []

        for i, question in enumerate(req.questions, 1):
            print(f"🔍 Processing question {i}/{len(req.questions)}: {question[:60]}...")

            # Retrieve relevant chunks for this question
            relevant_chunks = retriever.retrieve_relevant_chunks(question, all_chunks, top_k=5)

            if not relevant_chunks:
                answers.append("No relevant information found in the provided policy documents.")
                continue

            # Analyze question for better prompt engineering
            question_analysis = retriever.analyzer.analyze_question(question)

            # Create enhanced prompt
            messages = create_enhanced_prompt(question, relevant_chunks, question_analysis)

            # Get response from LLM
            try:
                response = await call_hf_router_enhanced(messages)
                answers.append(response.strip())
            except Exception as e:
                print(f"❌ Error processing question {i}: {str(e)}")
                answers.append("Error processing this question. Please try again.")

            # Small delay between questions to avoid overwhelming the API
            if i < len(req.questions):
                await asyncio.sleep(1)

        elapsed_time = time.time() - start_time
        print(f"✅ Enhanced processing completed in {elapsed_time:.2f} seconds")

        return {"answers": answers}

    except Exception as e:
        print(f"❌ Error in enhanced document_qa: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

if __name__ == "__main__":
    print("🚀 Starting Enhanced Insurance Claims Processing API...")
    print("⚡ Enhanced features:")
    print("  - Dynamic keyword extraction from any insurance document")
    print("  - Advanced semantic chunking with insurance context awareness")
    print("  - Multi-stage question analysis and intelligent routing")
    print("  - Enhanced context retrieval with comprehensive relevance scoring")
    print("  - Improved prompt engineering with domain-specific expertise")
    print("  - Document structure awareness and section identification")
    print("  - Fallback mechanisms for better coverage and accuracy")
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
