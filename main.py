"""
Enhanced InsuranceAI - Improved Version for Better Accuracy
----
Key improvements:
1. Advanced chunk creation with better context preservation
2. Enhanced keyword extraction with semantic relationships
3. Improved multi-pass retrieval with document structure awareness
4. Better question-context matching with domain knowledge
5. Robust error handling and fallback mechanisms
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, validator
import os
import tempfile
import time
import asyncio
from typing import List, Union, Dict, Tuple
import uvicorn
import traceback
import re
import fitz  # PyMuPDF
import httpx
from dotenv import load_dotenv
from collections import defaultdict, Counter
import math

# Initialize FastAPI app
app = FastAPI(
    title="Enhanced Insurance Claims Processing API",
    description="Enhanced insurance claims processing with improved accuracy for unknown documents",
    version="4.0.0"
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

class AsyncRateLimiter:
    def __init__(self, max_requests_per_minute=15):  # Reduced for stability
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

    if token in VALID_DEV_TOKENS or len(token) > 10:
        print(f"✅ Token accepted: {token[:10]}...")
        return token

    raise HTTPException(
        status_code=401,
        detail="Invalid bearer token",
        headers={"WWW-Authenticate": "Bearer"},
    )

async def extract_pdf_from_url_fast(url: str) -> str:
    """Enhanced PDF extraction with better text processing"""
    try:
        print(f"📄 Downloading PDF from: {url[:80]}...")

        async with httpx.AsyncClient(timeout=45) as client:
            response = await client.get(url, follow_redirects=True)
            response.raise_for_status()

        pdf_size = len(response.content)
        print(f"📖 Extracting text from PDF ({pdf_size} bytes)...")

        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_file.write(response.content)
            temp_path = temp_file.name

        try:
            doc = fitz.open(temp_path)
            text_pages = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                # Enhanced text extraction with better formatting
                text = page.get_text()
                
                # Clean and normalize text
                text = re.sub(r'\n\s*\n', '\n\n', text)  # Normalize paragraph breaks
                text = re.sub(r'([.!?])\s*\n', r'\1\n\n', text)  # Better sentence breaks
                text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
                
                text_pages.append(text)
            
            doc.close()
            full_text = "\n\n".join(text_pages)
            print(f"✅ Extracted {len(full_text)} characters from {len(text_pages)} pages")
            
            return full_text

        finally:
            try:
                os.unlink(temp_path)
            except:
                pass

    except Exception as e:
        print(f"❌ PDF extraction error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"PDF extraction failed: {str(e)}")

def extract_enhanced_keywords(text: str) -> Dict[str, List[str]]:
    """Enhanced keyword extraction with semantic grouping"""
    
    # Comprehensive insurance domain patterns
    insurance_patterns = {
        'time_periods': [
            r'grace period[s]?', r'waiting period[s]?', r'\d+\s*days?', r'\d+\s*months?', 
            r'\d+\s*years?', r'thirty\s*days?', r'sixty\s*days?', r'ninety\s*days?',
            r'twenty[- ]four\s*months?', r'thirty[- ]six\s*months?', r'continuous\s*coverage',
            r'policy\s*period', r'policy\s*year'
        ],
        'medical_conditions': [
            r'pre[- ]existing\s*disease[s]?', r'maternity\s*benefit[s]?', r'cataract\s*surgery',
            r'organ\s*donor', r'mental\s*illness', r'critical\s*illness', r'diabetes',
            r'hypertension', r'cancer', r'stroke', r'kidney\s*failure', r'heart\s*disease'
        ],
        'financial_terms': [
            r'no\s*claim\s*discount', r'premium\s*payment', r'sum\s*insured', r'deductible',
            r'co[- ]payment', r'room\s*rent', r'icu\s*charges', r'sub[- ]limit[s]?',
            r'\d+%', r'rupees?', r'rs\.?', r'inr', r'lacs?', r'lakhs?'
        ],
        'coverage_types': [
            r'in[- ]patient', r'out[- ]patient', r'day\s*care', r'domiciliary',
            r'hospitalization', r'ayush\s*treatment', r'ambulance', r'emergency',
            r'vaccination', r'health\s*check[- ]up[s]?'
        ],
        'policy_structure': [
            r'section\s+\d+', r'clause\s+\d+', r'article\s+\d+', r'exclusion[s]?',
            r'condition[s]?', r'benefit[s]?', r'coverage', r'definition[s]?'
        ]
    }
    
    keywords = defaultdict(list)
    text_lower = text.lower()
    
    # Extract pattern-based keywords by category
    for category, patterns in insurance_patterns.items():
        for pattern in patterns:
            matches = re.findall(pattern, text_lower)
            keywords[category].extend(matches)
    
    # Extract numerical contexts with better precision
    numerical_contexts = re.findall(
        r'([a-zA-Z\s]{2,20}\d+[\s]*(?:days?|months?|years?|%|rupees?|rs\.?|lacs?|lakhs?)[a-zA-Z\s]{0,20})', 
        text_lower
    )
    keywords['numerical'].extend([match.strip() for match in numerical_contexts])
    
    # Extract section headers and important structural elements
    section_headers = re.findall(
        r'(?:section|clause|article|chapter)\s+[\d\.]+[^\n]*', 
        text_lower
    )
    keywords['structure'].extend(section_headers)
    
    # Extract important definitions
    definitions = re.findall(
        r'([a-zA-Z\s]+)\s+means\s+([^.]{10,100})', 
        text_lower
    )
    keywords['definitions'].extend([f"{term} means {definition[:50]}" for term, definition in definitions])
    
    # Deduplicate and limit
    for category in keywords:
        keywords[category] = list(set(keywords[category]))[:50]
    
    total_keywords = sum(len(v) for v in keywords.values())
    print(f"📊 Extracted {total_keywords} enhanced keywords across {len(keywords)} categories")
    
    return dict(keywords)

def create_enhanced_chunks(text: str, chunk_size: int = 2000, overlap: int = 400) -> List[Dict[str, any]]:
    """Create enhanced chunks with metadata and better context preservation"""
    
    if len(text) <= chunk_size:
        return [{
            'content': text,
            'start_pos': 0,
            'end_pos': len(text),
            'section': 'full_document',
            'chunk_type': 'complete'
        }]

    chunks = []
    
    # First, try to split by major sections
    section_pattern = r'\n(?=(?:Section|SECTION|Chapter|CHAPTER)\s+\d+)'
    sections = re.split(section_pattern, text)
    
    for section_idx, section in enumerate(sections):
        if not section.strip():
            continue
            
        section_title = extract_section_title(section)
        
        if len(section) <= chunk_size:
            chunks.append({
                'content': section.strip(),
                'start_pos': text.find(section),
                'end_pos': text.find(section) + len(section),
                'section': section_title,
                'chunk_type': 'section'
            })
        else:
            # Further split large sections intelligently
            sub_chunks = split_text_intelligently(section, chunk_size, overlap)
            for i, sub_chunk in enumerate(sub_chunks):
                chunks.append({
                    'content': sub_chunk,
                    'start_pos': text.find(sub_chunk),
                    'end_pos': text.find(sub_chunk) + len(sub_chunk),
                    'section': f"{section_title}_part_{i+1}",
                    'chunk_type': 'subsection'
                })

    print(f"📚 Created {len(chunks)} enhanced chunks with metadata")
    return chunks

def extract_section_title(section_text: str) -> str:
    """Extract section title from text"""
    lines = section_text.strip().split('\n')
    first_line = lines[0].strip()
    
    # Look for section headers
    if re.match(r'(?:Section|SECTION|Chapter|CHAPTER)\s+\d+', first_line):
        return first_line[:50]
    
    # Look for numbered items
    if re.match(r'\d+\.?\s+', first_line):
        return first_line[:50]
    
    return first_line[:30] if first_line else "unknown_section"

def split_text_intelligently(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Enhanced intelligent text splitting with better boundary detection"""
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            # Enhanced break point detection
            break_points = [
                (r'\n\n(?=\d+\.)', 50),  # Numbered sections
                (r'\n\n(?=[A-Z][a-z])', 40),  # New paragraphs starting with capital
                (r'\.\s*\n', 30),  # Sentence ends with newline
                (r'[.!?]\s+', 20),  # Sentence ends
                (r';\s+', 15),  # Clause breaks
                (r',\s+', 10),  # Comma breaks
                (r'\s+', 5)  # Word breaks
            ]

            best_break = None
            for pattern, priority in break_points:
                matches = list(re.finditer(pattern, text[start:end]))
                if matches:
                    # Find the best match (closest to end but not too early)
                    for match in reversed(matches):
                        if match.start() > chunk_size // 3:  # Don't break too early
                            best_break = start + match.end()
                            break
                    if best_break:
                        break

            if best_break:
                end = best_break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Enhanced overlap calculation
        if end >= len(text):
            break
            
        # Find good overlap start point
        overlap_start = max(end - overlap, start + chunk_size // 2)
        start = overlap_start

    return chunks

def classify_question_type(question: str) -> str:
    """Classify question type for better retrieval strategy"""
    question_lower = question.lower()
    
    if any(word in question_lower for word in ['grace period', 'waiting period', 'days', 'months', 'years']):
        return 'time_period'
    elif any(word in question_lower for word in ['cover', 'coverage', 'benefit', 'include']):
        return 'coverage'
    elif any(word in question_lower for word in ['limit', 'maximum', 'minimum', 'amount', 'charges']):
        return 'financial'
    elif any(word in question_lower for word in ['define', 'definition', 'means', 'what is']):
        return 'definition'
    elif any(word in question_lower for word in ['exclude', 'exclusion', 'not cover']):
        return 'exclusion'
    else:
        return 'general'

def enhanced_multi_pass_retrieval(
    question: str, 
    chunks: List[Dict[str, any]], 
    keywords: Dict[str, List[str]]
) -> List[Dict[str, any]]:
    """Enhanced multi-pass chunk retrieval with improved scoring"""
    
    print(f"🔍 Starting enhanced retrieval for: {question[:60]}...")
    
    question_type = classify_question_type(question)
    question_lower = question.lower()
    question_words = set(re.findall(r'\b\w{3,}\b', question_lower))
    
    scored_chunks = []
    
    for chunk in chunks:
        chunk_content = chunk['content'].lower()
        chunk_words = set(re.findall(r'\b\w{3,}\b', chunk_content))
        
        # Base similarity score
        word_overlap = len(question_words & chunk_words)
        base_score = word_overlap / len(question_words) if question_words else 0
        
        # Enhanced scoring factors
        score_factors = {
            'base_similarity': base_score,
            'exact_phrase_match': 0,
            'keyword_category_match': 0,
            'numerical_match': 0,
            'section_relevance': 0,
            'question_type_boost': 0
        }
        
        # Exact phrase matching
        for word in question_words:
            if len(word) > 4 and word in chunk_content:
                score_factors['exact_phrase_match'] += 0.3
        
        # Keyword category matching based on question type
        relevant_categories = {
            'time_period': ['time_periods', 'policy_structure'],
            'coverage': ['coverage_types', 'medical_conditions'],
            'financial': ['financial_terms', 'numerical'],
            'definition': ['definitions', 'policy_structure'],
            'exclusion': ['policy_structure', 'medical_conditions']
        }.get(question_type, ['policy_structure'])
        
        for category in relevant_categories:
            if category in keywords:
                for keyword in keywords[category]:
                    if keyword in chunk_content:
                        score_factors['keyword_category_match'] += 0.2
        
        # Numerical pattern matching
        question_numbers = re.findall(r'\d+', question)
        chunk_numbers = re.findall(r'\d+', chunk_content)
        if question_numbers and chunk_numbers:
            common_numbers = set(question_numbers) & set(chunk_numbers)
            if common_numbers:
                score_factors['numerical_match'] = 0.4
        
        # Section relevance
        section_name = chunk.get('section', '').lower()
        if any(word in section_name for word in question_words):
            score_factors['section_relevance'] = 0.3
        
        # Question type specific boosts
        type_boosts = {
            'time_period': ['period', 'days', 'months', 'years', 'waiting', 'grace'],
            'coverage': ['cover', 'benefit', 'treatment', 'expense'],
            'financial': ['limit', 'amount', 'charges', 'premium', 'discount'],
            'definition': ['means', 'definition', 'refers', 'includes'],
            'exclusion': ['exclude', 'not', 'shall not', 'except']
        }
        
        if question_type in type_boosts:
            for boost_word in type_boosts[question_type]:
                if boost_word in chunk_content:
                    score_factors['question_type_boost'] += 0.1
        
        # Calculate final score
        final_score = sum(score_factors.values())
        
        scored_chunks.append({
            'chunk': chunk,
            'score': final_score,
            'score_breakdown': score_factors
        })
    
    # Sort by score and return top chunks
    scored_chunks.sort(key=lambda x: x['score'], reverse=True)
    
    # Enhanced selection logic
    selected_chunks = []
    total_length = 0
    max_total_length = 8000  # Increased context window
    
    for scored_chunk in scored_chunks:
        chunk = scored_chunk['chunk']
        if (len(selected_chunks) < 10 and 
            total_length + len(chunk['content']) < max_total_length and
            scored_chunk['score'] > 0.1):  # Minimum relevance threshold
            
            selected_chunks.append(chunk)
            total_length += len(chunk['content'])
    
    print(f"🎯 Selected {len(selected_chunks)} chunks with scores: {[f'{sc['score']:.2f}' for sc in scored_chunks[:5]]}")
    
    return selected_chunks

async def call_gemini_api_with_retry(prompt: str, max_retries: int = 3) -> str:
    """Enhanced Gemini API call with better error handling"""
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY environment variable not set")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"

    headers = {"Content-Type": "application/json"}

    payload = {
        "contents": [{
            "parts": [{
                "text": prompt
            }]
        }],
        "generationConfig": {
            "temperature": 0.1,  # Lower temperature for more consistent answers
            "topK": 40,
            "topP": 0.95,
            "maxOutputTokens": 2048,  # Increased for detailed answers
        }
    }

    for attempt in range(max_retries):
        try:
            await rate_limiter.acquire()
            print(f"🤖 Making enhanced Gemini API call... (attempt {attempt + 1})")

            async with httpx.AsyncClient(timeout=45) as client:
                response = await client.post(url, headers=headers, json=payload)

                if response.status_code == 503:
                    wait_time = (attempt + 1) * 3  # Longer backoff
                    print(f"❌ Gemini API request failed: Server error '503 Service Unavailable'")
                    if attempt < max_retries - 1:
                        print(f"⏰ Retrying in {wait_time} seconds...")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        return "I apologize, but the service is temporarily unavailable. Please try again later."

                response.raise_for_status()
                result = response.json()

                if 'candidates' in result and len(result['candidates']) > 0:
                    content = result['candidates'][0]['content']['parts'][0]['text']
                    print(f"✅ Received enhanced Gemini response: {len(content)} characters")
                    return content
                else:
                    raise HTTPException(status_code=500, detail="Unexpected API response format")

        except Exception as e:
            print(f"❌ Enhanced API call error: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep((attempt + 1) * 2)
                continue
            else:
                return "I apologize, but I encountered an error while processing your request. Please try again."

    return "I apologize, but the service is currently unavailable after multiple attempts. Please try again later."

@app.get("/")
async def root():
    return {
        "message": "Enhanced Insurance Claims Processing API",
        "version": "4.0.0",
        "model": "gemini-2.0-flash",
        "provider": "Google Gemini",
        "status": "enhanced",
        "gemini_api_configured": bool(GEMINI_API_KEY),
        "improvements": [
            "Enhanced chunk creation with metadata",
            "Advanced keyword extraction with semantic grouping",
            "Improved multi-pass retrieval with scoring",
            "Question type classification",
            "Better context preservation",
            "Robust error handling with fallbacks"
        ]
    }

@app.post("/api/v1/hackrx/run")
async def enhanced_document_qa(
    req: QARequest,
    token: str = Depends(verify_bearer_token)
):
    """Enhanced Document Q&A with improved accuracy for unknown documents"""
    start_time = time.time()

    try:
        print(f"🚀 Processing {len(req.questions)} questions with enhanced multi-pass retrieval")
        print(f"📄 Documents to process: {len(req.documents)}")

        # Step 1: Extract PDF text from all documents
        all_text = ""
        for i, doc_url in enumerate(req.documents, 1):
            print(f"📄 Processing document {i}/{len(req.documents)}")
            text = await extract_pdf_from_url_fast(doc_url)
            all_text += f"\n\n--- Document {i} ---\n\n" + text

        # Step 2: Enhanced keyword extraction with semantic grouping
        keywords = extract_enhanced_keywords(all_text)

        # Step 3: Create enhanced chunks with metadata
        chunks = create_enhanced_chunks(all_text, chunk_size=2000, overlap=400)

        # Step 4: Process each question with enhanced retrieval
        answers = []
        for i, question in enumerate(req.questions, 1):
            print(f"🔍 Processing question {i}/{len(req.questions)}: {question[:60]}...")

            try:
                # Enhanced multi-pass chunk retrieval
                relevant_chunks = enhanced_multi_pass_retrieval(question, chunks, keywords)

                # Create enhanced context
                context_parts = []
                for chunk in relevant_chunks[:6]:  # Use top 6 chunks
                    section_info = f"[Section: {chunk.get('section', 'unknown')}]"
                    context_parts.append(f"{section_info}\n{chunk['content']}")

                context = "\n\n---\n\n".join(context_parts)

                # Enhanced prompt with better instructions
                prompt = f"""You are a professional insurance policy analyst with expertise in policy interpretation. Answer the following question based STRICTLY on the provided policy document context.

Question: {question}

Policy Document Context:
{context}

Instructions:
- Answer based ONLY on the provided context - do not use external knowledge
- Be precise, specific, and comprehensive
- Include ALL relevant details like time periods, amounts, conditions, and exceptions
- Quote specific sections or clauses when applicable
- If multiple conditions apply, list them clearly
- If the context contains partial information, state what is available and what is missing
- If the context doesn't contain the answer, respond: "The provided context does not contain sufficient information to answer this question."
- Use clear, professional language suitable for insurance documentation

Answer:"""

                # Get enhanced answer from Gemini
                answer = await call_gemini_api_with_retry(prompt)
                answers.append(answer.strip())

                # Reduced rate limiting delay for better performance
                await asyncio.sleep(1)

            except Exception as e:
                print(f"❌ Error processing question {i}: {e}")
                answers.append("I apologize, but I encountered an error processing this question. Please try again.")

        elapsed_time = time.time() - start_time
        print(f"✅ Enhanced processing completed in {elapsed_time:.2f} seconds")

        return {"answers": answers}

    except Exception as e:
        print(f"❌ Error in enhanced_document_qa: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

if __name__ == "__main__":
    print("🚀 Starting Enhanced Insurance Claims Processing API...")
    print("🔧 Key enhancements:")
    print("  - Advanced chunk creation with metadata and better context preservation")
    print("  - Enhanced keyword extraction with semantic grouping")
    print("  - Improved multi-pass retrieval with sophisticated scoring")
    print("  - Question type classification for targeted retrieval")
    print("  - Better numerical and structural pattern matching")
    print("  - Robust error handling with intelligent fallbacks")
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