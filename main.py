from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, validator
import os
import tempfile
import json
import time
import asyncio
from typing import List, Dict, Any, Union
from pdfminer.high_level import extract_text
from dotenv import load_dotenv
import gc
import requests
import uvicorn
import traceback


# Initialize FastAPI app
app = FastAPI(
    title="Insurance Claims Processing API - Local Development (OpenRouter)",
    description="Local development server for insurance claims processing using OpenRouter Qwen",
    version="1.0.0"
)

load_dotenv()
# Get OpenRouter API key
LLM_KEY = os.getenv("LLM_KEY")


# Security scheme for bearer token
security = HTTPBearer()


class QueryRequest(BaseModel):
    query: str


class QARequest(BaseModel):
    documents: Union[str, List[str]]
    questions: List[str]

    @validator("documents", pre=True)
    def normalize_documents(cls, v):
        if isinstance(v, str):
            return [v]
        return v


class RateLimiter:
    def __init__(self, max_requests_per_minute=10):  # More conservative
        self.max_requests = max_requests_per_minute
        self.requests = []

    async def wait_if_needed(self):
        now = time.time()
        self.requests = [req_time for req_time in self.requests if now - req_time < 60]
        if len(self.requests) >= self.max_requests:
            sleep_time = 60 - (now - self.requests[0]) + 1
            print(f"⏰ Rate limit reached. Sleeping for {sleep_time:.1f} seconds...")
            await asyncio.sleep(sleep_time)  #  was missing!
            self.requests = []
        self.requests.append(now)


rate_limiter = RateLimiter()


def verify_bearer_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify bearer token - for local development"""
    token = credentials.credentials
    
    # Accept specific development tokens
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


async def extract_pdf_from_url(url: str) -> str:
    try:
        print(f"📄 Downloading PDF from: {url[:50]}...")
        response = requests.get(url, timeout=30, stream=True)
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    tmp_file.write(chunk)
            tmp_path = tmp_file.name
        
        try:
            print("📖 Extracting text from PDF...")
            text = extract_text(tmp_path)
            os.unlink(tmp_path)
            gc.collect()
            print(f"✅ Extracted {len(text)} characters from PDF")
            return text
        except Exception as e:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise e
    except Exception as e:
        print(f"❌ PDF extraction error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"PDF extraction failed: {str(e)}")


def chunk_text_memory_efficient(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            last_period = text.rfind('.', start + chunk_size - 200, end)
            last_newline = text.rfind('\n', start + chunk_size - 200, end)
            boundary = max(last_period, last_newline)
            if boundary > start:
                end = boundary + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap if end < len(text) else len(text)
        if start >= len(text):
            break
    
    return chunks


def find_relevant_chunks(question: str, chunks: List[str], top_k: int = 2) -> List[str]:
    if not chunks:
        return []
    
    question_words = set(word.lower().strip('.,!?;:"()[]') for word in question.split() if len(word) > 2)
    scored_chunks = []
    
    for chunk in chunks:
        chunk_words = set(word.lower().strip('.,!?;:"()[]') for word in chunk.split() if len(word) > 2)
        common_words = question_words.intersection(chunk_words)
        word_overlap_score = len(common_words) / max(len(question_words), 1)
        
        phrase_score = 0
        question_lower = question.lower()
        chunk_lower = chunk.lower()
        key_terms = ['waiting period', 'pre-existing', 'exclusion', 'coverage', 'benefit', 'limit']
        
        for term in key_terms:
            if term in question_lower and term in chunk_lower:
                phrase_score += 0.3
        
        total_score = word_overlap_score + phrase_score
        scored_chunks.append((total_score, chunk))
    
    scored_chunks.sort(reverse=True, key=lambda x: x[0])
    return [chunk for _, chunk in scored_chunks[:top_k] if _ > 0.1]


async def call_openrouter_with_retry(messages: List[Dict], max_tokens: int = 1000, max_retries: int = 3) -> str:
    """OpenRouter API call with enhanced debugging and retry logic"""
    
    if not LLM_KEY:
        print("❌ ERROR: LLM_Key environment variable not set")
        raise HTTPException(status_code=500, detail="LLM_Key environment variable not set")
    
    print(f"🔑 Using LLM_Key: {LLM_KEY[:10]}...")
    
    # Apply rate limiting
    await rate_limiter.wait_if_needed()
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {LLM_KEY}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": "qwen/qwen3-235b-a22b-2507:free",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.1,
        "stream": False
    }
    
    print(f"📤 Making API request with {len(messages)} messages...")
    
    for attempt in range(max_retries):
        try:
            print(f"🤖 OpenRouter API call attempt {attempt + 1}/{max_retries}")
            
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            
            # 🔥 ENHANCED DEBUG LOGGING
            print(f"📊 Response Status: {response.status_code}")
            print(f"📊 Response Headers: {dict(response.headers)}")
            
            if response.status_code == 429:  # Rate limit hit
                try:
                    error_data = response.json()
                    print(f"❌ Rate Limit Error Details: {json.dumps(error_data, indent=2)}")
                except:
                    print(f"❌ Rate Limit Error (raw response): {response.text}")
                
                wait_time = 2 ** attempt * 10  # 10s, 20s, 40s
                print(f"⚠️ Rate limit hit. Waiting {wait_time}s before retry...")
                await asyncio.sleep(wait_time)
                continue
            
            elif response.status_code == 401:
                print(f"❌ Authentication Error: {response.text}")
                raise HTTPException(status_code=401, detail=f"Authentication failed: {response.text}")
            
            elif response.status_code != 200:
                print(f"❌ API Error {response.status_code}: {response.text}")
                raise HTTPException(status_code=response.status_code, detail=f"API error: {response.text}")
            
            # Parse successful response
            try:
                result = response.json()
                print(f"✅ API Response received: {len(str(result))} characters")
            except json.JSONDecodeError as e:
                print(f"❌ JSON Parse Error: {str(e)}")
                print(f"Raw response: {response.text[:500]}...")
                raise HTTPException(status_code=500, detail="Invalid JSON response from API")
            
            if 'choices' in result and len(result['choices']) > 0:
                content = result['choices'][0]['message']['content']
                print(f"✅ Successfully extracted content: {len(content)} characters")
                return content
            else:
                print(f"❌ Unexpected API response structure: {json.dumps(result, indent=2)}")
                raise HTTPException(status_code=500, detail="Unexpected API response format")
                
        except requests.exceptions.Timeout as e:
            print(f"⏰ Timeout error on attempt {attempt + 1}: {str(e)}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt * 5
                print(f"⏰ Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
                continue
            print("❌ Final timeout - giving up")
            raise HTTPException(status_code=504, detail="API timeout after retries")
            
        except requests.exceptions.RequestException as e:
            print(f"🔄 Request error on attempt {attempt + 1}: {str(e)}")
            print(f"🔄 Error type: {type(e).__name__}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"🔄 Error response status: {e.response.status_code}")
                print(f"🔄 Error response text: {e.response.text}")
            
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt * 5
                print(f"🔄 Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
                continue
            print("❌ Final request error - giving up")
            raise HTTPException(status_code=500, detail=f"API request failed: {str(e)}")
        
        except Exception as e:
            print(f"💥 Unexpected error on attempt {attempt + 1}: {str(e)}")
            print(f"💥 Error type: {type(e).__name__}")
            print(f"💥 Full traceback:")
            traceback.print_exc()
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt * 5
                print(f"💥 Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
                continue
            print("❌ Final unexpected error - giving up")
            raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    
    print("❌ All retry attempts exhausted")
    raise HTTPException(status_code=500, detail="Max retries exceeded")


async def process_questions_in_batches(questions: List[str], relevant_chunks_map: Dict[str, List[str]]) -> List[Dict]:
    results = []
    batch_size = 5  
    
    print(f"🔄 Processing {len(questions)} questions in batches of {batch_size}")
    
    for i in range(0, len(questions), batch_size):
        batch_questions = questions[i:i + batch_size]
        batch_num = i // batch_size + 1
        
        print(f"\n🔄 === PROCESSING BATCH {batch_num} ===")
        print(f"🔄 Questions {i + 1}-{min(i + batch_size, len(questions))}")
        print(f"🔄 Batch questions: {[q[:50] + '...' for q in batch_questions]}")
        
        # Build context for this batch
        batch_context = ""
        batch_prompt = "You are an insurance policy expert. Answer the following questions based on the policy document sections provided.\n\n"
        
        for j, question in enumerate(batch_questions):
            relevant_chunks = relevant_chunks_map.get(question, [])
            print(f"🔍 Question {j+1} has {len(relevant_chunks)} relevant chunks")
            
            if relevant_chunks:
                batch_context += f"\n--- Context for Question {j + 1} ---\n"
                batch_context += "\n".join(relevant_chunks[:2])  # Top 2 chunks only
            
            batch_prompt += f"\nQuestion {j + 1}: {question}\n"
        
        batch_prompt += f"\nPolicy Document Sections:\n{batch_context}\n\n"
        batch_prompt += "Instructions:\n"
        batch_prompt += "- Answer each question separately and clearly\n"
        batch_prompt += "- Number your answers (Answer 1:, Answer 2:, etc.)\n"
        batch_prompt += "- Base answers ONLY on the provided policy sections\n"
        batch_prompt += "- If information is not found, state 'Information not found in provided sections'\n"
        batch_prompt += "- Be specific and cite relevant policy sections when possible\n"
        
        print(f"📝 Batch prompt length: {len(batch_prompt)} characters")
        
        messages = [
            {"role": "system", "content": "You are a helpful insurance policy analyst."},
            {"role": "user", "content": batch_prompt}
        ]
        
        try:
            print(f"📤 Sending batch {batch_num} to OpenRouter...")
            
            # Get batch response
            batch_response = await call_openrouter_with_retry(messages, max_tokens=1500)
            
            print(f"📥 Received response for batch {batch_num}: {len(batch_response)} characters")
            print(f"📥 Response preview: {batch_response[:200]}...")
            
            # Parse batch response into individual answers
            answers = parse_batch_response(batch_response, len(batch_questions))
            print(f"📋 Parsed {len(answers)} answers from batch response")
            
            # Create response objects
            for j, question in enumerate(batch_questions):
                answer = answers[j] if j < len(answers) else "Error processing this question"
                evidence = "\n".join(relevant_chunks_map.get(question, [])[:1])  # First chunk as evidence
                
                result_obj = {
                    "question": question,
                    "answer": answer
                }
                
                results.append(answer)
                print(f"✅ Added answer {j+1} for question: {question[:50]}...")
        
        except Exception as e:
            print(f"❌ ERROR PROCESSING BATCH {batch_num}")
            print(f"❌ Error type: {type(e).__name__}")
            print(f"❌ Error message: {str(e)}")
            print(f"❌ Full traceback:")
            traceback.print_exc()
            
            # Add error responses for this batch
            for question in batch_questions:
                error_answer = f"Error processing question in batch {batch_num}: {str(e)}"
                error_result = {
                    "question": question,
                    "answer": f"Error processing question in batch {batch_num}: {str(e)}"
                }
                results.append(error_answer)
                print(f"❌ Added error response for: {question[:50]}...")
        
        # Longer delay between batches
        if i + batch_size < len(questions):
            sleep_time = 2
            print(f"⏳ Waiting {sleep_time}s before next batch...")
            await asyncio.sleep(sleep_time)
    
    print(f"🏁 Completed processing all batches. Total results: {len(results)}")
    return results


def parse_batch_response(response: str, expected_count: int) -> List[str]:
    """Parse batch response into individual answers"""
    print(f"🔍 Parsing batch response for {expected_count} expected answers")
    answers = []
    
    # Try to split by "Answer X:" pattern
    import re
    answer_pattern = r'Answer\s+(\d+):\s*(.+?)(?=Answer\s+\d+:|$)'
    matches = re.findall(answer_pattern, response, re.DOTALL | re.IGNORECASE)
    
    if matches:
        print(f"✅ Found {len(matches)} numbered answers using regex")
        # Sort by answer number and extract content
        matches.sort(key=lambda x: int(x[0]))
        answers = [match[1].strip() for match in matches]
    else:
        print("⚠️ No numbered answers found, using fallback parsing")
        # Fallback: split by double newlines and take first N parts
        parts = response.split('\n\n')
        answers = [part.strip() for part in parts if part.strip()]
        print(f"📝 Fallback found {len(answers)} parts")
    
    # Ensure we have the right number of answers
    while len(answers) < expected_count:
        answers.append("Answer not found in response")
    
    final_answers = answers[:expected_count]
    print(f"📋 Final parsed answers: {len(final_answers)}")
    
    return final_answers


# API Routes
@app.get("/")
async def root():
    return {
        "message": "Insurance Claims Processing API - Local Development (OpenRouter)",
        "version": "1.0.0",
        "model": "qwen/qwen3-235b-a22b-2507:free",
        "provider": "OpenRouter",
        "endpoints": {
            "document_qa": "/api/v1/hackrx/run"
        },
        "status": "running",
        "server": "localhost:8000",
        "llm_key_configured": bool(LLM_KEY)
    }


@app.post("/api/v1/hackrx/run")
async def document_qa(
    req: QARequest, 
    token: str = Depends(verify_bearer_token)
):
    """Document Q&A with batch processing and bearer token authentication"""
    try:
        print(f"🔐 Processing {len(req.questions)} questions with token: {token[:10]}...")
        
        chunks = list()

        # Extract PDF text
        for doc in req.documents:

            pdf_text = await extract_pdf_from_url(doc)
            chunks.extend(chunk_text_memory_efficient(pdf_text, chunk_size=1200, overlap=200))


        print(f"📚 Created {len(chunks)} text chunks")
        
        # Clear memory
        del pdf_text
        gc.collect()
        
        # Find relevant chunks for each question
        print("🔍 Finding relevant chunks for each question...")
        relevant_chunks_map = {}
        for i, question in enumerate(req.questions):
            print(f"🔍 Processing question {i+1}: {question[:50]}...")
            relevant_chunks_map[question] = find_relevant_chunks(question, chunks, top_k=2)
            print(f"📋 Found {len(relevant_chunks_map[question])} relevant chunks")
        
        # Process questions in batches
        print("🚀 Starting batch processing...")
        responses = await process_questions_in_batches(req.questions, relevant_chunks_map)
        
        print(f"✅ Successfully processed all questions. Returning {len(responses)} responses")
        return {"answers": responses}
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR in document_qa: {str(e)}")
        print(f"❌ Error type: {type(e).__name__}")
        print("❌ Full traceback:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")


if __name__ == "__main__":
    print("🚀 Starting Insurance Claims Processing API...")
    print("🤖 Using OpenRouter Qwen Model: qwen/qwen3-235b-a22b-2507:free")
    print("📍 Server will be available at: http://localhost:8000")
    print("📚 API documentation: http://localhost:8000/docs")
    print("🔐 Bearer token authentication required")
    
    # Check if LLM_Key is configured
    if not LLM_KEY:
        print("❌ WARNING: LLM_Key environment variable not set!")
        print("💡 Please set your OpenRouter API key in the .env file")
    else:
        print(f"✅ LLM_Key configured: {LLM_KEY[:10]}...")
    
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000,
        reload=True,
        log_level="info"
    )
