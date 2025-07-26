from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import os
import tempfile
import json
import re
from typing import List, Dict, Any
from pdfminer.high_level import extract_text

# Initialize FastAPI app
app = FastAPI(title="Insurance Claims Processing API")

# Get HuggingFace token (your original working setup)
HF_TOKEN = os.getenv("HF_TOKEN")

class QueryRequest(BaseModel):
    query: str

class QARequest(BaseModel):
    document_url: str
    questions: List[str]

def extract_pdf_from_url(url: str) -> str:
    """Download PDF from URL and extract text content"""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(response.content)
            tmp_path = tmp_file.name
        
        try:
            text = extract_text(tmp_path)
            return text
        finally:
            os.unlink(tmp_path)
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error extracting PDF: {str(e)}")

def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 300) -> List[str]:
    """Split text into overlapping chunks for better context preservation"""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to end at a sentence boundary
        if end < len(text):
            # Look for sentence endings within the last 200 characters
            last_period = text.rfind('.', start + chunk_size - 200, end)
            last_newline = text.rfind('\n', start + chunk_size - 200, end)
            
            boundary = max(last_period, last_newline)
            if boundary > start:
                end = boundary + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position with overlap
        start = end - overlap if end < len(text) else len(text)
        
        if start >= len(text):
            break
    
    return chunks

def find_relevant_chunks_keyword_based(question: str, chunks: List[str], top_k: int = 3) -> List[str]:
    """Find most relevant chunks using keyword matching (lightweight alternative to semantic search)"""
    if not chunks:
        return []
    
    question_words = set(word.lower().strip('.,!?;:') for word in question.split() if len(word) > 2)
    
    scored_chunks = []
    for chunk in chunks:
        chunk_words = set(word.lower().strip('.,!?;:') for word in chunk.split() if len(word) > 2)
        
        # Calculate relevance score
        common_words = question_words.intersection(chunk_words)
        score = len(common_words)
        
        # Boost score for exact phrase matches
        question_lower = question.lower()
        chunk_lower = chunk.lower()
        for word in question_words:
            if word in chunk_lower:
                score += 0.5
        
        scored_chunks.append((score, chunk))
    
    # Sort by score and return top chunks
    scored_chunks.sort(reverse=True, key=lambda x: x[0])
    return [chunk for _, chunk in scored_chunks[:top_k] if _ > 0]

def call_hf_api(messages: List[Dict], max_tokens: int = 800) -> str:
    """Call HuggingFace API using your original working setup"""
    
    if not HF_TOKEN:
        raise HTTPException(status_code=500, detail="HF_TOKEN environment variable not set")
    
    # Your original working API setup
    url = "https://api.novita.ai/v3/openai/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",  # Your original working format
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "Qwen/Qwen3-Coder-480B-A35B-Instruct:novita",  # Your original model
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.3,
        "stream": False
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        
        if 'choices' in result and len(result['choices']) > 0:
            return result['choices'][0]['message']['content']
        else:
            raise HTTPException(status_code=500, detail="Unexpected API response format")
            
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"API Error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

def query_llm_for_claim(user_text: str) -> Dict[str, Any]:
    """Process insurance claim query using your original HF model"""
    prompt = (
        "You are an insurance claim processing assistant. "
        "Given the following customer query, extract all relevant details (age, condition, policy duration, location, etc), "
        "retrieve matching rules from typical Bajaj Allianz health insurance policy (waiting periods, exclusions, etc). "
        "Then output a JSON with: decision (APPROVED/REJECTED/UNDETERMINED), amount (if applicable), and a justification "
        "(mapping each decision to the clause/rule). Use this format:\n\n"
        "{\n"
        '  "decision": "APPROVED",\n'
        '  "amount": "Up to Sum Insured as per Policy Schedule",\n'
        '  "justification": [\n'
        '    {\n'
        '      "criteria": "—",\n'
        '      "status": "PASSED/FAILED",\n'
        '      "explanation": "—",\n'
        '      "clause_reference": "—"\n'
        '    }\n'
        "  ]\n"
        "}\n\n"
        "Customer Query:\n"
        f"{user_text}\n"
        "Be concise and specific. If information is missing, set 'decision' to 'UNDETERMINED' and explain what is needed."
    )

    messages = [
        {"role": "system", "content": "You are a helpful insurance claim processor."},
        {"role": "user", "content": prompt}
    ]
    
    try:
        content = call_hf_api(messages, max_tokens=800)
        
        # Extract JSON from response
        start = content.find("{")
        end = content.rfind("}") + 1
        if start != -1 and end > start:
            result_json = json.loads(content[start:end])
            return result_json
        else:
            return {
                "decision": "UNDETERMINED",
                "justification": "Unable to parse LLM response format"
            }
            
    except Exception as e:
        return {
            "decision": "UNDETERMINED", 
            "justification": f"Error processing query: {str(e)}"
        }

def answer_question_with_context(question: str, context_chunks: List[str]) -> str:
    """Answer question using provided context chunks"""
    context = "\n\n".join(context_chunks)
    
    prompt = f"""Based on the following insurance policy document sections, answer the question as specifically and accurately as possible.

Policy Document Sections:
{context}

Question: {question}

Instructions:
- Answer based ONLY on the information provided in the document sections above
- If the information is not found in the provided sections, state "Information not found in provided document sections"
- Be specific and cite relevant policy sections when possible
- Keep your answer concise but comprehensive"""

    messages = [
        {"role": "system", "content": "You are a helpful insurance policy analyst."},
        {"role": "user", "content": prompt}
    ]
    
    try:
        return call_hf_api(messages, max_tokens=500)
    except Exception as e:
        return f"Error processing question: {str(e)}"

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "qwen/Qwen3-Coder-480B-A35B-Instruct:novita"}

@app.post("/api/v1/insurance-claim")
async def process_claim(req: QueryRequest):
    """Process insurance claim with your original working model"""
    result = query_llm_for_claim(req.query)
    return result

@app.post("/api/v1/document-qa")
async def document_qa(req: QARequest):
    """Process document Q&A using lightweight approach"""
    try:
        # Extract PDF text
        pdf_text = extract_pdf_from_url(req.document_url)
        
        # Chunk the document
        chunks = chunk_text(pdf_text, chunk_size=1500, overlap=300)
        
        responses = []
        for question in req.questions:
            # Find relevant chunks using keyword matching
            relevant_chunks = find_relevant_chunks_keyword_based(question, chunks, top_k=3)
            
            if relevant_chunks:
                answer = answer_question_with_context(question, relevant_chunks)
                evidence_excerpt = relevant_chunks[0][:300] + "..." if relevant_chunks[0] else ""
            else:
                answer = "Information not found in document."
                evidence_excerpt = ""
            
            responses.append({
                "question": question,
                "answer": answer,
                "evidence_excerpt": evidence_excerpt,
                "confidence": 0.85
            })
        
        return {"answers": responses}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")
