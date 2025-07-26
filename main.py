from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import os
import tempfile
import json
from typing import List, Dict, Any
from pdfminer.high_level import extract_text

# Initialize FastAPI app
app = FastAPI(title="Insurance Claims Processing MVP")

# Get API key
LLM_API_KEY = os.getenv("LLM_Key")

class ClaimRequest(BaseModel):
    query: str

class DocumentQARequest(BaseModel):
    document_url: str
    questions: List[str]

def call_llm_api(messages: List[Dict], max_tokens: int = 800) -> str:
    """Call OpenRouter API"""
    if not LLM_API_KEY:
        raise HTTPException(status_code=500, detail="LLM_Key environment variable not set")
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "qwen/qwen3-235b-a22b-2507:free",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.3
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"API Error: {str(e)}")

def extract_pdf_from_url(url: str) -> str:
    """Download and extract text from PDF"""
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

def chunk_text(text: str, chunk_size: int = 2000) -> List[str]:
    """Split text into chunks"""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            # Find sentence boundary
            last_period = text.rfind('.', start, end)
            if last_period > start:
                end = last_period + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
    
    return chunks

def find_relevant_chunks(question: str, chunks: List[str], top_k: int = 2) -> List[str]:
    """Find relevant chunks using keyword matching"""
    question_words = set(word.lower() for word in question.split() if len(word) > 2)
    
    scored_chunks = []
    for chunk in chunks:
        chunk_words = set(word.lower() for word in chunk.split())
        score = len(question_words.intersection(chunk_words))
        if score > 0:
            scored_chunks.append((score, chunk))
    
    scored_chunks.sort(reverse=True)
    return [chunk for _, chunk in scored_chunks[:top_k]]

@app.get("/")
async def root():
    return {"message": "Insurance Claims Processing MVP", "status": "running"}

@app.post("/api/v1/insurance-claim")
async def process_claim(request: ClaimRequest):
    """Process insurance claim query"""
    prompt = f"""You are an insurance claims processor. Analyze this claim query and return a JSON response:

Query: {request.query}

Return JSON format:
{{
  "decision": "APPROVED/REJECTED/UNDETERMINED",
  "amount": "coverage amount or null",
  "justification": [
    {{
      "criteria": "evaluation criteria",
      "status": "PASSED/FAILED",
      "explanation": "detailed explanation",
      "clause_reference": "policy section"
    }}
  ]
}}

Consider standard insurance factors: waiting periods, pre-existing conditions, age limits, policy duration, medical necessity."""

    messages = [
        {"role": "system", "content": "You are a professional insurance claims processor."},
        {"role": "user", "content": prompt}
    ]
    
    try:
        response = call_llm_api(messages)
        
        # Extract JSON from response
        start = response.find("{")
        end = response.rfind("}") + 1
        if start != -1 and end > start:
            result = json.loads(response[start:end])
            return result
        else:
            return {
                "decision": "UNDETERMINED",
                "justification": "Unable to parse response"
            }
    except Exception as e:
        return {
            "decision": "UNDETERMINED",
            "justification": f"Error: {str(e)}"
        }

@app.post("/api/v1/document-qa")
async def document_qa(request: DocumentQARequest):
    """Answer questions about policy documents"""
    try:
        # Extract PDF content
        pdf_text = extract_pdf_from_url(request.document_url)
        chunks = chunk_text(pdf_text)
        
        answers = []
        for question in request.questions:
            # Find relevant chunks
            relevant_chunks = find_relevant_chunks(question, chunks)
            
            if relevant_chunks:
                context = "\n\n".join(relevant_chunks)
                prompt = f"""Based on the following insurance policy document, answer the question:

Document Content:
{context}

Question: {question}

Provide a specific answer based only on the document content. If not found, say "Information not found in document"."""

                messages = [
                    {"role": "system", "content": "You are an insurance policy analyst."},
                    {"role": "user", "content": prompt}
                ]
                
                answer = call_llm_api(messages, max_tokens=400)
                evidence = relevant_chunks[0][:200] + "..." if relevant_chunks else ""
            else:
                answer = "Information not found in document"
                evidence = ""
            
            answers.append({
                "question": question,
                "answer": answer,
                "evidence_excerpt": evidence
            })
        
        return {"answers": answers}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")