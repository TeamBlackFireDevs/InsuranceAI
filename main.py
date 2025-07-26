from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import os
import tempfile
import json
from pdfminer.high_level import extract_text
from typing import List, Optional

app = FastAPI()

# Hugging Face API configuration
HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = "https://router.huggingface.co/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

class QueryRequest(BaseModel):
    query: str

class QARequest(BaseModel):
    document_url: str
    questions: List[str]

class DocumentQAResponse(BaseModel):
    answers: List[dict]

def extract_pdf_from_url(url: str) -> str:
    """Download PDF from URL and extract text content"""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name
        
        try:
            text = extract_text(tmp_path)
            return text
        finally:
            os.remove(tmp_path)
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to extract PDF: {str(e)}")

def chunk_text(text: str, max_length: int = 2000) -> List[str]:
    """Split text into manageable chunks"""
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        if len(current_chunk + para) <= max_length:
            current_chunk += para + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def query_llm(prompt: str) -> str:
    """Send query to Hugging Face LLM and get response"""
    payload = {
        "model": "Qwen/Qwen3-Coder-480B-A35B-Instruct:novita",
        "messages": [
            {"role": "system", "content": "You are a helpful insurance claim and document analysis assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 500
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        response_data = response.json()
        return response_data["choices"][0]["message"]["content"]
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM query failed: {str(e)}")

def parse_llm_json_response(content: str) -> dict:
    """Parse JSON from LLM response, handling markdown code blocks"""
    try:
        # Remove markdown code blocks if present
        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            json_str = content[start:end].strip()
        elif "```" in content:
            start = content.find("```") + 3
            end = content.find("```", start)
            json_str = content[start:end].strip()
        else:
            # Try to find JSON object
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                json_str = content[start:end]
            else:
                json_str = content
        
        return json.loads(json_str)
    
    except Exception as e:
        return {"error": "Failed to parse JSON", "raw_content": content}

@app.post("/api/v1/insurance-claim")
async def process_claim(req: QueryRequest):
    """Original insurance claim processing endpoint"""
    prompt = (
        "You are an insurance claim processing assistant. "
        "Given the customer query, analyze and provide a decision. "
        "Return your response as valid JSON with the following structure:\n\n"
        "{\n"
        '  "decision": "APPROVED/REJECTED/UNDETERMINED",\n'
        '  "amount": "coverage amount or null",\n'
        '  "justification": [\n'
        '    {\n'
        '      "criteria": "evaluation criteria",\n'
        '      "status": "PASSED/FAILED",\n'
        '      "explanation": "detailed explanation",\n'
        '      "clause_reference": "policy section reference"\n'
        '    }\n'
        '  ]\n'
        "}\n\n"
        f"Customer Query: {req.query}\n\n"
        "Provide a structured JSON response analyzing this claim."
    )
    
    try:
        llm_response = query_llm(prompt)
        parsed_response = parse_llm_json_response(llm_response)
        
        if "error" in parsed_response:
            return {"result": llm_response, "parsed": False}
        
        return parsed_response
    
    except Exception as e:
        return {"error": str(e), "query": req.query}

@app.post("/api/v1/document-qa")
async def document_qa(req: QARequest):
    """Enhanced document Q&A endpoint for processing policy documents"""
    try:
        # Extract text from PDF
        pdf_text = extract_pdf_from_url(req.document_url)
        
        if not pdf_text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from the document")
        
        # Chunk the document for better processing
        chunks = chunk_text(pdf_text, max_length=3000)
        
        # Process each question
        answers = []
        
        for question in req.questions:
            # For each question, use the most relevant chunk or full document
            # In a production system, you'd use semantic search here
            relevant_text = chunks[0] if chunks else pdf_text[:3000]  # Use first chunk as primary context
            
            prompt = (
                f"Document Content:\n{relevant_text}\n\n"
                f"Question: {question}\n\n"
                "Instructions:\n"
                "- Answer the question based ONLY on the document content provided above\n"
                "- Be specific and cite relevant sections when possible\n"
                "- If the information is not in the document, state 'Information not found in document'\n"
                "- Provide a clear, concise answer\n\n"
                "Answer:"
            )
            
            try:
                answer = query_llm(prompt)
                
                answers.append({
                    "question": question,
                    "answer": answer.strip(),
                    "evidence_excerpt": relevant_text[:200] + "..." if len(relevant_text) > 200 else relevant_text,
                    "confidence": 0.85  # Static confidence for MVP
                })
                
            except Exception as e:
                answers.append({
                    "question": question,
                    "answer": f"Error processing question: {str(e)}",
                    "evidence_excerpt": "",
                    "confidence": 0.0
                })
        
        return {"answers": answers}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Insurance Claims Processing API",
        "endpoints": [
            "/api/v1/insurance-claim - Process insurance claims",
            "/api/v1/document-qa - Document Q&A processing"
        ],
        "status": "active"
    }

@app.get("/health")
async def health_check():
    """Health check for monitoring"""
    return {"status": "healthy", "api": "insurance-claims"}