# Insurance Claims Processing API - Updated for Vercel Deployment
# Removes heavy ML dependencies while maintaining HuggingFace Qwen model integration

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import tempfile
import os
import json
import re
from typing import List, Dict, Any
from pdfminer.high_level import extract_text

app = FastAPI(title="Insurance Claims Processing API")

class QueryRequest(BaseModel):
    query: str

class QARequest(BaseModel):
    document_url: str
    questions: List[str]

class DocumentChunk:
    def __init__(self, text: str, chunk_id: int, page_num: int = None):
        self.text = text
        self.chunk_id = chunk_id
        self.page_num = page_num

def extract_pdf_from_url(url: str) -> str:
    """Extract text from PDF at given URL"""
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
        raise HTTPException(status_code=400, detail=f"Error extracting PDF: {str(e)}")

def chunk_text_intelligently(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[DocumentChunk]:
    """
    Intelligently chunk text preserving sentence boundaries
    Lightweight version without ML dependencies
    """
    chunks = []
    
    # Split by paragraphs first, then by sentences if needed
    paragraphs = text.split('\n\n')
    current_chunk = ""
    chunk_id = 0
    
    for paragraph in paragraphs:
        # If paragraph is very long, split by sentences
        if len(paragraph) > chunk_size:
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            
            for sentence in sentences:
                potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
                
                if len(potential_chunk) > chunk_size and current_chunk:
                    # Save current chunk
                    chunks.append(DocumentChunk(current_chunk.strip(), chunk_id))
                    chunk_id += 1
                    
                    # Start new chunk with overlap
                    overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else ""
                    current_chunk = overlap_text + " " + sentence
                else:
                    current_chunk = potential_chunk
        else:
            # Add whole paragraph
            potential_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
            
            if len(potential_chunk) > chunk_size and current_chunk:
                # Save current chunk
                chunks.append(DocumentChunk(current_chunk.strip(), chunk_id))
                chunk_id += 1
                
                # Start new chunk with overlap
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else ""
                current_chunk = overlap_text + "\n\n" + paragraph
            else:
                current_chunk = potential_chunk
    
    # Add the final chunk
    if current_chunk.strip():
        chunks.append(DocumentChunk(current_chunk.strip(), chunk_id))
    
    return chunks

def find_relevant_chunks_simple(query: str, chunks: List[DocumentChunk], top_k: int = 3) -> List[DocumentChunk]:
    """
    Find relevant chunks using simple keyword matching and scoring
    Replaces semantic search to avoid heavy ML dependencies
    """
    if not chunks:
        return []
    
    # Convert query to lowercase and extract keywords
    query_words = set(re.findall(r'\b\w+\b', query.lower()))
    
    # Score each chunk based on keyword overlap
    scored_chunks = []
    
    for chunk in chunks:
        chunk_words = set(re.findall(r'\b\w+\b', chunk.text.lower()))
        
        # Calculate overlap score
        overlap = len(query_words.intersection(chunk_words))
        
        # Bonus for exact phrase matches
        phrase_bonus = 0
        query_phrases = [query[i:i+20] for i in range(len(query)-19)]
        for phrase in query_phrases:
            if phrase.lower() in chunk.text.lower():
                phrase_bonus += 2
        
        # Total score
        total_score = overlap + phrase_bonus
        
        if total_score > 0:  # Only include chunks with some relevance
            scored_chunks.append((total_score, chunk))
    
    # Sort by score and return top chunks
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored_chunks[:top_k]]

def call_huggingface_qwen_api(prompt: str, max_tokens: int = 800) -> str:
    """
    Call HuggingFace API with Qwen model - maintaining your existing integration
    """
    try:
        url = "https://api.novita.ai/v3/openai/chat/completions"
        headers = {
            "Authorization": f"Bearer {os.getenv('HF_TOKEN')}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "qwen/Qwen3-Coder-480B-A35B-Instruct:novita",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": 0.1,
            "stream": False
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            return f"API Error: {response.status_code} - {response.text}"
            
    except Exception as e:
        return f"Error calling HuggingFace API: {str(e)}"

def generate_answer_with_context(question: str, relevant_chunks: List[DocumentChunk]) -> Dict[str, Any]:
    """Generate answer using HuggingFace Qwen API with retrieved context"""
    
    # Combine relevant chunks into context
    context = "\n\n".join([f"Document Section {chunk.chunk_id}: {chunk.text}" for chunk in relevant_chunks])
    
    # Prepare the prompt for Qwen model
    prompt = f"""Based on the following insurance policy document sections, answer the question accurately and specifically. If the information is not found in the provided sections, state that clearly.

Insurance Policy Document Sections:
{context}

Question: {question}

Instructions:
- Answer only based on the information provided in the document sections above
- If the specific information is not found, state "Information not found in the provided document sections"  
- If found, quote the relevant part and provide clear details
- Be precise, factual, and cite the document section when possible
- Format your response clearly and professionally

Answer:"""

    # Call HuggingFace Qwen API
    answer = call_huggingface_qwen_api(prompt, max_tokens=500)
    
    return {
        "question": question,
        "answer": answer,
        "evidence_chunks": [
            {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text[:300] + "..." if len(chunk.text) > 300 else chunk.text
            } for chunk in relevant_chunks
        ],
        "confidence": 0.85
    }

def process_insurance_claim(query: str) -> Dict[str, Any]:
    """Process insurance claim using HuggingFace Qwen model - your original functionality"""
    
    prompt = f"""You are an expert insurance claim processing assistant. Analyze the following insurance claim query and provide a structured decision.

Based on typical insurance policy rules (waiting periods, age limits, pre-existing conditions, coverage terms), evaluate this claim:

Claim Query: {query}

Provide your response in this exact JSON format:
{{
  "decision": "APPROVED" or "REJECTED" or "UNDETERMINED",
  "amount": "coverage amount if applicable or null",
  "justification": [
    {{
      "criteria": "evaluation criteria name",
      "status": "PASSED" or "FAILED" or "UNDETERMINED", 
      "explanation": "detailed explanation of this criteria evaluation",
      "clause_reference": "relevant policy section or clause"
    }}
  ]
}}

Instructions:
- Use APPROVED only if all criteria clearly pass
- Use REJECTED if any critical criteria fail  
- Use UNDETERMINED if insufficient information is provided
- Provide detailed explanations for each criteria
- Reference typical insurance policy clauses
- Be thorough and professional

Response:"""

    # Call HuggingFace Qwen API
    response_text = call_huggingface_qwen_api(prompt, max_tokens=800)
    
    # Try to extract JSON from response
    try:
        # Find JSON in the response
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            result = json.loads(json_str)
            return result
        else:
            # Fallback if JSON parsing fails
            return {
                "decision": "UNDETERMINED",
                "amount": None,
                "justification": [
                    {
                        "criteria": "Response Processing",
                        "status": "FAILED",
                        "explanation": f"Could not parse structured response. Raw response: {response_text[:200]}...",
                        "clause_reference": "System Processing Error"
                    }
                ]
            }
    except json.JSONDecodeError:
        return {
            "decision": "UNDETERMINED", 
            "amount": None,
            "justification": [
                {
                    "criteria": "JSON Parsing",
                    "status": "FAILED",
                    "explanation": f"Invalid JSON in response: {response_text[:200]}...",
                    "clause_reference": "System Processing Error"
                }
            ]
        }

# API Endpoints
@app.post("/api/v1/insurance-claim")
async def insurance_claim(req: QueryRequest):
    """Process insurance claim query - your original endpoint"""
    try:
        result = process_insurance_claim(req.query)
        return result
    except Exception as e:
        return {
            "decision": "UNDETERMINED",
            "amount": None, 
            "justification": [
                {
                    "criteria": "System Error",
                    "status": "FAILED",
                    "explanation": f"Processing error: {str(e)}",
                    "clause_reference": "System Error"
                }
            ]
        }

@app.post("/api/v1/document-qa")
async def document_qa(req: QARequest):
    """Process document Q&A with lightweight RAG implementation"""
    try:
        # Step 1: Extract PDF text
        pdf_text = extract_pdf_from_url(req.document_url)
        
        # Step 2: Chunk the document intelligently  
        chunks = chunk_text_intelligently(pdf_text, chunk_size=1200, overlap=200)
        
        if not chunks:
            return {"error": "No content could be extracted from the document"}
        
        # Step 3: Process each question
        answers = []
        for question in req.questions:
            # Find relevant chunks using simple matching
            relevant_chunks = find_relevant_chunks_simple(question, chunks, top_k=3)
            
            # Generate answer with context using HuggingFace Qwen
            answer_data = generate_answer_with_context(question, relevant_chunks)
            answers.append(answer_data)
        
        return {
            "answers": answers,
            "document_stats": {
                "total_chunks": len(chunks),
                "total_characters": len(pdf_text),
                "avg_chunk_size": sum(len(chunk.text) for chunk in chunks) // len(chunks) if chunks else 0
            }
        }
        
    except Exception as e:
        return {"error": f"Processing failed: {str(e)}"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "model": "HuggingFace Qwen3-Coder-480B-A35B-Instruct",
        "dependencies": "lightweight"
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Insurance Claims Processing API", 
        "endpoints": {
            "claims": "/api/v1/insurance-claim",
            "document_qa": "/api/v1/document-qa",
            "health": "/health"
        }
    }