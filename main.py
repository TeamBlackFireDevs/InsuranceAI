from fastapi import FastAPI
from pydantic import BaseModel
import requests
from pdfminer.high_level import extract_text
import tempfile
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Dict, Any
import json

app = FastAPI()

# Initialize the sentence transformer model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

class QARequest(BaseModel):
    document_url: str
    questions: List[str]

class DocumentChunk:
    def __init__(self, text: str, chunk_id: int, page_num: int = None):
        self.text = text
        self.chunk_id = chunk_id
        self.page_num = page_num
        self.embedding = None

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
        return f"Error extracting PDF: {str(e)}"

def chunk_text_intelligently(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[DocumentChunk]:
    """
    Intelligently chunk text preserving sentence boundaries
    """
    chunks = []
    sentences = text.split('. ')
    
    current_chunk = ""
    chunk_id = 0
    
    for sentence in sentences:
        # Add sentence to current chunk
        potential_chunk = current_chunk + sentence + ". "
        
        # If adding this sentence exceeds chunk size, save current chunk and start new
        if len(potential_chunk) > chunk_size and current_chunk:
            chunks.append(DocumentChunk(current_chunk.strip(), chunk_id))
            
            # Start new chunk with overlap from previous chunk
            overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
            current_chunk = overlap_text + sentence + ". "
            chunk_id += 1
        else:
            current_chunk = potential_chunk
    
    # Add the last chunk
    if current_chunk.strip():
        chunks.append(DocumentChunk(current_chunk.strip(), chunk_id))
    
    return chunks

def create_embeddings(chunks: List[DocumentChunk]) -> List[DocumentChunk]:
    """Create embeddings for document chunks"""
    texts = [chunk.text for chunk in chunks]
    embeddings = model.encode(texts)
    
    for i, chunk in enumerate(chunks):
        chunk.embedding = embeddings[i]
    
    return chunks

def build_faiss_index(chunks: List[DocumentChunk]) -> faiss.IndexFlatIP:
    """Build FAISS index for semantic search"""
    embeddings = np.array([chunk.embedding for chunk in chunks]).astype('float32')
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
    index.add(embeddings)
    
    return index

def retrieve_relevant_chunks(query: str, chunks: List[DocumentChunk], index: faiss.IndexFlatIP, top_k: int = 3) -> List[DocumentChunk]:
    """Retrieve most relevant chunks for a query"""
    # Encode query
    query_embedding = model.encode([query]).astype('float32')
    faiss.normalize_L2(query_embedding)
    
    # Search in FAISS index
    scores, indices = index.search(query_embedding, top_k)
    
    # Return relevant chunks
    relevant_chunks = [chunks[idx] for idx in indices[0]]
    return relevant_chunks

def generate_answer_with_context(question: str, relevant_chunks: List[DocumentChunk]) -> Dict[str, Any]:
    """Generate answer using Hugging Face API with retrieved context"""
    
    # Combine relevant chunks into context
    context = "\n\n".join([f"Chunk {chunk.chunk_id}: {chunk.text}" for chunk in relevant_chunks])
    
    # Prepare the prompt
    prompt = f"""Based on the following policy document sections, answer the question accurately and specifically. If the information is not found in the provided sections, state that clearly.

Policy Document Sections:
{context}

Question: {question}

Instructions:
- Answer only based on the information provided in the document sections above
- If the specific information is not found, state "Information not found in the provided document sections"
- If found, quote the relevant part and provide the chunk reference
- Be precise and factual

Answer:"""

    try:
        # Call Hugging Face API (using your existing API setup)
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
            "max_tokens": 500,
            "temperature": 0.1,
            "stream": False
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            answer = result['choices'][0]['message']['content']
            
            return {
                "question": question,
                "answer": answer,
                "evidence_chunks": [
                    {
                        "chunk_id": chunk.chunk_id,
                        "text": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text
                    } for chunk in relevant_chunks
                ],
                "confidence": 0.85
            }
        else:
            return {
                "question": question,
                "answer": f"API Error: {response.status_code}",
                "evidence_chunks": [],
                "confidence": 0.0
            }
            
    except Exception as e:
        return {
            "question": question,
            "answer": f"Processing error: {str(e)}",
            "evidence_chunks": [],
            "confidence": 0.0
        }

@app.post("/api/v1/document-qa")
async def document_qa(req: QARequest):
    """Process document Q&A with RAG implementation"""
    try:
        # Step 1: Extract PDF text
        pdf_text = extract_pdf_from_url(req.document_url)
        
        if pdf_text.startswith("Error"):
            return {"error": pdf_text}
        
        # Step 2: Chunk the document intelligently
        chunks = chunk_text_intelligently(pdf_text, chunk_size=1200, overlap=200)
        
        if not chunks:
            return {"error": "No content could be extracted from the document"}
        
        # Step 3: Create embeddings
        chunks_with_embeddings = create_embeddings(chunks)
        
        # Step 4: Build FAISS index for semantic search
        index = build_faiss_index(chunks_with_embeddings)
        
        # Step 5: Process each question
        answers = []
        for question in req.questions:
            # Retrieve relevant chunks
            relevant_chunks = retrieve_relevant_chunks(question, chunks_with_embeddings, index, top_k=3)
            
            # Generate answer with context
            answer_data = generate_answer_with_context(question, relevant_chunks)
            answers.append(answer_data)
        
        return {
            "answers": answers,
            "document_stats": {
                "total_chunks": len(chunks),
                "total_characters": len(pdf_text),
                "avg_chunk_size": sum(len(chunk.text) for chunk in chunks) // len(chunks)
            }
        }
        
    except Exception as e:
        return {"error": f"Processing failed: {str(e)}"}

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}