from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import tempfile
import json
import time
import asyncio
from typing import List, Dict, Any
from pdfminer.high_level import extract_text
import gc

import openai

# Set OpenAI API key (must be set in your deployment/environment variables)
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI(title="Insurance Claims Processing API - GPT-4")

class QueryRequest(BaseModel):
    query: str

class QARequest(BaseModel):
    document_url: str
    questions: List[str]

class RateLimiter:
    def __init__(self, max_requests_per_minute=15):
        self.max_requests = max_requests_per_minute
        self.requests = []

    async def wait_if_needed(self):
        now = time.time()
        self.requests = [req_time for req_time in self.requests if now - req_time < 60]
        if len(self.requests) >= self.max_requests:
            sleep_time = 60 - (now - self.requests[0]) + 1
            print(f"Rate limit reached. Sleeping for {sleep_time:.1f} seconds...")
            await asyncio.sleep(sleep_time)
            self.requests = []
        self.requests.append(now)

rate_limiter = RateLimiter()

async def extract_pdf_from_url_optimized(url: str) -> str:
    try:
        print("Downloading PDF...")
        import requests
        response = requests.get(url, timeout=30, stream=True)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    tmp_file.write(chunk)
            tmp_path = tmp_file.name
        try:
            print("Extracting text from PDF...")
            text = extract_text(tmp_path)
            os.unlink(tmp_path)
            gc.collect()
            return text
        except Exception as e:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise e
    except Exception as e:
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

def find_relevant_chunks_optimized(question: str, chunks: List[str], top_k: int = 2) -> List[str]:
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

async def call_openai_api(messages: List[Dict], max_tokens: int = 1000, max_retries: int = 3) -> str:
    if not openai.api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY environment variable not set")
    await rate_limiter.wait_if_needed()
    for attempt in range(max_retries):
        try:
            print(f"OpenAI API call attempt {attempt + 1}/{max_retries}")
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.1,
                n=1,
                stop=None
            )
            content = response.choices[0].message.content
            return content
        except openai.error.RateLimitError:
            wait_time = 2 ** attempt * 10
            print(f"Rate limit hit. Waiting {wait_time}s before retry...")
            await asyncio.sleep(wait_time)
            continue
        except openai.error.Timeout:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt * 5
                print(f"Timeout occurred. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
                continue
            raise HTTPException(status_code=504, detail="API timeout after retries")
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt * 5
                print(f"Request failed: {str(e)}. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
                continue
            raise HTTPException(status_code=500, detail=f"API request failed: {str(e)}")
    raise HTTPException(status_code=500, detail="Max retries exceeded")

async def process_questions_in_batches(questions: List[str], relevant_chunks_map: Dict[str, List[str]]) -> List[Dict]:
    results = []
    batch_size = 3
    for i in range(0, len(questions), batch_size):
        batch_questions = questions[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1}: questions {i + 1}-{min(i + batch_size, len(questions))}")
        batch_context = ""
        batch_prompt = "You are an insurance policy expert. Answer the following questions based on the policy document sections provided.\n\n"
        for j, question in enumerate(batch_questions):
            relevant_chunks = relevant_chunks_map.get(question, [])
            if relevant_chunks:
                batch_context += f"\n--- Context for Question {j + 1} ---\n"
                batch_context += "\n".join(relevant_chunks[:2])
            batch_prompt += f"\nQuestion {j + 1}: {question}\n"
        batch_prompt += f"\nPolicy Document Sections:\n{batch_context}\n\n"
        batch_prompt += "Instructions:\n"
        batch_prompt += "- Answer each question separately and clearly\n"
        batch_prompt += "- Number your answers (Answer 1:, Answer 2:, etc.)\n"
        batch_prompt += "- Base answers ONLY on the provided policy sections\n"
        batch_prompt += "- If information is not found, state 'Information not found in provided sections'\n"
        batch_prompt += "- Be specific and cite relevant policy sections when possible\n"
        messages = [
            {"role": "system", "content": "You are a helpful insurance policy analyst."},
            {"role": "user", "content": batch_prompt}
        ]
        try:
            batch_response = await call_openai_api(messages, max_tokens=1500)
            answers = parse_batch_response(batch_response, len(batch_questions))
            for j, question in enumerate(batch_questions):
                answer = answers[j] if j < len(answers) else "Error processing this question"
                evidence = "\n".join(relevant_chunks_map.get(question, [])[:1])
                results.append({
                    "question": question,
                    "answer": answer,
                    "evidence_excerpt": evidence[:300] + "..." if len(evidence) > 300 else evidence,
                    "confidence": 0.85
                })
        except Exception as e:
            print(f"Error processing batch {i // batch_size + 1}: {str(e)}")
            for question in batch_questions:
                results.append({
                    "question": question,
                    "answer": f"Error processing question: {str(e)}",
                    "evidence_excerpt": "",
                    "confidence": 0.0
                })
        if i + batch_size < len(questions):
            await asyncio.sleep(2)
    return results

def parse_batch_response(response: str, expected_count: int) -> List[str]:
    answers = []
    import re
    answer_pattern = r'Answer\s+(\d+):\s*(.+?)(?=Answer\s+\d+:|$)'
    matches = re.findall(answer_pattern, response, re.DOTALL | re.IGNORECASE)
    if matches:
        matches.sort(key=lambda x: int(x[0]))
        answers = [match[1].strip() for match in matches]
    else:
        parts = response.split('\n\n')
        answers = [part.strip() for part in parts if part.strip()]
    while len(answers) < expected_count:
        answers.append("Answer not found in response")
    return answers[:expected_count]

async def query_llm_for_claim(user_text: str) -> Dict[str, Any]:
    prompt = (
        "You are an insurance claim processing assistant. "
        "Analyze the following claim and provide a structured decision.\n\n"
        "Return your response as JSON with this exact format:\n"
        "{\n"
        '  "decision": "APPROVED/REJECTED/UNDETERMINED",\n'
        '  "amount": "coverage details",\n'
        '  "justification": [\n'
        '    {\n'
        '      "criteria": "evaluation criteria",\n'
        '      "status": "PASSED/FAILED",\n'
        '      "explanation": "detailed reasoning",\n'
        '      "clause_reference": "policy section"\n'
        '    }\n'
        "  ]\n"
        "}\n\n"
        f"Claim Query: {user_text}\n"
        "Provide detailed analysis based on typical health insurance policy terms."
    )
    messages = [
        {"role": "system", "content": "You are a helpful insurance claim processor."},
        {"role": "user", "content": prompt}
    ]
    try:
        content = await call_openai_api(messages, max_tokens=800)
        start = content.find("{")
        end = content.rfind("}") + 1
        if start != -1 and end > start:
            result_json = json.loads(content[start:end])
            return result_json
        else:
            return {
                "decision": "UNDETERMINED",
                "justification": "Unable to parse response format"
            }
    except Exception as e:
        return {
            "decision": "UNDETERMINED", 
            "justification": f"Error processing query: {str(e)}"
        }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "model": "gpt-4",
        "provider": "OpenAI",
        "optimizations": ["batch_processing", "rate_limiting", "memory_optimization"]
    }

@app.post("/api/v1/insurance-claim")
async def process_claim(req: QueryRequest):
    result = await query_llm_for_claim(req.query)
    return result

@app.post("/api/v1/document-qa")
async def document_qa_optimized(req: QARequest):
    try:
        print(f"Processing {len(req.questions)} questions...")
        pdf_text = await extract_pdf_from_url_optimized(req.document_url)
        print(f"Extracted {len(pdf_text)} characters from PDF")
        chunks = chunk_text_memory_efficient(pdf_text, chunk_size=1200, overlap=200)
        print(f"Created {len(chunks)} text chunks")
        del pdf_text
        gc.collect()
        relevant_chunks_map = {}
        for question in req.questions:
            relevant_chunks_map[question] = find_relevant_chunks_optimized(question, chunks, top_k=2)
        responses = await process_questions_in_batches(req.questions, relevant_chunks_map)
        return {"answers": responses}
    except Exception as e:
        print(f"Error in document_qa_optimized: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
