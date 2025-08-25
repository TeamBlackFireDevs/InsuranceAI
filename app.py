import os
os.environ['HF_HOME'] = '/tmp'
os.environ['TRANSFORMERS_CACHE'] = '/tmp'
os.environ['SENTENCE_TRANSFORMERS_HOME'] = '/tmp'
import json
import requests
from urllib.parse import urlparse, unquote
import openai
from openai import OpenAI
import mimetypes
from docx import Document
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, request, jsonify
import pdfplumber
import io
import re
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

client = OpenAI(api_key=GEMINI_API_KEY, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

try:
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("Sentence transformer model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load sentence transformer: {e}")
    embedder = None

# In-memory cache for embeddings keyed by document URL
embedding_cache = {}
embedding_cache_lock = threading.Lock()

TRAINING_PATTERNS = {
    "numerical_terms": [
        "days", "months", "years", "percent", "%", "beds", "lakhs",
        "thirty", "twenty", "fifteen", "ten", "36", "24", "30"
    ],
    "definition_terms": [
        "means", "defined as", "definition", "shall mean", "refers to",
        "institution", "establishment", "criteria", "requirements"
    ],
    "medical_terms": [
        "hospital", "hospitalization", "treatment", "surgery", "procedure",
        "qualified nursing", "operation theatre", "ICU", "emergency"
    ]
}

QUERY_SPECIFIC_PATTERNS = {
    "hospital_definition": [
        "hospital means", "institution established", "inpatient beds",
        "qualified nursing staff", "operation theatre", "minimum criteria"
    ],
    "waiting_periods": [
        "waiting period", "months of continuous coverage", "cataract", "24 months"
    ],
    "room_rent": [
        "room rent", "ICU charges", "percentage of sum insured", "1% of sum insured"
    ]
}

def call_openai_api(prompt, max_retries=3):  
    for attempt in range(max_retries):  
        try:  
            response = client.chat.completions.create(  
                model="gemini-2.5-flash",  
                messages=[{"role": "user", "content": prompt}],  
                temperature=0.02,  
                top_p=0.6,  
                max_tokens=350,  
                n=1,  
                stop=None
            )  
            if response.choices and len(response.choices) > 0:  
                return response.choices[0].message.content  
            else:  
                logging.warning("No response generated from OpenAI API")  
                return "No response generated"  
        except openai.APIError as e:  
            logging.warning(f"OpenAI API request failed on attempt {attempt+1}: {e}")  
            time.sleep(2 ** attempt)  # exponential backoff  
        except Exception as e:  
            logging.warning(f"Unexpected error on attempt {attempt+1}: {e}")  
            time.sleep(2 ** attempt)  
    return "Error: OpenAI API request failed after retries"

def extract_text_from_pdf(pdf_content):  
    try:  
        with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:  
            text = ''  
            for page in pdf.pages:  
                page_text = page.extract_text()  
                if page_text:  
                    text += page_text + '\n'  
        return text.strip()  
    except Exception as e:  
        logger.error(f"Error extracting text from PDF with pdfplumber: {e}")  
        return ""
    
def extract_text_from_docx(docx_content):  
    try:  
        doc = Document(io.BytesIO(docx_content))  
        full_text = []  
        for para in doc.paragraphs:  
            full_text.append(para.text)  
        return '\n'.join(full_text).strip()  
    except Exception as e:  
        logger.error(f"Error extracting text from DOCX: {e}")  
        return ""

def optimized_chunk_text(text, chunk_size=3000, overlap=300):
    if not text:
        return []

    chunks = []
    paragraphs = text.split('\n\n')

    current_chunk = ""
    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) > chunk_size:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            words = current_chunk.split()
            overlap_words = min(overlap // 10, len(words) // 3)
            current_chunk = ' '.join(words[-overlap_words:]) + " " + paragraph if overlap_words > 0 else paragraph
        else:
            current_chunk += "\n\n" + paragraph if current_chunk else paragraph

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return [chunk for chunk in chunks if len(chunk.strip()) > 30]

def get_query_type(query):
    query_lower = query.lower()
    if "hospital" in query_lower and ("definition" in query_lower or "define" in query_lower or "means" in query_lower):
        return "hospital_definition"
    elif "waiting period" in query_lower:
        return "waiting_periods"
    elif any(term in query_lower for term in ["room rent", "icu", "charges"]):
        return "room_rent"
    elif "maternity" in query_lower:
        return "maternity"
    else:
        return "general"

def prefilter_chunks_by_keywords(chunks, query):
    # Quick filter: keep chunks containing any keyword from query or training patterns
    query_lower = query.lower()
    keywords = set(query_lower.split())
    filtered = []
    for chunk in chunks:
        chunk_lower = chunk.lower()
        if any(k in chunk_lower for k in keywords):
            filtered.append(chunk)
            continue
        # Also check training patterns for numerical or definition terms
        if any(term in chunk_lower for term in TRAINING_PATTERNS["numerical_terms"] + TRAINING_PATTERNS["definition_terms"]):
            filtered.append(chunk)
    return filtered if filtered else chunks  # fallback to all if none matched

def fast_similarity_search(query, chunks, top_k=8):
    if not embedder or not chunks:
        return []

    try:
        # Pre-filter chunks to reduce embedding calls
        filtered_chunks = prefilter_chunks_by_keywords(chunks, query)
        query_embedding = embedder.encode([query])
        chunk_embeddings = embedder.encode(filtered_chunks)
        similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]

        query_words = set(query.lower().split())
        query_lower = query.lower()
        query_type = get_query_type(query)

        combined_scores = []
        for i, chunk in enumerate(filtered_chunks):
            chunk_lower = chunk.lower()
            chunk_words = set(chunk_lower.split())

            keyword_overlap = len(query_words.intersection(chunk_words))
            keyword_score = keyword_overlap / len(query_words) if query_words else 0

            numerical_score = 0
            for num_term in TRAINING_PATTERNS["numerical_terms"]:
                if num_term.lower() in chunk_lower:
                    numerical_score += 0.1
                if num_term.lower() in query_lower:
                    numerical_score += 0.2
                    break

            specific_score = 0
            if query_type in QUERY_SPECIFIC_PATTERNS:
                for pattern in QUERY_SPECIFIC_PATTERNS[query_type]:
                    if pattern.lower() in chunk_lower:
                        specific_score += 0.4
                        break

            if query_type == "hospital_definition":
                if "hospital means" in chunk_lower or "minimum criteria" in chunk_lower:
                    specific_score += 0.5
                combined_score = similarities[i] * 0.3 + keyword_score * 0.2 + specific_score * 0.5
            else:
                combined_score = similarities[i] * 0.5 + keyword_score * 0.3 + numerical_score * 0.1 + specific_score * 0.1

            combined_scores.append(combined_score)

        top_indices = np.argsort(combined_scores)[-top_k:][::-1]

        return [
            {
                'text': filtered_chunks[i],
                'score': float(combined_scores[i])
            }
            for i in top_indices if combined_scores[i] > 0.05
        ]

    except Exception as e:
        logger.error(f"Error in similarity search: {e}")
        return []

def create_simple_direct_prompt(query, relevant_chunks):
    base_prompt = (
        "You are an expert insurance policy analyst. Answer the following question based ONLY on the provided policy content.\n\n"
        "Instructions:\n"
        "- Provide extremely concise and direct answers. Prioritize the main answer.\n"
        "- Include only the most critical details, numbers, and conditions.\n"
        "- Absolutely avoid any unnecessary repetition or lengthy explanations.\n"
        "- Use simple, easy-to-understand language.\n"
        "- Do not add information not present in the policy content.\n"
        "- If information is missing, say \"Information not available in the policy.\"\n\n"
    )

    context_text = "\n\n".join([chunk['text'] for chunk in relevant_chunks[:8]])

    return f"""{base_prompt}
POLICY CONTENT:
{context_text}

QUESTION:
{query}

Answer:"""

def clean_answer_optimized(answer, query):
    answer = answer.strip()
    answer = re.sub(r'\n+', ' ', answer)
    answer = re.sub(r'\s+', ' ', answer)

    if "This information is not available in the provided policy document" in answer:
        content_before = answer.split("This information is not available")[0].strip()
        if len(content_before) > 50:
            answer = content_before
            logger.info("Removed fallback message as content was found")

    return answer

AUTH_TOKEN = "36ef8e0c602e88f944e5475c5ecbe62ecca6aef1702bb1a6f70854a3b7993ed5"

def get_filename_from_url(url):  
    path = urlparse(url).path  # e.g. '/assets/Test%20/Mediclaim%20Insurance%20Policy.docx'  
    filename = os.path.basename(path)  # e.g. 'Mediclaim%20Insurance%20Policy.docx'  
    return unquote(filename)  # e.g. 'Mediclaim Insurance Policy.docx'

@app.route('/api/v1/hackrx/run', methods=['POST'])
def analyze_document_json():
    total_start_time = time.time()

    auth_header = request.headers.get('Authorization', '')
    if not auth_header.startswith('Bearer '):
        return jsonify({'error': 'Missing or invalid Authorization header'}), 401

    token = auth_header.split(' ')[1]
    if token != AUTH_TOKEN:
        return jsonify({'error': 'Invalid token'}), 403

    data = request.get_json(force=True)
    pdf_url = data.get('documents')
    queries = data.get('questions', [])

    logger.info(f"Received request with document URL: {pdf_url}")
    logger.info(f"Questions: {queries}")

    if not pdf_url or not queries:
        return jsonify({'error': 'Missing documents URL or questions'}), 400

    try:
        start_time = time.time()
        pdf_response = requests.get(pdf_url, timeout=20)
        pdf_response.raise_for_status()
        pdf_content = pdf_response.content
        logger.info(f"Downloaded PDF in {time.time() - start_time:.2f}s, size: {len(pdf_content)/1024:.2f} KB")
    except Exception as e:
        return jsonify({'error': f'Failed to download PDF: {e}'}), 400
    
    start_time = time.time()
    text = ""
    filename = get_filename_from_url(pdf_url)  
    if filename.lower().endswith('.pdf'):  
        text = extract_text_from_pdf(pdf_content)  
    elif filename.lower().endswith('.docx'):  
        text = extract_text_from_docx(pdf_content)  
    else:  
        return jsonify({'error': 'Unsupported document type'}), 400

    #text = extract_text_from_pdf(pdf_content)
    if not text:
        return jsonify({'error': 'Could not extract text from PDF'}), 400
    logger.info(f"Extracted text in {time.time() - start_time:.2f}s, length: {len(text)} chars")

    start_time = time.time()
    chunks = optimized_chunk_text(text, chunk_size=3000, overlap=300)
    if not chunks:
        return jsonify({'error': 'Could not create text chunks'}), 400
    logger.info(f"Created {len(chunks)} chunks for processing in {time.time() - start_time:.2f}s")

    # Cache embeddings per document URL
    with embedding_cache_lock:
        if pdf_url in embedding_cache:
            logger.info("Using cached embeddings for document")
            cached_embeddings = embedding_cache[pdf_url]
        else:
            logger.info("Computing embeddings for document chunks")
            cached_embeddings = embedder.encode(chunks)
            embedding_cache[pdf_url] = cached_embeddings

    results = []
    max_workers = min(2, len(queries))

    def process_query(query):
        try:
            start = time.time()
            # Pre-filter chunks by keywords to reduce search space
            filtered_chunks = prefilter_chunks_by_keywords(chunks, query)
            # Use cached embeddings for filtered chunks
            indices = [chunks.index(c) for c in filtered_chunks]
            filtered_embeddings = cached_embeddings[indices]
            query_embedding = embedder.encode([query])
            similarities = cosine_similarity(query_embedding, filtered_embeddings)[0]

            # Score and select top 5 chunks
            scored_chunks = []
            query_words = set(query.lower().split())
            query_lower = query.lower()
            query_type = get_query_type(query)
            for i, chunk in enumerate(filtered_chunks):
                chunk_lower = chunk.lower()
                chunk_words = set(chunk_lower.split())
                keyword_overlap = len(query_words.intersection(chunk_words))
                keyword_score = keyword_overlap / len(query_words) if query_words else 0
                numerical_score = 0
                for num_term in TRAINING_PATTERNS["numerical_terms"]:
                    if num_term.lower() in chunk_lower:
                        numerical_score += 0.1
                    if num_term.lower() in query_lower:
                        numerical_score += 0.2
                        break
                specific_score = 0
                if query_type in QUERY_SPECIFIC_PATTERNS:
                    for pattern in QUERY_SPECIFIC_PATTERNS[query_type]:
                        if pattern.lower() in chunk_lower:
                            specific_score += 0.4
                            break
                if query_type == "hospital_definition":
                    if "hospital means" in chunk_lower or "minimum criteria" in chunk_lower:
                        specific_score += 0.5
                    combined_score = similarities[i] * 0.3 + keyword_score * 0.2 + specific_score * 0.5
                else:
                    combined_score = similarities[i] * 0.5 + keyword_score * 0.3 + numerical_score * 0.1 + specific_score * 0.1
                scored_chunks.append((chunk, combined_score))
            scored_chunks.sort(key=lambda x: x[1], reverse=True)
            top_chunks = [{'text': c[0], 'score': c[1]} for c in scored_chunks[:5]]

            if not top_chunks:
                return {
                    'query': query,
                    'answer': 'Information not available in the policy.',
                    'confidence': 0.0,
                    'processing_time': time.time() - start
                }

            prompt = create_simple_direct_prompt(query, top_chunks)
            start_time = time.time()
            answer = call_openai_api(prompt)
            logger.info(f"LLM API call completed in {time.time() - start_time:.2f}s")
            answer = clean_answer_optimized(answer, query)
            avg_score = np.mean([chunk['score'] for chunk in top_chunks])
            confidence = min(avg_score * 1.2, 1.0)

            return {
                'query': query,
                'answer': answer,
                'confidence': confidence,
                'processing_time': time.time() - start
            }
        except Exception as e:
            logger.error(f"Error processing query '{query}': {e}")
            return {
                'query': query,
                'answer': f'Error processing query: {str(e)}',
                'confidence': 0.0,
                'processing_time': 0
            }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_query, q): q for q in queries}
        for future in as_completed(futures):
            results.append(future.result())

    # Maintain original question order
    query_to_result = {r['query']: r for r in results}
    ordered_results = [query_to_result.get(q, {
        'query': q,
        'answer': 'Error: Query not processed',
        'confidence': 0.0,
        'processing_time': 0
    }) for q in queries]

    answers = [r['answer'] for r in ordered_results]

    total_time = time.time() - total_start_time
    logger.info(f"Processed {len(queries)} queries in {total_time:.2f}s")
    logger.info(f"Answers: {answers}")

    return jsonify({"answers": answers})

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'embedder_loaded': embedder is not None,
        'llm_api_configured': GEMINI_API_KEY is not None,
        'version': '3.0_aggressive_optimized',
        'features': [
            'embedding_caching',
            'chunk_size_increase',
            'prefilter_chunks',
            'reduced_top_k',
            'retry_llm_api',
            'threadpool_parallelism'
        ]
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    app.run(host='0.0.0.0', port=port, debug=False)