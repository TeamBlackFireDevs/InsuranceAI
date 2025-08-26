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
from flask import Flask, request, jsonify, render_template, send_from_directory
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

app = Flask(__name__, static_folder="build/static", template_folder="build")

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
    ],
    "grace_period": [
        "grace period means", "grace period for payment"
    ]
}

def call_openai_api(prompt, max_retries=3, max_tokens=350):  
    for attempt in range(max_retries):  
        try:  
            response = client.chat.completions.create(  
                model="gemini-2.5-flash",  
                messages=[{"role": "user", "content": prompt}],  
                temperature=0.02,  
                top_p=0.6,  
                max_tokens=max_tokens
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

def optimized_chunk_text(text, chunk_size=1500, overlap=200):
    """
    Splits text into chunks of ~chunk_size with overlap.
    Works even if PDF text has no clean paragraph breaks.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap  # slide window with overlap
        if start < 0:
            start = 0
    return chunks


def get_query_type(query):
    q = (query or "").lower()
    # Advice/opinion triggers
    if any(t in q for t in [
        "is this policy good", "is this good", "is it good", "worth", "worth it",
        "should i buy", "recommend", "recommendation", "advice", "pros", "cons",
        "compare", "better", "suitable", "fit for me"
    ]):
        return "advice"
    # Specific factual categories
    if "grace period" in q:
        return "grace_period"
    if "hospital" in q and ("definition" in q or "define" in q or "means" in q):
        return "hospital_definition"
    elif "waiting period" in q:
        return "waiting_periods"
    elif any(term in q for term in ["room rent", "icu", "charges"]):
        return "room_rent"
    elif "maternity" in q:
        return "maternity"
    else:
        return "general"

    
def normalize_query(query: str) -> str:
    q = (query or "").strip()
    if not q:
        return q
    # Expand 1–2 word queries into a question
    if len(q.split()) <= 2:
        return f"What does the policy say about {q}?"
    return q

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

def create_advice_prompt(user_question, relevant_chunks):
    """
    Produces a two-part response:
    1) Answer: strictly from policy content in the provided context.
    2) Advisor Note: practical guidance based on (1), using general good practice, but do NOT invent policy facts.
    """
    base = (
        "You are an insurance advisor. Use only the POLICY CONTENT for facts.\n\n"
        "Write the answer in two parts:\n"
        "Answer: (1–3 sentences, factual, numbers/limits as stated)\n"
        "Advisor Note: (1–3 sentences, practical guidance or implications for a buyer; "
        "do not add policy facts not present in content; keep it neutral and non-salesy)\n\n"
        "If the content does not contain the requested facts, say exactly: "
        "\"Information not available in the policy.\" for the Answer part, and keep Advisor Note generic and brief.\n"
    )
    context_text = "\n\n".join([chunk['text'] for chunk in relevant_chunks[:8]])
    return f"""{base}
POLICY CONTENT:
{context_text}

USER QUESTION:
{user_question}

Now write:
Answer:
Advisor Note:"""


def clean_answer_optimized(answer, query):
    if answer is None or not isinstance(answer, str):
        return "Information not available in the policy."
    answer = answer.strip()
    if not answer:
        return "Information not available in the policy."

    # Flatten whitespace
    answer = re.sub(r'\n+', ' ', answer)
    answer = re.sub(r'\s+', ' ', answer).strip()

    # Normalize common fallback phrasings to your standard
    FALLBACKS = [
        "This information is not available in the provided policy document",
        "Information not available in the policy",
        "Information not available in the policy."
    ]
    for fb in FALLBACKS:
        if fb.lower() in answer.lower() and len(answer) > len(fb) + 50:
            # If the model added a long preface before fallback, keep the preface
            before = answer.lower().split(fb.lower())[0].strip()
            if len(before) > 50:
                return before
            return "Information not available in the policy."

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
    chunks = optimized_chunk_text(text, chunk_size=1500, overlap=200)
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

            original_query = query
            normalized = normalize_query(original_query)
            query_type = get_query_type(original_query)

            # Pre-filter chunks by keywords to reduce search space
            filtered_chunks = prefilter_chunks_by_keywords(chunks, normalized)

            # Use cached embeddings for filtered chunks (map once to avoid O(n^2) .index lookups)
            chunk_to_idx = {c: i for i, c in enumerate(chunks)}
            indices = [chunk_to_idx.get(c) for c in filtered_chunks]
            indices = [i for i in indices if i is not None]
            filtered_embeddings = cached_embeddings[indices]

            # Embed normalized query (more robust for short queries)
            query_embedding = embedder.encode([normalized])
            similarities = cosine_similarity(query_embedding, filtered_embeddings)[0]

            # Score and select top 5 chunks (same signal mixing as your current code)
            scored_chunks = []
            query_words = set(normalized.lower().split())
            query_lower = normalized.lower()

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
                qt = query_type
                if qt in QUERY_SPECIFIC_PATTERNS:
                    for pattern in QUERY_SPECIFIC_PATTERNS[qt]:
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
            top_chunks = [{'text': c[0], 'score': float(c[1])} for c in scored_chunks[:5] if c[1] > 0.05]

            if not top_chunks:
                return {
                    'query': original_query,
                    'answer': 'Information not available in the policy.',
                    'confidence': 0.0,
                    'processing_time': time.time() - start
                }

            # ---- Prompt selection: advice vs factual ----
            if query_type == "advice":
                prompt = create_advice_prompt(original_query, top_chunks)
                max_tokens = 500
            else:
                prompt = create_simple_direct_prompt(original_query, top_chunks)
                max_tokens = 350

            llm_start = time.time()
            answer = call_openai_api(prompt, max_tokens=max_tokens)
            logger.info(f"LLM API call completed in {time.time() - llm_start:.2f}s")

            answer = clean_answer_optimized(answer, original_query)
            avg_score = float(np.mean([c['score'] for c in top_chunks])) if top_chunks else 0.0
            confidence = min(avg_score * 1.2, 1.0)

            return {
                'query': original_query,
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

@app.route("/")
def index():
    return render_template("index.html")   # serves React app

UPLOAD_FOLDER = "/tmp/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    file_url = request.host_url + "uploads/" + file.filename
    return jsonify({"url": file_url})

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    app.run(host='0.0.0.0', port=port, debug=False)