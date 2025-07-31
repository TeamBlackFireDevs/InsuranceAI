import os
import json
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, request, jsonify
import PyPDF2
import io
import re
import numpy as np
from collections import Counter
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Gemini API configuration
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Training data for insurance domain
TRAINING_PATTERNS = {
    "policy_terms": [
        "grace period", "waiting period", "pre-existing conditions", "sum insured",
        "deductible", "co-payment", "room rent", "ICU charges", "daycare procedures"
    ],
    "coverage_types": [
        "inpatient", "outpatient", "daycare", "AYUSH", "domiciliary", "maternity",
        "dental", "optical", "preventive health check"
    ],
    "exclusions": [
        "suicide", "war", "nuclear risks", "cosmetic surgery", "experimental treatment",
        "self-inflicted injury", "alcohol", "drugs"
    ],
    "claim_process": [
        "cashless", "reimbursement", "pre-authorization", "claim settlement",
        "network hospital", "TPA", "claim documents"
    ]
}

SAMPLE_QA_PATTERNS = [
    {
        "pattern": "grace period",
        "response_format": "The grace period is [X days/months] from the due date of premium payment."
    },
    {
        "pattern": "hospital definition",
        "response_format": "A hospital is defined as [specific definition from policy]."
    },
    {
        "pattern": "room rent coverage",
        "response_format": "Room rent is covered up to [amount/percentage] as per the plan selected."
    }
]

def call_gemini_api(prompt):
    """Call Gemini API with specified configuration"""
    API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={GEMINI_API_KEY}"
    
    headers = {
        "Content-Type": "application/json"
    }

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "topP": 0.8,
            "maxOutputTokens": 1500,
            "candidateCount": 1
        },
        "safetySettings": [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            }
        ]
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=25)
        response.raise_for_status()
        
        result = response.json()
        if 'candidates' in result and len(result['candidates']) > 0:
            return result['candidates'][0]['content']['parts'][0]['text']
        else:
            return "No response generated"
            
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {e}")
        return "Error: API request failed"
    except Exception as e:
        logger.error(f"Error processing response: {e}")
        return "Error: Failed to process response"

def extract_text_from_pdf(pdf_content):
    """Extract text from PDF with improved parsing"""
    try:
        pdf_file = io.BytesIO(pdf_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        text = ""
        for page_num, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                # Clean and normalize text
                page_text = re.sub(r'\s+', ' ', page_text)
                page_text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]]', ' ', page_text)
                text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
            except Exception as e:
                logger.warning(f"Error extracting text from page {page_num + 1}: {e}")
                continue
        
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        return ""

def smart_chunk_text(text, chunk_size=800, overlap=200):
    """Create intelligent chunks with semantic boundaries"""
    if not text:
        return []
    
    # Split by sections first (looking for headers, numbered items, etc.)
    section_patterns = [
        r'\n\d+\.\s+[A-Z][^.]*\n',  # Numbered sections
        r'\n[A-Z][A-Z\s]{10,}\n',   # ALL CAPS headers
        r'\n[A-Z][a-z\s]{5,}:\s*\n', # Title: format
        r'\n---[^-]*---\n'          # Page breaks
    ]
    
    chunks = []
    current_chunk = ""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                # Keep overlap
                words = current_chunk.split()
                overlap_text = ' '.join(words[-overlap//5:]) if len(words) > overlap//5 else ""
                current_chunk = overlap_text + " " + sentence
            else:
                current_chunk = sentence
        else:
            current_chunk += " " + sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return [chunk for chunk in chunks if len(chunk.strip()) > 50]

def calculate_keyword_similarity(query, text):
    """Calculate similarity based on keyword matching"""
    query_words = set(re.findall(r'\b\w+\b', query.lower()))
    text_words = set(re.findall(r'\b\w+\b', text.lower()))
    
    if not query_words:
        return 0.0
    
    # Exact word matches
    exact_matches = len(query_words.intersection(text_words))
    exact_score = exact_matches / len(query_words)
    
    # Partial matches (substring matching)
    partial_matches = 0
    for q_word in query_words:
        for t_word in text_words:
            if len(q_word) > 3 and (q_word in t_word or t_word in q_word):
                partial_matches += 0.5
                break
    
    partial_score = min(partial_matches / len(query_words), 0.5)
    
    return exact_score + partial_score

def calculate_phrase_similarity(query, text):
    """Calculate similarity based on phrase matching"""
    query_lower = query.lower()
    text_lower = text.lower()
    
    # Extract phrases (2-4 words)
    query_phrases = []
    text_phrases = []
    
    query_words = query_lower.split()
    text_words = text_lower.split()
    
    # Generate 2-3 word phrases
    for i in range(len(query_words) - 1):
        query_phrases.append(' '.join(query_words[i:i+2]))
        if i < len(query_words) - 2:
            query_phrases.append(' '.join(query_words[i:i+3]))
    
    for i in range(len(text_words) - 1):
        text_phrases.append(' '.join(text_words[i:i+2]))
        if i < len(text_words) - 2:
            text_phrases.append(' '.join(text_words[i:i+3]))
    
    if not query_phrases:
        return 0.0
    
    matches = 0
    for q_phrase in query_phrases:
        if any(q_phrase in t_phrase or t_phrase in q_phrase for t_phrase in text_phrases):
            matches += 1
    
    return matches / len(query_phrases)

def enhanced_keyword_search(query, chunks, top_k=5):
    """Enhanced keyword-based search with multiple scoring methods"""
    if not chunks:
        return []
    
    try:
        scored_chunks = []
        
        for chunk in chunks:
            # Keyword similarity
            keyword_score = calculate_keyword_similarity(query, chunk)
            
            # Phrase similarity
            phrase_score = calculate_phrase_similarity(query, chunk)
            
            # Insurance domain boost
            domain_score = 0
            chunk_lower = chunk.lower()
            for category, terms in TRAINING_PATTERNS.items():
                for term in terms:
                    if term.lower() in chunk_lower:
                        domain_score += 0.1
            domain_score = min(domain_score, 1.0)
            
            # Position boost (earlier chunks might be more important)
            position_score = 1.0 / (chunks.index(chunk) + 1) * 0.1
            
            # Length normalization (prefer chunks with reasonable length)
            length_score = min(len(chunk) / 1000, 1.0) * 0.1
            
            # Combined scoring
            combined_score = (
                keyword_score * 0.4 +
                phrase_score * 0.3 +
                domain_score * 0.2 +
                position_score * 0.05 +
                length_score * 0.05
            )
            
            scored_chunks.append({
                'text': chunk,
                'score': combined_score,
                'keyword_score': keyword_score,
                'phrase_score': phrase_score,
                'domain_score': domain_score
            })
        
        # Sort by score and return top k
        scored_chunks.sort(key=lambda x: x['score'], reverse=True)
        
        return [chunk for chunk in scored_chunks[:top_k] if chunk['score'] > 0.1]
        
    except Exception as e:
        logger.error(f"Error in keyword search: {e}")
        return []

def enhance_query_with_training(query):
    """Enhance query with insurance domain knowledge"""
    enhanced_query = query
    
    # Add context based on training patterns
    query_lower = query.lower()
    
    for category, terms in TRAINING_PATTERNS.items():
        for term in terms:
            if term.lower() in query_lower:
                enhanced_query += f" (Related to {category}: {term})"
                break
    
    return enhanced_query

def create_enhanced_prompt(query, relevant_chunks, training_context=True):
    """Create an enhanced prompt with training context"""
    
    base_prompt = f"""You are an expert insurance policy analyst. Answer the following question based ONLY on the provided policy document content.

IMPORTANT INSTRUCTIONS:
1. Answer based ONLY on the information provided in the context below
2. If the information is not available in the context, clearly state "This information is not available in the provided policy document"
3. Be specific and cite relevant sections when possible
4. For numerical values, amounts, or percentages, quote them exactly as mentioned
5. Use clear, professional language suitable for insurance customers

"""

    if training_context:
        base_prompt += """
INSURANCE DOMAIN GUIDELINES:
- Grace periods are typically mentioned in premium payment sections
- Coverage details are usually in benefits/coverage sections
- Exclusions are listed in separate exclusion sections
- Definitions are typically at the beginning or in a glossary section
- Room rent and ICU charges are in benefit schedules or tables

"""

    context_text = "\n\n".join([
        f"RELEVANT SECTION {i+1} (Score: {chunk['score']:.3f}):\n{chunk['text']}"
        for i, chunk in enumerate(relevant_chunks[:3])
    ])
    
    full_prompt = f"""{base_prompt}

POLICY DOCUMENT CONTEXT:
{context_text}

QUESTION: {query}

ANSWER:"""
    
    return full_prompt

def process_single_query(query, all_chunks):
    """Process a single query with enhanced retrieval"""
    try:
        start_time = time.time()
        
        # Enhance query
        enhanced_query = enhance_query_with_training(query)
        
        # Get relevant chunks using keyword search
        relevant_chunks = enhanced_keyword_search(enhanced_query, all_chunks, top_k=5)
        
        if not relevant_chunks:
            return {
                'query': query,
                'answer': 'No relevant information found in the policy document.',
                'confidence': 0.0,
                'processing_time': time.time() - start_time
            }
        
        # Create prompt
        prompt = create_enhanced_prompt(query, relevant_chunks)
        
        # Get answer from Gemini
        answer = call_gemini_api(prompt)
        
        # Calculate confidence based on chunk scores
        avg_score = sum(chunk['score'] for chunk in relevant_chunks) / len(relevant_chunks)
        confidence = min(avg_score * 1.5, 1.0)  # Scale confidence
        
        processing_time = time.time() - start_time
        
        return {
            'query': query,
            'answer': answer,
            'confidence': confidence,
            'processing_time': processing_time,
            'relevant_chunks_count': len(relevant_chunks),
            'top_chunk_score': relevant_chunks[0]['score'] if relevant_chunks else 0
        }
        
    except Exception as e:
        logger.error(f"Error processing query '{query}': {e}")
        return {
            'query': query,
            'answer': f'Error processing query: {str(e)}',
            'confidence': 0.0,
            'processing_time': time.time() - start_time if 'start_time' in locals() else 0
        }

@app.route('/analyze', methods=['POST'])
def analyze_document():
    """Main endpoint for document analysis"""
    start_time = time.time()
    
    try:
        # Get file and queries
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        queries = request.form.get('queries', '[]')
        try:
            queries = json.loads(queries)
        except json.JSONDecodeError:
            return jsonify({'error': 'Invalid queries format'}), 400
        
        if not queries:
            return jsonify({'error': 'No queries provided'}), 400
        
        # Extract text from PDF
        logger.info("Extracting text from PDF...")
        pdf_content = file.read()
        text = extract_text_from_pdf(pdf_content)
        
        if not text:
            return jsonify({'error': 'Could not extract text from PDF'}), 400
        
        # Create chunks
        logger.info("Creating text chunks...")
        chunks = smart_chunk_text(text, chunk_size=800, overlap=200)
        
        if not chunks:
            return jsonify({'error': 'Could not create text chunks'}), 400
        
        logger.info(f"Created {len(chunks)} chunks")
        
        # Process queries in parallel
        logger.info(f"Processing {len(queries)} queries...")
        results = []
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_query = {
                executor.submit(process_single_query, query, chunks): query 
                for query in queries
            }
            
            for future in as_completed(future_to_query):
                try:
                    result = future.result(timeout=25)  # 25 second timeout per query
                    results.append(result)
                except Exception as e:
                    query = future_to_query[future]
                    logger.error(f"Query '{query}' failed: {e}")
                    results.append({
                        'query': query,
                        'answer': f'Error: {str(e)}',
                        'confidence': 0.0,
                        'processing_time': 0
                    })
        
        # Sort results by original query order
        query_to_result = {r['query']: r for r in results}
        ordered_results = [query_to_result.get(q, {
            'query': q,
            'answer': 'Error: Query not processed',
            'confidence': 0.0,
            'processing_time': 0
        }) for q in queries]
        
        total_time = time.time() - start_time
        
        # Calculate overall statistics
        avg_confidence = sum(r['confidence'] for r in ordered_results) / len(ordered_results)
        successful_queries = len([r for r in ordered_results if not r['answer'].startswith('Error')])
        
        response = {
            'results': ordered_results,
            'summary': {
                'total_queries': len(queries),
                'successful_queries': successful_queries,
                'average_confidence': avg_confidence,
                'total_processing_time': total_time,
                'chunks_created': len(chunks),
                'text_length': len(text)
            }
        }
        
        logger.info(f"Analysis completed in {total_time:.2f} seconds")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in analyze_document: {e}")
        return jsonify({
            'error': f'Internal server error: {str(e)}',
            'processing_time': time.time() - start_time
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'gemini_api_configured': GEMINI_API_KEY is not None
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)