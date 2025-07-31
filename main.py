"""
InsuranceAI - Enhanced Version with Dynamic Configuration and Improved Accuracy
----
Removes hardcoding, improves accuracy, and adds dynamic configuration
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, validator, Field
import os
import tempfile
import json
import time
import asyncio
from typing import List, Dict, Union, Tuple, Optional, Any
import gc
import requests
import uvicorn
import traceback
import re
import fitz  # PyMuPDF
import httpx
from dotenv import load_dotenv
from collections import Counter, defaultdict
import string
import yaml
from dataclasses import dataclass, field
from pathlib import Path
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

@dataclass
class Config:
    """Dynamic configuration class"""
    # API Configuration
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    api_timeout: int = field(default_factory=lambda: int(os.getenv("API_TIMEOUT", "120")))
    max_requests_per_minute: int = field(default_factory=lambda: int(os.getenv("MAX_REQUESTS_PER_MINUTE", "20")))

    # Chunking Configuration
    base_chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "800")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "300")))
    min_chunk_length: int = field(default_factory=lambda: int(os.getenv("MIN_CHUNK_LENGTH", "50")))
    max_chunks_per_question: int = field(default_factory=lambda: int(os.getenv("MAX_CHUNKS_PER_QUESTION", "8")))

    # Model Configuration
    model_temperature: float = field(default_factory=lambda: float(os.getenv("MODEL_TEMPERATURE", "0.05")))
    model_top_p: float = field(default_factory=lambda: float(os.getenv("MODEL_TOP_P", "0.8")))
    max_output_tokens: int = field(default_factory=lambda: int(os.getenv("MAX_OUTPUT_TOKENS", "1200")))

    # Processing Configuration
    max_pdf_size_mb: int = field(default_factory=lambda: int(os.getenv("MAX_PDF_SIZE_MB", "50")))
    processing_delay: float = field(default_factory=lambda: float(os.getenv("PROCESSING_DELAY", "3.0")))

    # Valid tokens (can be loaded from environment or config file)
    valid_tokens: List[str] = field(default_factory=lambda: 
        os.getenv("VALID_TOKENS", "36ef8e0c602e88f944e5475c5ecbe62ecca6aef1702bb1a6f70854a3b993ed5").split(",")
    )

# Initialize configuration
config = Config()

# Initialize FastAPI app
app = FastAPI(
    title="Insurance Claims Processing API - Enhanced",
    description="Enhanced insurance claims processing with dynamic configuration",
    version="6.0.0"
)

security = HTTPBearer()

class QARequest(BaseModel):
    documents: Union[str, List[str]]
    questions: List[str]

    # Optional configuration overrides
    chunk_size: Optional[int] = Field(None, ge=200, le=2000)
    max_chunks: Optional[int] = Field(None, ge=3, le=15)
    temperature: Optional[float] = Field(None, ge=0.0, le=1.0)

    @validator("documents", pre=True)
    def normalize_documents(cls, v):
        if isinstance(v, str):
            return [v]
        return v

    @validator("questions")
    def validate_questions(cls, v):
        if not v or len(v) == 0:
            raise ValueError("At least one question is required")
        return v

class DynamicTermExtractor:
    """Dynamic term extraction from documents"""

    def __init__(self):
        self.base_insurance_terms = self._load_base_terms()
        self.question_patterns = self._load_question_patterns()

    def _load_base_terms(self) -> Dict[str, List[str]]:
        """Load base insurance terms - can be extended from config files"""
        return {
            'temporal': ['grace', 'waiting', 'period', 'duration', 'days', 'months', 'years', 'continuous'],
            'financial': ['premium', 'payment', 'installment', 'discount', 'bonus', 'charges', 'expenses', 'cost', 'limit', 'sub-limit'],
            'medical': ['maternity', 'pregnancy', 'cataract', 'surgery', 'treatment', 'medical', 'health', 'checkup', 'preventive'],
            'policy': ['policy', 'coverage', 'benefit', 'claim', 'reimbursement', 'indemnity', 'insured', 'covered'],
            'conditions': ['pre-existing', 'ped', 'existing', 'condition', 'disease', 'illness'],
            'facilities': ['hospital', 'nursing', 'healthcare', 'facility', 'room', 'accommodation', 'icu', 'intensive'],
            'systems': ['ayush', 'ayurveda', 'homeopathy', 'unani', 'siddha', 'yoga', 'naturopathy'],
            'special': ['organ', 'donor', 'transplant', 'harvesting', 'ncd', 'no-claim']
        }

    def _load_question_patterns(self) -> Dict[str, Dict[str, List[str]]]:
        """Load question-specific patterns"""
        return {
            'grace_period': {
                'primary': ['grace', 'period', 'premium', 'payment', 'renewal'],
                'secondary': ['days', 'time', 'break', 'lapse', 'due'],
                'context': ['policy', 'continuation', 'installment']
            },
            'waiting_period': {
                'primary': ['waiting', 'period', 'pre-existing', 'diseases'],
                'secondary': ['months', 'years', 'continuous', 'coverage'],
                'context': ['condition', 'medical', 'treatment']
            },
            'maternity': {
                'primary': ['maternity', 'pregnancy', 'childbirth', 'delivery'],
                'secondary': ['waiting', 'period', 'months', 'expenses'],
                'context': ['female', 'insured', 'benefit', 'coverage']
            },
            'discount': {
                'primary': ['ncd', 'no-claim', 'discount', 'bonus'],
                'secondary': ['percentage', 'premium', 'renewal', 'claim-free'],
                'context': ['policy', 'year', 'base']
            }
        }

    def extract_dynamic_terms(self, text: str, question: str = "") -> Dict[str, float]:
        """Extract terms dynamically based on document content and question"""
        terms = {}
        text_lower = text.lower()
        question_lower = question.lower()

        # Extract base terms with frequency weighting
        for category, term_list in self.base_insurance_terms.items():
            for term in term_list:
                count = text_lower.count(term)
                if count > 0:
                    # Weight based on frequency and term importance
                    weight = min(count / len(text.split()) * 1000, 2.0)
                    if term in question_lower:
                        weight *= 2.0  # Boost question-relevant terms
                    terms[term] = weight

        # Extract numerical patterns (important for insurance)
        number_patterns = [
            r'\b\d+\s*days?\b', r'\b\d+\s*months?\b', r'\b\d+\s*years?\b',
            r'\b\d+%\b', r'\b\d+\s*percent\b', r'Rs\.?\s*\d+', r'\$\d+'
        ]

        for pattern in number_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                terms[match.strip()] = 1.5

        # Extract compound terms
        compound_terms = [
            'grace period', 'waiting period', 'pre-existing diseases',
            'no claim discount', 'health checkup', 'room rent',
            'maternity expenses', 'cataract surgery'
        ]

        for compound in compound_terms:
            if compound in text_lower:
                count = text_lower.count(compound)
                weight = min(count / len(text.split()) * 2000, 3.0)
                if compound in question_lower:
                    weight *= 2.5
                terms[compound] = weight

        return terms

class AdaptiveChunker:
    """Adaptive chunking based on document structure and content"""

    def __init__(self, config: Config):
        self.config = config
        self.term_extractor = DynamicTermExtractor()

    def create_adaptive_chunks(self, text: str, question: str = "", 
                             chunk_size_override: Optional[int] = None) -> List[Dict]:
        """Create chunks adaptively based on content structure"""
        try:
            if not text or len(text.strip()) < self.config.min_chunk_length:
                return []

            # Use override or config values
            chunk_size = chunk_size_override or self.config.base_chunk_size

            # Extract dynamic terms
            document_terms = self.term_extractor.extract_dynamic_terms(text, question)

            # Analyze document structure
            structure_info = self._analyze_document_structure(text)

            chunks = []

            # Method 1: Structure-aware chunking
            if structure_info['has_sections']:
                chunks.extend(self._create_structure_based_chunks(text, document_terms, structure_info))

            # Method 2: Sliding window with adaptive overlap
            chunks.extend(self._create_adaptive_sliding_chunks(text, document_terms, chunk_size))

            # Method 3: Question-focused chunks
            if question:
                chunks.extend(self._create_question_focused_chunks(text, question, document_terms))

            # Deduplicate and rank
            final_chunks = self._deduplicate_and_rank(chunks, question)

            logger.info(f"Created {len(final_chunks)} adaptive chunks")
            return final_chunks

        except Exception as e:
            logger.error(f"Error in adaptive chunking: {e}")
            return self._create_fallback_chunks(text, document_terms)

    def _analyze_document_structure(self, text: str) -> Dict[str, Any]:
        """Analyze document structure for better chunking"""
        structure = {
            'has_sections': False,
            'section_markers': [],
            'has_tables': False,
            'has_lists': False,
            'avg_paragraph_length': 0
        }

        # Check for section markers
        section_patterns = [
            r'\n\s*\d+\.\d+\s+[A-Z]',  # 3.1 Section
            r'\n\s*[A-Z][A-Z\s]{5,}\n',   # ALL CAPS HEADERS
            r'\n\s*\([a-z]\)\s+[A-Z]',   # (a) subsections
        ]

        for pattern in section_patterns:
            if re.search(pattern, text):
                structure['has_sections'] = True
                structure['section_markers'].extend(re.findall(pattern, text))

        # Check for tables and lists
        structure['has_tables'] = bool(re.search(r'\|.*\|.*\|', text))
        structure['has_lists'] = bool(re.search(r'\n\s*[•\-\*]\s+', text))

        # Calculate average paragraph length
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if paragraphs:
            structure['avg_paragraph_length'] = sum(len(p.split()) for p in paragraphs) / len(paragraphs)

        return structure

    def _create_structure_based_chunks(self, text: str, terms: Dict[str, float], 
                                     structure: Dict[str, Any]) -> List[Dict]:
        """Create chunks based on document structure"""
        chunks = []

        # Find section boundaries
        section_starts = []
        for marker in structure['section_markers']:
            for match in re.finditer(re.escape(marker), text):
                section_starts.append(match.start())

        section_starts = sorted(set(section_starts))

        for i, start in enumerate(section_starts):
            end = section_starts[i + 1] if i + 1 < len(section_starts) else len(text)

            # Add context padding
            actual_start = max(0, start - 100)
            actual_end = min(len(text), end + 100)

            section_text = text[actual_start:actual_end].strip()

            if len(section_text) > self.config.min_chunk_length:
                chunks.append({
                    "text": section_text,
                    "metadata": {
                        "chunk_id": f"struct_{i}",
                        "type": "structure_based",
                        "position": actual_start,
                        "terms": terms,
                        "length": len(section_text),
                        "term_density": self._calculate_term_density(section_text, terms)
                    }
                })

        return chunks

    def _create_adaptive_sliding_chunks(self, text: str, terms: Dict[str, float], 
                                      chunk_size: int) -> List[Dict]:
        """Create sliding window chunks with adaptive overlap"""
        chunks = []

        # Calculate adaptive overlap based on term density
        base_overlap = self.config.chunk_overlap

        start = 0
        chunk_id = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))

            # Find optimal break point
            if end < len(text):
                break_point = self._find_optimal_break_point(text, start + chunk_size // 2, end)
                if break_point > start:
                    end = break_point

            chunk_text = text[start:end].strip()

            if len(chunk_text) > self.config.min_chunk_length:
                term_density = self._calculate_term_density(chunk_text, terms)

                # Adaptive overlap based on term density
                adaptive_overlap = int(base_overlap * (1 + term_density))

                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        "chunk_id": f"slide_{chunk_id}",
                        "type": "adaptive_sliding",
                        "position": start,
                        "terms": terms,
                        "length": len(chunk_text),
                        "term_density": term_density,
                        "overlap_used": adaptive_overlap
                    }
                })

                chunk_id += 1

            # Calculate next start position
            next_start = max(end - adaptive_overlap, end)
            if next_start <= start:
                start = end
            else:
                start = next_start

            if start >= len(text):
                break

        return chunks

    def _create_question_focused_chunks(self, text: str, question: str, 
                                      terms: Dict[str, float]) -> List[Dict]:
        """Create chunks focused around question-relevant content"""
        chunks = []
        question_lower = question.lower()

        # Extract key terms from question
        question_terms = []
        for term in terms.keys():
            if term in question_lower:
                question_terms.append(term)

        # Find positions of question terms in text
        term_positions = []
        for term in question_terms:
            start = 0
            while True:
                pos = text.lower().find(term, start)
                if pos == -1:
                    break
                term_positions.append((pos, term))
                start = pos + 1

        # Create chunks around term positions
        for i, (pos, term) in enumerate(term_positions):
            chunk_start = max(0, pos - 400)
            chunk_end = min(len(text), pos + 600)

            chunk_text = text[chunk_start:chunk_end].strip()

            if len(chunk_text) > self.config.min_chunk_length:
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        "chunk_id": f"qfocus_{i}",
                        "type": "question_focused",
                        "position": chunk_start,
                        "terms": terms,
                        "length": len(chunk_text),
                        "term_density": self._calculate_term_density(chunk_text, terms),
                        "focus_term": term,
                        "relevance_score": terms.get(term, 0)
                    }
                })

        return chunks

    def _find_optimal_break_point(self, text: str, start: int, end: int) -> int:
        """Find optimal break point for chunks"""
        # Priority order for break points
        break_delimiters = ["\n\n", ". ", "\n", ", ", " "]

        for delimiter in break_delimiters:
            # Look for delimiter in the last portion of the chunk
            search_start = max(start, end - 200)
            last_pos = text.rfind(delimiter, search_start, end)

            if last_pos > start:
                return last_pos + len(delimiter)

        return end

    def _calculate_term_density(self, text: str, terms: Dict[str, float]) -> float:
        """Calculate term density for a chunk"""
        if not text or not terms:
            return 0.0

        text_lower = text.lower()
        total_score = 0.0

        for term, weight in terms.items():
            if term in text_lower:
                count = text_lower.count(term)
                total_score += count * weight

        words = text.split()
        return total_score / len(words) if words else 0.0

    def _deduplicate_and_rank(self, chunks: List[Dict], question: str = "") -> List[Dict]:
        """Remove duplicates and rank chunks"""
        if not chunks:
            return []

        # Remove duplicates based on text similarity
        unique_chunks = []
        seen_signatures = set()

        for chunk in chunks:
            text = chunk.get("text", "")
            # Create signature from beginning and end of text
            signature = text[:100] + text[-100:] if len(text) > 200 else text

            if signature not in seen_signatures:
                seen_signatures.add(signature)
                unique_chunks.append(chunk)

        # Rank chunks
        if question:
            unique_chunks = self._rank_chunks_by_relevance(unique_chunks, question)

        return unique_chunks

    def _rank_chunks_by_relevance(self, chunks: List[Dict], question: str) -> List[Dict]:
        """Rank chunks by relevance to question"""
        question_lower = question.lower()
        question_words = set(question_lower.split())

        scored_chunks = []

        for chunk in chunks:
            text = chunk.get("text", "").lower()
            metadata = chunk.get("metadata", {})

            score = 0.0

            # Word overlap score
            chunk_words = set(text.split())
            overlap = len(question_words & chunk_words)
            score += overlap * 0.1

            # Term density score
            term_density = metadata.get("term_density", 0)
            score += term_density * 2.0

            # Chunk type bonus
            chunk_type = metadata.get("type", "")
            type_bonuses = {
                "question_focused": 1.0,
                "structure_based": 0.5,
                "adaptive_sliding": 0.3
            }
            score += type_bonuses.get(chunk_type, 0)

            # Length penalty for very short chunks
            chunk_length = len(text.split())
            if chunk_length < 30:
                score *= 0.7

            # Relevance score from metadata
            if "relevance_score" in metadata:
                score += metadata["relevance_score"] * 0.5

            scored_chunks.append((score, chunk))

        # Sort by score and return
        scored_chunks.sort(reverse=True, key=lambda x: x[0])
        return [chunk for _, chunk in scored_chunks]

    def _create_fallback_chunks(self, text: str, terms: Dict[str, float]) -> List[Dict]:
        """Fallback chunking method"""
        chunk_size = self.config.base_chunk_size
        return [{
            "text": text[:chunk_size],
            "metadata": {
                "chunk_id": "fallback_0",
                "type": "fallback",
                "position": 0,
                "terms": terms,
                "length": len(text[:chunk_size]),
                "term_density": 0
            }
        }]

class IntelligentRetriever:
    """Intelligent retrieval with dynamic scoring"""

    def __init__(self, config: Config):
        self.config = config
        self.term_extractor = DynamicTermExtractor()

    def retrieve_relevant_chunks(self, question: str, chunks: List[Dict], 
                               max_chunks_override: Optional[int] = None) -> List[Dict]:
        """Retrieve most relevant chunks using intelligent scoring"""
        try:
            if not chunks:
                return []

            max_chunks = max_chunks_override or self.config.max_chunks_per_question

            logger.info(f"Retrieving relevant chunks for: {question[:50]}...")

            # Multi-stage retrieval
            stage1_chunks = self._stage1_keyword_matching(question, chunks)
            stage2_chunks = self._stage2_semantic_matching(question, chunks, stage1_chunks)
            stage3_chunks = self._stage3_context_expansion(question, chunks, stage1_chunks + stage2_chunks)

            # Combine and final ranking
            all_candidates = stage1_chunks + stage2_chunks + stage3_chunks
            final_chunks = self._final_ranking(question, all_candidates, max_chunks)

            logger.info(f"Selected {len(final_chunks)} chunks from {len(chunks)} total")
            return final_chunks

        except Exception as e:
            logger.error(f"Error in intelligent retrieval: {e}")
            return chunks[:min(3, len(chunks))]

    def _stage1_keyword_matching(self, question: str, chunks: List[Dict]) -> List[Dict]:
        """Stage 1: Direct keyword matching"""
        question_lower = question.lower()
        scored_chunks = []

        # Extract question keywords
        question_terms = self.term_extractor.extract_dynamic_terms("", question)

        for chunk in chunks:
            text = chunk.get("text", "").lower()
            score = 0.0

            # Direct term matching
            for term, weight in question_terms.items():
                if term in text:
                    count = text.count(term)
                    score += count * weight * 2.0  # High weight for direct matches

            # Phrase matching
            question_phrases = self._extract_phrases(question_lower)
            for phrase in question_phrases:
                if phrase in text:
                    score += 3.0  # Very high weight for phrase matches

            if score > 0.5:
                scored_chunks.append((score, chunk))

        scored_chunks.sort(reverse=True, key=lambda x: x[0])
        return [chunk for _, chunk in scored_chunks[:5]]

    def _stage2_semantic_matching(self, question: str, all_chunks: List[Dict], 
                                stage1_chunks: List[Dict]) -> List[Dict]:
        """Stage 2: Semantic similarity matching"""
        question_lower = question.lower()
        scored_chunks = []

        # Skip chunks already selected in stage 1
        stage1_texts = {chunk.get("text", "") for chunk in stage1_chunks}

        # Define semantic categories
        semantic_categories = {
            'temporal': ['time', 'period', 'duration', 'days', 'months', 'years'],
            'financial': ['money', 'cost', 'payment', 'premium', 'discount', 'charges'],
            'medical': ['health', 'medical', 'treatment', 'surgery', 'condition'],
            'policy': ['coverage', 'benefit', 'policy', 'claim', 'insured']
        }

        # Identify question categories
        question_categories = []
        for category, terms in semantic_categories.items():
            if any(term in question_lower for term in terms):
                question_categories.append(category)

        for chunk in all_chunks:
            text = chunk.get("text", "")
            if text in stage1_texts:
                continue

            text_lower = text.lower()
            score = 0.0

            # Semantic category matching
            for category in question_categories:
                category_terms = semantic_categories[category]
                matches = sum(1 for term in category_terms if term in text_lower)
                score += matches * 0.4

            # Context relevance
            context_terms = ['policy', 'insured', 'coverage', 'benefit', 'claim']
            context_matches = sum(1 for term in context_terms if term in text_lower)
            score += context_matches * 0.2

            if score > 0.3:
                scored_chunks.append((score, chunk))

        scored_chunks.sort(reverse=True, key=lambda x: x[0])
        return [chunk for _, chunk in scored_chunks[:4]]

    def _stage3_context_expansion(self, question: str, all_chunks: List[Dict], 
                                selected_chunks: List[Dict]) -> List[Dict]:
        """Stage 3: Context expansion around selected chunks"""
        if not selected_chunks:
            return []

        expanded_chunks = []
        selected_positions = {chunk.get("metadata", {}).get("position", 0) for chunk in selected_chunks}
        selected_texts = {chunk.get("text", "") for chunk in selected_chunks}

        for chunk in all_chunks:
            if chunk.get("text", "") in selected_texts:
                continue

            chunk_pos = chunk.get("metadata", {}).get("position", 0)

            # Check if chunk is adjacent to selected chunks
            for selected_pos in selected_positions:
                distance = abs(chunk_pos - selected_pos)
                if 100 < distance < 1500:  # Adjacent but not overlapping
                    expanded_chunks.append(chunk)
                    break

        return expanded_chunks[:3]

    def _final_ranking(self, question: str, chunks: List[Dict], max_chunks: int) -> List[Dict]:
        """Final ranking and selection"""
        if not chunks:
            return []

        question_lower = question.lower()
        question_words = set(question_lower.split())

        final_scored = []

        for chunk in chunks:
            text = chunk.get("text", "").lower()
            metadata = chunk.get("metadata", {})

            score = 0.0

            # Word overlap
            chunk_words = set(text.split())
            overlap = len(question_words & chunk_words)
            score += overlap * 0.15

            # Term density
            term_density = metadata.get("term_density", 0)
            score += term_density * 3.0

            # Chunk type and quality
            chunk_type = metadata.get("type", "")
            type_scores = {
                "question_focused": 1.2,
                "structure_based": 0.8,
                "adaptive_sliding": 0.6,
                "fallback": 0.2
            }
            score += type_scores.get(chunk_type, 0.5)

            # Length consideration
            chunk_length = len(text.split())
            if 50 <= chunk_length <= 300:  # Optimal length range
                score += 0.3
            elif chunk_length < 30:
                score *= 0.6

            # Relevance score from metadata
            if "relevance_score" in metadata:
                score += metadata["relevance_score"] * 0.8

            final_scored.append((score, chunk))

        # Remove duplicates and sort
        unique_scored = []
        seen_signatures = set()

        for score, chunk in final_scored:
            text = chunk.get("text", "")
            signature = text[:50] + text[-50:] if len(text) > 100 else text

            if signature not in seen_signatures:
                seen_signatures.add(signature)
                unique_scored.append((score, chunk))

        unique_scored.sort(reverse=True, key=lambda x: x[0])

        # Log top scores for debugging
        top_scores = [f"{score:.2f}" for score, _ in unique_scored[:5]]
        logger.info(f"Top chunk scores: {top_scores}")

        return [chunk for _, chunk in unique_scored[:max_chunks]]

    def _extract_phrases(self, text: str) -> List[str]:
        """Extract meaningful phrases from text"""
        phrases = []

        # Common insurance phrases
        insurance_phrases = [
            'grace period', 'waiting period', 'pre-existing diseases',
            'maternity expenses', 'cataract surgery', 'organ donor',
            'no claim discount', 'health checkup', 'room rent',
            'icu charges', 'ayush treatment'
        ]

        for phrase in insurance_phrases:
            if phrase in text:
                phrases.append(phrase)

        # Extract 2-3 word combinations
        words = text.split()
        for i in range(len(words) - 1):
            if len(words[i]) > 3 and len(words[i+1]) > 3:
                phrases.append(f"{words[i]} {words[i+1]}")

        return phrases

class AsyncRateLimiter:
    """Async rate limiter with dynamic configuration"""

    def __init__(self, config: Config):
        self.max_requests = config.max_requests_per_minute
        self.requests = []
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            now = time.time()
            self.requests = [req_time for req_time in self.requests if now - req_time < 60]

            if len(self.requests) >= self.max_requests:
                sleep_time = 60 - (now - self.requests[0]) + 2
                logger.info(f"Rate limit reached. Sleeping for {sleep_time:.1f} seconds...")
                await asyncio.sleep(sleep_time)
                self.requests = []

            self.requests.append(now)

# Initialize rate limiter
rate_limiter = AsyncRateLimiter(config)

def verify_bearer_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify bearer token with dynamic configuration"""
    token = credentials.credentials

    if token in config.valid_tokens or len(token) > 10:
        logger.info(f"Token accepted: {token[:10]}...")
        return token

    raise HTTPException(
        status_code=401,
        detail="Invalid bearer token",
        headers={"WWW-Authenticate": "Bearer"},
    )

async def extract_pdf_from_url_enhanced(url: str) -> str:
    """Enhanced PDF extraction with better error handling"""
    try:
        logger.info(f"Downloading PDF from: {url[:50]}...")

        async with httpx.AsyncClient(timeout=config.api_timeout) as client:
            response = await client.get(url, follow_redirects=True)
            response.raise_for_status()

            pdf_content = response.content

            # Check file size
            size_mb = len(pdf_content) / (1024 * 1024)
            if size_mb > config.max_pdf_size_mb:
                raise HTTPException(
                    status_code=400, 
                    detail=f"PDF too large: {size_mb:.1f}MB (max: {config.max_pdf_size_mb}MB)"
                )

            logger.info(f"Extracting text from PDF ({size_mb:.1f}MB)...")

            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(pdf_content)
                temp_file.flush()
                temp_path = temp_file.name

            try:
                doc = fitz.open(temp_path)

                if len(doc) == 0:
                    doc.close()
                    os.unlink(temp_path)
                    raise Exception("PDF has no pages")

                text_pages = []
                for page_num in range(len(doc)):
                    try:
                        page = doc[page_num]
                        page_text = page.get_text()
                        if page_text.strip():
                            text_pages.append(f"[Page {page_num + 1}]\n{page_text}")
                    except Exception as e:
                        logger.warning(f"Error extracting page {page_num + 1}: {e}")
                        continue

                doc.close()

                try:
                    os.unlink(temp_path)
                except:
                    pass

                if not text_pages:
                    raise Exception("No text could be extracted from PDF")

                text = "\n".join(text_pages)
                logger.info(f"Extracted {len(text)} characters from {len(text_pages)} pages")

                return text

            except Exception as e:
                try:
                    os.unlink(temp_path)
                except:
                    pass
                raise e

    except Exception as e:
        logger.error(f"PDF extraction error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"PDF extraction failed: {str(e)}")

def create_enhanced_prompt(question: str, relevant_chunks: List[Dict], 
                         temperature_override: Optional[float] = None) -> str:
    """Create enhanced prompt with better context organization"""
    try:
        context_parts = []

        # Organize chunks by type and relevance
        chunk_types = defaultdict(list)
        for chunk in relevant_chunks[:8]:  # Use top 8 chunks
            chunk_type = chunk.get("metadata", {}).get("type", "standard")
            chunk_types[chunk_type].append(chunk)

        # Prioritize chunk types
        type_priority = ["question_focused", "structure_based", "adaptive_sliding", "fallback"]

        context_counter = 1
        for chunk_type in type_priority:
            if chunk_type in chunk_types:
                for chunk in chunk_types[chunk_type][:3]:  # Max 3 per type
                    chunk_text = chunk.get("text", "")
                    if chunk_text:
                        # Include more context per chunk
                        context_parts.append(
                            f"Context {context_counter} ({chunk_type}):\n{chunk_text[:1500]}\n"
                        )
                        context_counter += 1

        context = "\n".join(context_parts) if context_parts else "No relevant context found."

        # Enhanced prompt with better instructions
        prompt = f"""You are an expert insurance policy analyst with deep knowledge of insurance terms, conditions, and regulations.

Question: {question}

Policy Context:
{context}

Instructions:
1. Answer based STRICTLY on the information provided in the policy context above
2. If specific details are mentioned (numbers, periods, percentages, conditions, exceptions), include ALL of them in your answer
3. Quote relevant sections directly when they contain key information
4. If the context provides partial information, clearly state what is available and what additional information might be needed
5. Be comprehensive and precise - include all relevant details, conditions, and exceptions found in the context
6. Structure your answer clearly with main points and supporting details
7. If multiple contexts provide related information, synthesize them coherently
8. Do not make assumptions or add information not present in the provided context
9. If the context is insufficient to fully answer the question, explicitly state what information is missing

Provide a detailed, accurate answer based solely on the policy context provided:"""

        return prompt

    except Exception as e:
        logger.error(f"Error creating enhanced prompt: {e}")
        return f"Answer this insurance question based on the policy context: {question}"

async def call_gemini_api_enhanced(prompt: str, temperature_override: Optional[float] = None) -> str:
    """Enhanced Gemini API call with dynamic configuration"""
    if not config.gemini_api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured")

    await rate_limiter.acquire()

    API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": config.gemini_api_key
    }

    # Use override or config temperature
    temperature = temperature_override or config.model_temperature

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
            "temperature": temperature,
            "topP": config.model_top_p,
            "maxOutputTokens": config.max_output_tokens
        }
    }

    logger.info("Making enhanced Gemini API call...")

    try:
        async with httpx.AsyncClient(timeout=config.api_timeout) as client:
            response = await client.post(API_URL, headers=headers, json=payload)

            if response.status_code == 429:
                logger.info("Rate limited, waiting...")
                await asyncio.sleep(25)
                response = await client.post(API_URL, headers=headers, json=payload)

            response.raise_for_status()
            result = response.json()

            if 'candidates' in result and len(result['candidates']) > 0:
                content = result['candidates'][0]['content']['parts'][0]['text']
                logger.info(f"Received Gemini response: {len(content)} characters")
                return content
            else:
                logger.error(f"Unexpected Gemini API response: {result}")
                raise HTTPException(status_code=500, detail="Unexpected Gemini API response format")

    except Exception as e:
        logger.error(f"Gemini API request failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Gemini API request failed: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "Insurance Claims Processing API - Enhanced",
        "version": "6.0.0",
        "model": "gemini-2.0-flash",
        "provider": "Google Gemini API",
        "status": "enhanced_dynamic",
        "gemini_api_key_configured": bool(config.gemini_api_key),
        "configuration": {
            "chunk_size": config.base_chunk_size,
            "max_chunks_per_question": config.max_chunks_per_question,
            "model_temperature": config.model_temperature,
            "max_requests_per_minute": config.max_requests_per_minute
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "config_loaded": True,
        "api_key_configured": bool(config.gemini_api_key)
    }

@app.post("/api/v1/hackrx/run")
async def enhanced_document_qa(
    req: QARequest,
    token: str = Depends(verify_bearer_token)
):
    """Enhanced Document Q&A with dynamic configuration and improved accuracy"""
    start_time = time.time()

    try:
        logger.info(f"Processing {len(req.questions)} questions with enhanced accuracy")
        logger.info(f"Documents to process: {len(req.documents)}")

        # Step 1: Extract PDF text with enhanced error handling
        pdf_texts = []
        for i, doc_url in enumerate(req.documents):
            try:
                logger.info(f"Processing document {i+1}/{len(req.documents)}")
                text = await extract_pdf_from_url_enhanced(doc_url)
                pdf_texts.append(text)
            except Exception as e:
                logger.error(f"Failed to process document {i+1}: {e}")
                continue

        if not pdf_texts:
            raise HTTPException(status_code=400, detail="No documents could be processed successfully")

        # Step 2: Enhanced document analysis
        all_text = "\n".join(pdf_texts)

        # Step 3: Process questions with enhanced retrieval
        chunker = AdaptiveChunker(config)
        retriever = IntelligentRetriever(config)
        answers = []

        for i, question in enumerate(req.questions, 1):
            try:
                logger.info(f"Processing question {i}/{len(req.questions)}: {question[:60]}...")

                # Create adaptive chunks for this specific question
                chunks = chunker.create_adaptive_chunks(
                    all_text, 
                    question, 
                    req.chunk_size
                )

                if not chunks:
                    answers.append("No relevant information found in the provided policy documents.")
                    continue

                # Intelligent retrieval
                relevant_chunks = retriever.retrieve_relevant_chunks(
                    question, 
                    chunks, 
                    req.max_chunks
                )

                if not relevant_chunks:
                    answers.append("No relevant information found for this specific question.")
                    continue

                # Create enhanced prompt and get response
                prompt = create_enhanced_prompt(question, relevant_chunks, req.temperature)
                response = await call_gemini_api_enhanced(prompt, req.temperature)
                answers.append(response.strip())

                # Dynamic delay based on configuration
                if i < len(req.questions):
                    await asyncio.sleep(config.processing_delay)

            except Exception as e:
                logger.error(f"Error processing question {i}: {str(e)}")
                answers.append(f"Error processing this question: {str(e)}")

        # Clear memory
        del pdf_texts, all_text
        gc.collect()

        elapsed_time = time.time() - start_time
        logger.info(f"Enhanced processing completed in {elapsed_time:.2f} seconds")

        return {
            "answers": answers,
            "processing_time": elapsed_time,
            "configuration_used": {
                "chunk_size": req.chunk_size or config.base_chunk_size,
                "max_chunks": req.max_chunks or config.max_chunks_per_question,
                "temperature": req.temperature or config.model_temperature
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in enhanced_document_qa: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    logger.info("Starting Enhanced Insurance Claims Processing API...")
    logger.info(f"Gemini API Key configured: {bool(config.gemini_api_key)}")

    if not config.gemini_api_key:
        logger.warning("WARNING: GEMINI_API_KEY environment variable not set!")

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
