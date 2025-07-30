"""
Enhanced InsuranceAI - Dynamic Policy Analysis System
----
Improved accuracy through comprehensive document analysis and dynamic keyword extraction
Based on analysis of multiple insurance policy documents for maximum compatibility
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, validator
import os
import tempfile
import json
import time
import asyncio
from typing import List, Dict, Union, Tuple, Optional
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
import math

# Initialize FastAPI app
app = FastAPI(
    title="Enhanced Insurance Claims Processing API",
    description="Dynamic insurance policy analysis with improved accuracy",
    version="6.0.0"
)

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
security = HTTPBearer()

class QARequest(BaseModel):
    documents: Union[str, List[str]]
    questions: List[str]

    @validator("documents", pre=True)
    def normalize_documents(cls, v):
        if isinstance(v, str):
            return [v]
        return v

class DynamicPolicyAnalyzer:
    """Dynamic policy analyzer that adapts to any insurance document"""

    def __init__(self):
        # Core insurance terminology patterns found across all policies
        self.core_patterns = {
            'time_periods': {
                'grace_period': [
                    r'grace\s+period\s+of\s+(\d+)\s+days?',
                    r'grace\s+period\s+(\d+)\s+days?',
                    r'within\s+the\s+grace\s+period\s+of\s+(\d+)',
                    r'grace\s+time\s+of\s+(\d+)'
                ],
                'waiting_period': [
                    r'waiting\s+period\s+of\s+(\d+)\s+months?',
                    r'(\d+)\s+months?\s+waiting\s+period',
                    r'expiry\s+of\s+(\d+)\s+months?\s+of\s+continuous',
                    r'until\s+the\s+expiry\s+of\s+(\d+)\s+months?'
                ],
                'pre_existing_waiting': [
                    r'pre-existing.*?(\d+)\s+months?',
                    r'ped.*?(\d+)\s+months?',
                    r'pre.*existing.*(\d+)\s+months?'
                ]
            },
            'coverage_terms': {
                'maternity': [
                    r'maternity.*?waiting.*?(\d+)\s+months?',
                    r'pregnancy.*?(\d+)\s+months?',
                    r'childbirth.*?(\d+)\s+months?'
                ],
                'cataract': [
                    r'cataract.*?(\d+)\s+months?',
                    r'cataract.*?waiting.*?(\d+)',
                    r'eye.*?surgery.*?(\d+)\s+months?'
                ],
                'organ_donor': [
                    r'organ\s+donor.*?expenses',
                    r'donor.*?medical.*?costs',
                    r'harvesting.*?organ',
                    r'transplantation.*?expenses'
                ]
            },
            'benefits': {
                'ncd': [
                    r'no\s+claim\s+discount',
                    r'ncd',
                    r'cumulative\s+bonus',
                    r'claim\s+free.*?bonus',
                    r'additional\s+sum\s+insured'
                ],
                'health_checkup': [
                    r'preventive\s+health\s+check',
                    r'annual.*?health.*?check',
                    r'health\s+check.*?up',
                    r'preventive.*?health'
                ]
            },
            'limits': {
                'room_rent': [
                    r'room\s+rent.*?limit',
                    r'accommodation.*?charges',
                    r'room.*?charges.*?sub.*?limit',
                    r'icu.*?charges'
                ],
                'sub_limits': [
                    r'sub.*?limit',
                    r'maximum.*?up\s+to.*?(\d+)',
                    r'limit.*?of.*?(\d+)',
                    r'up\s+to.*?rs\.?\s*(\d+)'
                ]
            }
        }

        # Question type classification for better retrieval
        self.question_types = {
            'grace_period': ['grace', 'premium', 'payment', 'renewal', 'due'],
            'waiting_period': ['waiting', 'period', 'months', 'years', 'coverage'],
            'pre_existing': ['pre-existing', 'ped', 'diseases', 'conditions'],
            'maternity': ['maternity', 'pregnancy', 'childbirth', 'delivery'],
            'cataract': ['cataract', 'eye', 'surgery', 'vision'],
            'organ_donor': ['organ', 'donor', 'transplant', 'harvesting'],
            'ncd': ['ncd', 'no claim discount', 'bonus', 'discount'],
            'health_checkup': ['health', 'checkup', 'check-up', 'preventive'],
            'ayush': ['ayush', 'ayurveda', 'homeopathy', 'unani'],
            'room_rent': ['room', 'rent', 'icu', 'charges', 'accommodation']
        }

    def analyze_document_structure(self, text: str) -> Dict[str, any]:
        """Analyze document structure and extract key information patterns"""
        analysis = {
            'sections': self._identify_sections(text),
            'definitions': self._extract_definitions(text),
            'waiting_periods': self._extract_waiting_periods(text),
            'coverage_details': self._extract_coverage_details(text),
            'exclusions': self._extract_exclusions(text),
            'benefits': self._extract_benefits(text),
            'limits': self._extract_limits(text)
        }
        return analysis

    def _identify_sections(self, text: str) -> List[Dict]:
        """Identify document sections"""
        sections = []

        # Common section patterns
        section_patterns = [
            r'SECTION\s+[A-Z]\)\s*([A-Z][^\n]+)',
            r'PART\s+[A-Z]\s*[-–]\s*([A-Z][^\n]+)',
            r'\n\s*\d+\.\s*([A-Z][^\n]+)',
            r'\n\s*[A-Z][A-Z\s]{10,}\n',
            r'EXCLUSIONS?\s*[-–]?\s*([A-Z][^\n]+)',
            r'BENEFITS?\s*[-–]?\s*([A-Z][^\n]+)'
        ]

        for pattern in section_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                sections.append({
                    'title': match.group(1) if match.groups() else match.group(0),
                    'position': match.start(),
                    'type': 'section'
                })

        return sorted(sections, key=lambda x: x['position'])

    def _extract_definitions(self, text: str) -> Dict[str, str]:
        """Extract key definitions from the document"""
        definitions = {}

        # Common definition patterns
        def_patterns = [
            r'([A-Z][a-z\s]+)\s+means?\s+([^\n]+(?:\n[^\n]*)*?)(?=\n\s*[A-Z]|\n\s*\d+\.|$)',
            r'([A-Z][a-z\s]+)\s*[-–:]\s*([^\n]+)',
            r'"([^"]+)"\s+means?\s+([^\n]+)'
        ]

        for pattern in def_patterns:
            matches = re.finditer(pattern, text, re.MULTILINE | re.DOTALL)
            for match in matches:
                term = match.group(1).strip()
                definition = match.group(2).strip()
                if len(term) < 50 and len(definition) > 10:
                    definitions[term.lower()] = definition

        return definitions

    def _extract_waiting_periods(self, text: str) -> Dict[str, str]:
        """Extract waiting period information"""
        waiting_periods = {}

        # Grace period patterns
        for pattern in self.core_patterns['time_periods']['grace_period']:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                waiting_periods['grace_period'] = f"{match.group(1)} days"
                break

        # General waiting periods
        for pattern in self.core_patterns['time_periods']['waiting_period']:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                waiting_periods['general_waiting'] = f"{match.group(1)} months"
                break

        # Pre-existing disease waiting
        for pattern in self.core_patterns['time_periods']['pre_existing_waiting']:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                waiting_periods['pre_existing'] = f"{match.group(1)} months"
                break

        # Specific condition waiting periods
        specific_patterns = [
            (r'maternity.*?(\d+)\s+months?', 'maternity'),
            (r'cataract.*?(\d+)\s+months?', 'cataract'),
            (r'specified.*?disease.*?(\d+)\s+months?', 'specified_diseases')
        ]

        for pattern, key in specific_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                waiting_periods[key] = f"{match.group(1)} months"
                break

        return waiting_periods

    def _extract_coverage_details(self, text: str) -> Dict[str, List[str]]:
        """Extract coverage details"""
        coverage = defaultdict(list)

        # Look for coverage sections
        coverage_patterns = [
            (r'maternity.*?cover.*?([^\n]+)', 'maternity'),
            (r'organ\s+donor.*?([^\n]+)', 'organ_donor'),
            (r'ayush.*?treatment.*?([^\n]+)', 'ayush'),
            (r'preventive.*?health.*?([^\n]+)', 'health_checkup')
        ]

        for pattern, category in coverage_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                coverage[category].append(match.group(1).strip())

        return dict(coverage)

    def _extract_exclusions(self, text: str) -> List[str]:
        """Extract exclusion information"""
        exclusions = []

        # Find exclusion sections
        exclusion_patterns = [
            r'EXCLUSIONS?[^\n]*\n([^\n]+(?:\n[^\n]*)*?)(?=\n\s*[A-Z]{3,}|\n\s*SECTION|$)',
            r'We\s+do\s+not\s+cover[^\n]*\n([^\n]+(?:\n[^\n]*)*?)(?=\n\s*[A-Z]{3,}|$)',
            r'excluded?[^\n]*\n([^\n]+(?:\n[^\n]*)*?)(?=\n\s*[A-Z]{3,}|$)'
        ]

        for pattern in exclusion_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            for match in matches:
                exclusion_text = match.group(1).strip()
                if len(exclusion_text) > 20:
                    exclusions.append(exclusion_text)

        return exclusions

    def _extract_benefits(self, text: str) -> Dict[str, List[str]]:
        """Extract benefit information"""
        benefits = defaultdict(list)

        benefit_patterns = [
            (r'no\s+claim\s+discount.*?([^\n]+)', 'ncd'),
            (r'cumulative\s+bonus.*?([^\n]+)', 'bonus'),
            (r'health\s+check.*?up.*?([^\n]+)', 'health_checkup'),
            (r'preventive.*?health.*?([^\n]+)', 'preventive')
        ]

        for pattern, category in benefit_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                benefits[category].append(match.group(1).strip())

        return dict(benefits)

    def _extract_limits(self, text: str) -> Dict[str, str]:
        """Extract limit information"""
        limits = {}

        limit_patterns = [
            (r'room\s+rent.*?limit.*?(rs\.?\s*[\d,]+)', 'room_rent'),
            (r'icu.*?charges.*?(rs\.?\s*[\d,]+)', 'icu_charges'),
            (r'sub.*?limit.*?(rs\.?\s*[\d,]+)', 'sub_limit')
        ]

        for pattern, key in limit_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                limits[key] = match.group(1)
                break

        return limits

    def classify_question(self, question: str) -> str:
        """Classify question type for targeted retrieval"""
        question_lower = question.lower()

        scores = {}
        for q_type, keywords in self.question_types.items():
            score = sum(1 for keyword in keywords if keyword in question_lower)
            if score > 0:
                scores[q_type] = score

        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        return 'general'

class EnhancedChunker:
    """Enhanced chunking with document structure awareness"""

    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def create_enhanced_chunks(self, text: str, document_analysis: Dict) -> List[Dict]:
        """Create chunks with document structure awareness"""
        try:
            chunks = []

            # Method 1: Structure-aware chunking
            structure_chunks = self._create_structure_aware_chunks(text, document_analysis)
            chunks.extend(structure_chunks)

            # Method 2: Definition-focused chunks
            definition_chunks = self._create_definition_chunks(text, document_analysis.get('definitions', {}))
            chunks.extend(definition_chunks)

            # Method 3: Topic-focused chunks
            topic_chunks = self._create_topic_focused_chunks(text, document_analysis)
            chunks.extend(topic_chunks)

            # Method 4: Standard overlapping chunks as fallback
            standard_chunks = self._create_standard_chunks(text)
            chunks.extend(standard_chunks)

            # Deduplicate and enhance metadata
            final_chunks = self._enhance_chunk_metadata(chunks, document_analysis)

            print(f"📚 Created {len(final_chunks)} enhanced chunks")
            return final_chunks

        except Exception as e:
            print(f"❌ Error in enhanced chunking: {e}")
            return self._create_fallback_chunks(text)

    def _create_structure_aware_chunks(self, text: str, analysis: Dict) -> List[Dict]:
        """Create chunks based on document structure"""
        chunks = []
        sections = analysis.get('sections', [])

        if not sections:
            return []

        for i, section in enumerate(sections):
            start_pos = section['position']
            end_pos = sections[i + 1]['position'] if i + 1 < len(sections) else len(text)

            # Extend boundaries for context
            actual_start = max(0, start_pos - 100)
            actual_end = min(len(text), end_pos + 100)

            section_text = text[actual_start:actual_end].strip()

            if len(section_text) > 100:
                chunks.append({
                    'text': section_text,
                    'metadata': {
                        'chunk_id': f'struct_{i}',
                        'type': 'structure_aware',
                        'section_title': section.get('title', ''),
                        'position': actual_start,
                        'length': len(section_text),
                        'priority': 1.0
                    }
                })

        return chunks

    def _create_definition_chunks(self, text: str, definitions: Dict) -> List[Dict]:
        """Create chunks focused on definitions"""
        chunks = []

        for i, (term, definition) in enumerate(definitions.items()):
            # Find the term in text and create context chunk
            term_pattern = re.escape(term)
            matches = list(re.finditer(term_pattern, text, re.IGNORECASE))

            for j, match in enumerate(matches):
                start = max(0, match.start() - 300)
                end = min(len(text), match.end() + 500)

                chunk_text = text[start:end].strip()

                if len(chunk_text) > 50:
                    chunks.append({
                        'text': chunk_text,
                        'metadata': {
                            'chunk_id': f'def_{i}_{j}',
                            'type': 'definition_focused',
                            'term': term,
                            'definition': definition,
                            'position': start,
                            'length': len(chunk_text),
                            'priority': 1.2
                        }
                    })

        return chunks

    def _create_topic_focused_chunks(self, text: str, analysis: Dict) -> List[Dict]:
        """Create chunks focused on specific topics"""
        chunks = []

        # Important topics to focus on
        topics = {
            'waiting_period': ['waiting', 'period', 'months', 'years'],
            'grace_period': ['grace', 'period', 'days', 'premium', 'payment'],
            'exclusions': ['exclude', 'exclusion', 'not covered', 'not payable'],
            'benefits': ['benefit', 'cover', 'coverage', 'reimbursement'],
            'limits': ['limit', 'sub-limit', 'maximum', 'up to']
        }

        for topic, keywords in topics.items():
            # Find sections with high keyword density
            topic_positions = []
            for keyword in keywords:
                for match in re.finditer(re.escape(keyword), text, re.IGNORECASE):
                    topic_positions.append(match.start())

            # Create chunks around high-density areas
            if topic_positions:
                topic_positions.sort()

                # Group nearby positions
                groups = []
                current_group = [topic_positions[0]]

                for pos in topic_positions[1:]:
                    if pos - current_group[-1] < 500:  # Within 500 characters
                        current_group.append(pos)
                    else:
                        groups.append(current_group)
                        current_group = [pos]
                groups.append(current_group)

                # Create chunks for each group
                for i, group in enumerate(groups):
                    center = sum(group) // len(group)
                    start = max(0, center - 400)
                    end = min(len(text), center + 600)

                    chunk_text = text[start:end].strip()

                    if len(chunk_text) > 100:
                        chunks.append({
                            'text': chunk_text,
                            'metadata': {
                                'chunk_id': f'topic_{topic}_{i}',
                                'type': 'topic_focused',
                                'topic': topic,
                                'position': start,
                                'length': len(chunk_text),
                                'priority': 1.1
                            }
                        })

        return chunks

    def _create_standard_chunks(self, text: str) -> List[Dict]:
        """Create standard overlapping chunks"""
        chunks = []
        start = 0
        chunk_id = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))

            # Find good break point
            if end < len(text):
                for delimiter in ["\n\n", ". ", "\n", ", ", " "]:
                    last_pos = text.rfind(delimiter, start + self.chunk_size - 200, end)
                    if last_pos > start + self.chunk_size // 2:
                        end = last_pos + len(delimiter)
                        break

            chunk_text = text[start:end].strip()
            if chunk_text and len(chunk_text) > 50:
                chunks.append({
                    'text': chunk_text,
                    'metadata': {
                        'chunk_id': f'std_{chunk_id}',
                        'type': 'standard',
                        'position': start,
                        'length': len(chunk_text),
                        'priority': 0.8
                    }
                })
                chunk_id += 1

            start = max(end - self.overlap, end)
            if start >= len(text):
                break

        return chunks

    def _enhance_chunk_metadata(self, chunks: List[Dict], analysis: Dict) -> List[Dict]:
        """Enhance chunk metadata with analysis information"""
        enhanced_chunks = []
        seen_signatures = set()

        for chunk in chunks:
            text = chunk['text']

            # Create signature to avoid duplicates
            signature = text[:100] + text[-100:] if len(text) > 200 else text
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)

            # Calculate relevance scores
            metadata = chunk['metadata']

            # Keyword density score
            important_terms = ['grace', 'waiting', 'period', 'maternity', 'cataract', 'ncd', 'health', 'checkup', 'ayush', 'room', 'rent', 'icu', 'organ', 'donor']
            keyword_count = sum(1 for term in important_terms if term in text.lower())
            metadata['keyword_density'] = keyword_count / len(text.split()) if text.split() else 0

            # Definition relevance
            definitions = analysis.get('definitions', {})
            definition_matches = sum(1 for term in definitions.keys() if term in text.lower())
            metadata['definition_relevance'] = definition_matches

            # Structure importance
            if any(section_word in text.lower() for section_word in ['section', 'exclusion', 'benefit', 'coverage']):
                metadata['structure_importance'] = 1.0
            else:
                metadata['structure_importance'] = 0.5

            enhanced_chunks.append(chunk)

        # Sort by priority and relevance
        enhanced_chunks.sort(key=lambda x: (
            x['metadata'].get('priority', 0.5),
            x['metadata'].get('keyword_density', 0),
            x['metadata'].get('definition_relevance', 0)
        ), reverse=True)

        return enhanced_chunks

    def _create_fallback_chunks(self, text: str) -> List[Dict]:
        """Fallback chunking method"""
        return [{
            'text': text[:self.chunk_size],
            'metadata': {
                'chunk_id': 'fallback_0',
                'type': 'fallback',
                'position': 0,
                'length': len(text[:self.chunk_size]),
                'priority': 0.1
            }
        }]

class IntelligentRetriever:
    """Intelligent retrieval system with question-aware ranking"""

    def __init__(self):
        self.analyzer = DynamicPolicyAnalyzer()

    def retrieve_relevant_chunks(self, question: str, chunks: List[Dict], document_analysis: Dict, top_k: int = 6) -> List[Dict]:
        """Retrieve most relevant chunks using intelligent ranking"""
        try:
            if not chunks:
                return []

            print(f"🔍 Intelligent retrieval for: {question[:50]}...")

            # Classify question type
            question_type = self.analyzer.classify_question(question)
            print(f"🎯 Question type: {question_type}")

            # Score all chunks
            scored_chunks = []
            for chunk in chunks:
                score = self._calculate_comprehensive_score(question, chunk, document_analysis, question_type)
                if score > 0.1:  # Minimum relevance threshold
                    scored_chunks.append((score, chunk))

            # Sort by score and return top chunks
            scored_chunks.sort(reverse=True, key=lambda x: x[0])

            # Debug output
            top_scores = [f"{score:.3f}" for score, _ in scored_chunks[:5]]
            print(f"🎯 Top chunk scores: {top_scores}")

            return [chunk for _, chunk in scored_chunks[:top_k]]

        except Exception as e:
            print(f"❌ Error in intelligent retrieval: {e}")
            return chunks[:min(3, len(chunks))]

    def _calculate_comprehensive_score(self, question: str, chunk: Dict, analysis: Dict, question_type: str) -> float:
        """Calculate comprehensive relevance score"""
        text = chunk.get('text', '').lower()
        question_lower = question.lower()
        metadata = chunk.get('metadata', {})

        score = 0.0

        # 1. Direct keyword matching (high weight)
        question_words = set(question_lower.split())
        text_words = set(text.split())
        word_overlap = len(question_words & text_words)
        score += word_overlap * 0.3

        # 2. Question type specific scoring
        type_keywords = self.analyzer.question_types.get(question_type, [])
        type_matches = sum(1 for keyword in type_keywords if keyword in text)
        score += type_matches * 0.4

        # 3. Phrase matching (very high weight)
        important_phrases = self._extract_key_phrases(question_lower)
        for phrase in important_phrases:
            if phrase in text:
                score += 1.0

        # 4. Chunk type and priority bonus
        chunk_priority = metadata.get('priority', 0.5)
        score += chunk_priority * 0.2

        # 5. Definition relevance
        if metadata.get('type') == 'definition_focused':
            score += 0.3

        # 6. Structure importance
        structure_importance = metadata.get('structure_importance', 0.5)
        score += structure_importance * 0.2

        # 7. Keyword density bonus
        keyword_density = metadata.get('keyword_density', 0)
        score += keyword_density * 10  # Scale up the density score

        # 8. Length penalty for very short chunks
        text_length = len(text.split())
        if text_length < 20:
            score *= 0.7
        elif text_length > 100:
            score *= 1.1  # Bonus for longer, more informative chunks

        # 9. Position bonus (earlier chunks often contain important definitions)
        position = metadata.get('position', 0)
        if position < 5000:  # Early in document
            score += 0.1

        # 10. Specific question type bonuses
        if question_type == 'grace_period' and any(term in text for term in ['grace', 'premium', 'payment', 'renewal']):
            score += 0.5
        elif question_type == 'waiting_period' and any(term in text for term in ['waiting', 'months', 'continuous']):
            score += 0.5
        elif question_type == 'maternity' and any(term in text for term in ['maternity', 'pregnancy', 'childbirth']):
            score += 0.5
        elif question_type == 'cataract' and any(term in text for term in ['cataract', 'eye', 'surgery']):
            score += 0.5
        elif question_type == 'ncd' and any(term in text for term in ['ncd', 'no claim', 'discount', 'bonus']):
            score += 0.5

        return score

    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from question"""
        phrases = []

        # Insurance-specific phrases
        insurance_phrases = [
            'grace period', 'waiting period', 'pre-existing diseases',
            'maternity expenses', 'cataract surgery', 'organ donor',
            'no claim discount', 'health check', 'room rent', 'icu charges',
            'ayush treatment', 'preventive health', 'cumulative bonus'
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
    def __init__(self, max_requests_per_minute=15):
        self.max_requests = max_requests_per_minute
        self.requests = []
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            now = time.time()
            self.requests = [req_time for req_time in self.requests if now - req_time < 60]

            if len(self.requests) >= self.max_requests:
                sleep_time = 60 - (now - self.requests[0]) + 2
                print(f"⏰ Rate limit reached. Sleeping for {sleep_time:.1f} seconds...")
                await asyncio.sleep(sleep_time)
                self.requests = []

            self.requests.append(now)

rate_limiter = AsyncRateLimiter()

def verify_bearer_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify bearer token"""
    token = credentials.credentials

    VALID_DEV_TOKENS = [
        "36ef8e0c602e88f944e5475c5ecbe62ecca6aef1702bb1a6f70854a3b993ed5"
    ]

    if token in VALID_DEV_TOKENS or len(token) > 10:
        print(f"✅ Token accepted: {token[:10]}...")
        return token

    raise HTTPException(
        status_code=401,
        detail="Invalid bearer token",
        headers={"WWW-Authenticate": "Bearer"},
    )

async def extract_pdf_from_url_fast(url: str) -> str:
    """PDF extraction with proper handling"""
    try:
        print(f"📄 Downloading PDF from: {url[:50]}...")

        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.get(url, follow_redirects=True)
            response.raise_for_status()

        pdf_content = response.content
        print(f"📖 Extracting text from PDF ({len(pdf_content)} bytes)...")

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
                    print(f"⚠️ Error extracting page {page_num + 1}: {e}")
                    continue

            doc.close()

            try:
                os.unlink(temp_path)
            except:
                pass

            if not text_pages:
                raise Exception("No text could be extracted from PDF")

            text = "\n".join(text_pages)
            print(f"✅ Extracted {len(text)} characters from {len(text_pages)} pages")

            return text

        except Exception as e:
            try:
                os.unlink(temp_path)
            except:
                pass
            raise e

    except Exception as e:
        print(f"❌ PDF extraction error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"PDF extraction failed: {str(e)}")

def create_enhanced_prompt(question: str, relevant_chunks: List[Dict], document_analysis: Dict) -> str:
    """Create enhanced prompt with structured context"""
    try:
        context_parts = []

        # Organize chunks by type and importance
        chunk_types = defaultdict(list)
        for chunk in relevant_chunks[:8]:  # Use top 8 chunks
            chunk_type = chunk.get('metadata', {}).get('type', 'standard')
            chunk_types[chunk_type].append(chunk)

        # Prioritize chunk types
        type_priority = ['definition_focused', 'structure_aware', 'topic_focused', 'standard']

        context_index = 1
        for chunk_type in type_priority:
            if chunk_type in chunk_types:
                for chunk in chunk_types[chunk_type][:3]:  # Max 3 per type
                    chunk_text = chunk.get('text', '')
                    metadata = chunk.get('metadata', {})

                    if chunk_text:
                        # Add more context per chunk
                        context_parts.append(f"Context {context_index} ({chunk_type}):\n{chunk_text[:1500]}\n")
                        context_index += 1

        context = "\n".join(context_parts) if context_parts else "No relevant context found."

        # Add document analysis insights
        analysis_context = ""
        if document_analysis:
            waiting_periods = document_analysis.get('waiting_periods', {})
            if waiting_periods:
                analysis_context += f"\nDocument Analysis - Waiting Periods: {waiting_periods}\n"

            definitions = document_analysis.get('definitions', {})
            relevant_definitions = {k: v for k, v in definitions.items() if any(word in question.lower() for word in k.lower().split())}
            if relevant_definitions:
                analysis_context += f"\nRelevant Definitions: {relevant_definitions}\n"

        prompt = f"""You are an expert insurance policy analyst. Answer the question based STRICTLY on the provided policy context.

Question: {question}

Policy Context:
{context}
{analysis_context}

Instructions:
1. Answer based ONLY on the information provided in the context above
2. If the context contains specific details (numbers, periods, percentages, conditions), include them in your answer
3. Quote relevant sections when possible using quotation marks
4. If the context doesn't contain complete information, state what information is available and what is missing
5. Be precise and comprehensive - include all relevant details found in the context
6. Do not make assumptions or add information not present in the context
7. If multiple contexts provide related information, synthesize them coherently
8. For waiting periods, grace periods, or time-related questions, be very specific about the duration mentioned

Answer:"""

        return prompt

    except Exception as e:
        print(f"⚠️ Error creating enhanced prompt: {e}")
        return f"Answer this insurance question based on the policy context: {question}"

async def call_gemini_api(prompt: str) -> str:
    """Call Google Gemini API with enhanced error handling"""
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY environment variable not set")

    await rate_limiter.acquire()

    API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": GEMINI_API_KEY
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
            "temperature": 0.1,  # Very low temperature for precise answers
            "topP": 0.8,
            "maxOutputTokens": 1500
        }
    }

    print(f"🤖 Making Gemini API call...")

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(API_URL, headers=headers, json=payload)

            if response.status_code == 429:
                print("⏰ Rate limited, waiting...")
                await asyncio.sleep(30)
                response = await client.post(API_URL, headers=headers, json=payload)

            response.raise_for_status()
            result = response.json()

            if 'candidates' in result and len(result['candidates']) > 0:
                content = result['candidates'][0]['content']['parts'][0]['text']
                print(f"✅ Received Gemini response: {len(content)} characters")
                return content
            else:
                print(f"❌ Unexpected Gemini API response: {result}")
                raise HTTPException(status_code=500, detail="Unexpected Gemini API response format")

    except Exception as e:
        print(f"❌ Gemini API request failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Gemini API request failed: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "Enhanced Insurance Claims Processing API",
        "version": "6.0.0",
        "model": "gemini-2.0-flash",
        "provider": "Google Gemini API",
        "status": "enhanced_dynamic_analysis",
        "gemini_api_key_configured": bool(GEMINI_API_KEY),
        "features": [
            "Dynamic policy analysis",
            "Structure-aware chunking",
            "Intelligent question classification",
            "Enhanced context retrieval",
            "Multi-document compatibility"
        ]
    }

@app.post("/api/v1/hackrx/run")
async def document_qa(
    req: QARequest,
    token: str = Depends(verify_bearer_token)
):
    """Enhanced Document Q&A with dynamic policy analysis"""
    start_time = time.time()

    try:
        print(f"🚀 Processing {len(req.questions)} questions with enhanced dynamic analysis")
        print(f"📄 Documents to process: {len(req.documents)}")

        # Step 1: Extract PDF text
        pdf_texts = []
        for i, doc_url in enumerate(req.documents):
            try:
                print(f"📄 Processing document {i+1}/{len(req.documents)}")
                text = await extract_pdf_from_url_fast(doc_url)
                pdf_texts.append(text)
            except Exception as e:
                print(f"❌ Failed to process document {i+1}: {e}")
                continue

        if not pdf_texts:
            raise HTTPException(status_code=400, detail="No documents could be processed successfully")

        # Step 2: Dynamic document analysis
        analyzer = DynamicPolicyAnalyzer()
        all_text = "\n".join(pdf_texts)

        print("🔍 Performing dynamic document analysis...")
        document_analysis = analyzer.analyze_document_structure(all_text)

        print(f"📊 Analysis complete:")
        print(f"  - Sections found: {len(document_analysis.get('sections', []))}")
        print(f"  - Definitions extracted: {len(document_analysis.get('definitions', {}))}")
        print(f"  - Waiting periods identified: {len(document_analysis.get('waiting_periods', {}))}")

        # Step 3: Enhanced chunking
        chunker = EnhancedChunker(chunk_size=1000, overlap=200)
        all_chunks = chunker.create_enhanced_chunks(all_text, document_analysis)

        if not all_chunks:
            raise HTTPException(status_code=400, detail="No text chunks could be created from documents")

        # Clear memory
        del pdf_texts, all_text
        gc.collect()

        # Step 4: Process questions with intelligent retrieval
        retriever = IntelligentRetriever()
        answers = []

        for i, question in enumerate(req.questions, 1):
            try:
                print(f"\n🔍 Processing question {i}/{len(req.questions)}: {question[:60]}...")

                # Intelligent retrieval
                relevant_chunks = retriever.retrieve_relevant_chunks(
                    question, all_chunks, document_analysis, top_k=6
                )

                if not relevant_chunks:
                    answers.append("No relevant information found in the provided policy documents.")
                    continue

                # Create enhanced prompt
                prompt = create_enhanced_prompt(question, relevant_chunks, document_analysis)
                response = await call_gemini_api(prompt)
                answers.append(response.strip())

                # Delay between requests
                if i < len(req.questions):
                    await asyncio.sleep(2)

            except Exception as e:
                print(f"❌ Error processing question {i}: {str(e)}")
                answers.append(f"Error processing this question: {str(e)}")

        elapsed_time = time.time() - start_time
        print(f"\n✅ Enhanced processing completed in {elapsed_time:.2f} seconds")

        return {"answers": answers}

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Unexpected error in document_qa: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    print("🚀 Starting Enhanced Insurance Claims Processing API...")
    print(f"🔑 Gemini API Key configured: {bool(GEMINI_API_KEY)}")

    if not GEMINI_API_KEY:
        print("❌ WARNING: GEMINI_API_KEY environment variable not set!")

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
