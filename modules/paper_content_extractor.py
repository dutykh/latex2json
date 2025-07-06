# -*- coding: utf-8 -*-
"""
Paper content extractor using LLM with fallback support.

This module extracts abstracts, keywords, and metadata from publisher URLs
using LLM with automatic fallback between providers.

Author: Enhanced by Claude
Date: 2025-01-05
"""

import json
import re
import time
import requests
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from bs4 import BeautifulSoup
import hashlib
import xml.etree.ElementTree as ET

from .config_manager import ConfigManager
from .llm_provider import LLMProviderManager
from .publisher_identifier import PublisherIdentifier
from .display_utils import TerminalDisplay, Icons


class PaperContentExtractor:
    """Extract paper metadata from publisher URLs using LLM."""
    
    # Publisher-specific selectors for common elements
    PUBLISHER_SELECTORS = {
        'Elsevier': {
            'abstract': ['div.abstract', 'section#ab0010', 'div#abstracts'],
            'keywords': ['div.keywords', 'div.keyword', 'section#keywords'],
            'doi': ['a.doi', 'span.doi', 'meta[name="citation_doi"]'],
            'authors': ['div.author-group', 'div.authors', 'meta[name="citation_author"]']
        },
        'Springer': {
            'abstract': ['section.Abstract', 'div#Abs1', 'p.Para'],
            'keywords': ['ul.c-article-subject-list', 'div.KeywordGroup', 'span.Keyword'],
            'doi': ['span.c-bibliographic-information__value', 'meta[name="citation_doi"]'],
            'authors': ['ul.c-article-author-list', 'meta[name="citation_author"]']
        },
        'Wiley': {
            'abstract': ['section.article-section__abstract', 'div.abstract-group'],
            'keywords': ['div.article-section__keywords', 'ul.keywords-list'],
            'doi': ['span.epub-doi', 'meta[name="citation_doi"]'],
            'authors': ['div.loa-authors', 'meta[name="citation_author"]']
        },
        'Nature': {
            'abstract': ['div#Abs1', 'section[data-title="Abstract"]', 'p.Para'],
            'keywords': ['div.c-article-subject-list', 'span.c-article-subject-list__subject'],
            'doi': ['span.c-bibliographic-information__value', 'meta[name="citation_doi"]'],
            'authors': ['ul.c-article-author-list', 'meta[name="citation_author"]']
        },
        'IEEE': {
            'abstract': ['div.abstract-text', 'div.u-mb-1'],
            'keywords': ['ul.doc-keywords-list', 'span.stats-keywords-list-item'],
            'doi': ['div.stats-document-abstract-doi', 'meta[name="citation_doi"]'],
            'authors': ['span.authors-info', 'meta[name="citation_author"]']
        }
    }
    
    def __init__(self, config: ConfigManager, publisher_identifier: PublisherIdentifier,
                 cache_file: Optional[Path] = None, verbose: int = 2):
        """Initialize the content extractor.
        
        Args:
            config: Configuration manager
            publisher_identifier: Publisher identification module
            cache_file: Path to cache file
            verbose: Verbosity level (0-3)
        """
        self.config = config
        self.publisher_id = publisher_identifier
        self.verbose = verbose
        self.display = TerminalDisplay(verbose=verbose)
        
        # Initialize LLM provider manager
        self.llm_manager = LLMProviderManager(config)
        
        # Set up cache
        if cache_file is None:
            cache_dir = Path(__file__).parent.parent / "cache"
            cache_dir.mkdir(exist_ok=True)
            cache_file = cache_dir / ".content_extraction_cache.json"
        self.cache_file = cache_file
        self.cache = self._load_cache()
        
        # Configuration
        self.use_structured_extraction = config.get('extraction.use_structured', True)
        self.max_retries = config.get('extraction.max_retries', 2)
        
        # Statistics
        self.stats = {
            'urls_processed': 0,
            'extractions_successful': 0,
            'cache_hits': 0,
            'llm_extractions': 0,
            'structured_extractions': 0,
            'validation_failures': 0
        }
    
    def _load_cache(self) -> Dict[str, Any]:
        """Load cache from file."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                if self.verbose >= 2:
                    self.display.warning(f"Failed to load extraction cache: {e}")
                return {}
        return {}
    
    def _save_cache(self):
        """Save cache to file."""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            if self.verbose >= 2:
                self.display.warning(f"Failed to save extraction cache: {e}")
    
    async def extract_from_url(self, url: str, paper_data: Dict[str, Any],
                             publisher: Optional[str] = None) -> Dict[str, Any]:
        """Extract metadata from a paper URL.
        
        Args:
            url: URL to extract from
            paper_data: Known paper data for validation
            publisher: Publisher name (if known)
            
        Returns:
            Dictionary with extracted metadata including:
            - abstract: Full abstract text
            - keywords: List of keywords
            - doi: DOI if found
            - additional_metadata: Other extracted data
            - extraction_confidence: Confidence score
        """
        if not url:
            return self._empty_result()
        
        self.stats['urls_processed'] += 1
        
        # Check cache
        cache_key = self._make_cache_key(url)
        if cache_key in self.cache:
            self.stats['cache_hits'] += 1
            cached_result = self.cache[cache_key]
            
            # Check if this is a failed extraction (blocked/403)
            if cached_result.get('extraction_failed') and cached_result.get('failure_reason') == 'blocked':
                if self.verbose >= 3:
                    self.display.info("Cache hit", f"Known blocked URL: {cached_result.get('publisher', 'Unknown')}", Icons.BLOCKED)
                return cached_result
            elif self.verbose >= 3:
                self.display.info("Cache hit", f"Found cached extraction for URL", Icons.CHECK)
            return cached_result
        
        # Check if this is a preprint URL
        if 'arxiv.org' in url:
            result = await self._extract_from_arxiv(url, paper_data)
            if result and result.get('abstract'):
                self.stats['extractions_successful'] += 1
                # Cache and return
                self.cache[cache_key] = result
                self._save_cache()
                return result
        elif 'hal.archives' in url or 'hal.science' in url:
            result = await self._extract_from_hal(url, paper_data)
            if result and result.get('abstract'):
                self.stats['extractions_successful'] += 1
                # Cache and return
                self.cache[cache_key] = result
                self._save_cache()
                return result
        
        # Check if publisher is in blocked list
        publisher_from_url = self._identify_publisher_from_url(url)
        blocked_publishers = self.config.get('paper_search_settings.blocked_publishers', [])
        
        if publisher_from_url in blocked_publishers:
            if self.verbose >= 2:
                self.display.warning(f"Skipping known blocked publisher: {publisher_from_url}")
            # Cache as blocked
            blocked_result = {
                'extraction_failed': True,
                'failure_reason': 'blocked',
                'publisher': publisher_from_url,
                'timestamp': time.time()
            }
            self.cache[cache_key] = blocked_result
            self._save_cache()
            return self._empty_result()
        
        # For non-preprint URLs, continue with existing logic
        # Fetch page content
        html_content = await self._fetch_url_content(url)
        if not html_content:
            return self._empty_result()
        
        # Try structured extraction first if publisher is known
        result = None
        if publisher and self.use_structured_extraction:
            result = self._structured_extraction(html_content, publisher)
            if result and result.get('abstract'):
                self.stats['structured_extractions'] += 1
        
        # Use LLM extraction if structured failed or not available
        if not result or not result.get('abstract'):
            result = await self._llm_extraction(html_content, paper_data, publisher)
            if result and result.get('abstract'):
                self.stats['llm_extractions'] += 1
        
        # Validate extracted content
        if result:
            result = self._validate_extraction(result, paper_data)
            if result.get('validation_passed'):
                self.stats['extractions_successful'] += 1
            else:
                self.stats['validation_failures'] += 1
        
        # Cache successful extractions
        if result and result.get('abstract'):
            self.cache[cache_key] = result
            self._save_cache()
        
        return result or self._empty_result()
    
    async def _fetch_url_content(self, url: str) -> Optional[str]:
        """Fetch HTML content from URL."""
        try:
            headers = {
                'User-Agent': self.config.get('scraping.user_agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
            }
            
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            return response.text
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                # Cache this as a blocked URL
                cache_key = self._make_cache_key(url)
                blocked_result = {
                    'extraction_failed': True,
                    'failure_reason': 'blocked',
                    'status_code': 403,
                    'publisher': self.publisher_id.get_publisher_from_url(url) if hasattr(self.publisher_id, 'get_publisher_from_url') else 'Unknown',
                    'timestamp': time.time()
                }
                self.cache[cache_key] = blocked_result
                self._save_cache()
                
                if self.verbose >= 2:
                    self.display.warning(f"URL blocked (403): {url}")
            else:
                if self.verbose >= 2:
                    self.display.warning(f"HTTP error {e.response.status_code}: {e}")
            return None
        except Exception as e:
            if self.verbose >= 2:
                self.display.warning(f"Failed to fetch URL: {e}")
            return None
    
    def _structured_extraction(self, html: str, publisher: str) -> Optional[Dict[str, Any]]:
        """Extract metadata using publisher-specific selectors."""
        if publisher not in self.PUBLISHER_SELECTORS:
            return None
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            selectors = self.PUBLISHER_SELECTORS[publisher]
            
            result = {
                'abstract': None,
                'keywords': [],
                'doi': None,
                'extraction_method': 'structured',
                'publisher': publisher
            }
            
            # Extract abstract
            for selector in selectors.get('abstract', []):
                if selector.startswith('meta'):
                    # Handle meta tags
                    tag = soup.find('meta', attrs={'name': selector.split('[name="')[1].rstrip('"]')})
                    if tag and tag.get('content'):
                        result['abstract'] = tag['content'].strip()
                        break
                else:
                    # Handle regular selectors
                    element = soup.select_one(selector)
                    if element:
                        result['abstract'] = element.get_text(strip=True)
                        break
            
            # Extract keywords
            for selector in selectors.get('keywords', []):
                elements = soup.select(selector)
                for elem in elements:
                    keyword = elem.get_text(strip=True)
                    if keyword and len(keyword) < 100:  # Sanity check
                        result['keywords'].append(keyword)
            
            # Extract DOI
            for selector in selectors.get('doi', []):
                if selector.startswith('meta'):
                    tag = soup.find('meta', attrs={'name': selector.split('[name="')[1].rstrip('"]')})
                    if tag and tag.get('content'):
                        result['doi'] = tag['content'].strip()
                        break
                else:
                    element = soup.select_one(selector)
                    if element:
                        doi_text = element.get_text(strip=True)
                        # Extract DOI pattern
                        doi_match = re.search(r'10\.\d{4,}/[-._;()/:\w]+', doi_text)
                        if doi_match:
                            result['doi'] = doi_match.group(0)
                            break
            
            # Clean up keywords
            result['keywords'] = list(set(result['keywords']))[:20]  # Limit to 20 unique keywords
            
            if self.verbose >= 3 and result['abstract']:
                self.display.info("Structured extraction", 
                                f"Found abstract ({len(result['abstract'])} chars), "
                                f"{len(result['keywords'])} keywords",
                                Icons.CHECK)
            
            return result if result['abstract'] else None
            
        except Exception as e:
            if self.verbose >= 2:
                self.display.warning(f"Structured extraction failed: {e}")
            return None
    
    async def _llm_extraction(self, html: str, paper_data: Dict[str, Any], 
                            publisher: Optional[str]) -> Optional[Dict[str, Any]]:
        """Extract metadata using LLM."""
        self.stats['llm_extractions'] += 1
        
        # Clean HTML for LLM (remove scripts, styles, etc.)
        clean_html = self._clean_html_for_llm(html)
        
        # Prepare paper info for validation
        paper_info = {
            'title': paper_data.get('title', ''),
            'authors': [self._format_author_name(a) for a in paper_data.get('authors', [])[:3]],
            'year': paper_data.get('year', ''),
            'journal': paper_data.get('journal', '')
        }
        
        prompt = f"""Extract metadata from this academic paper webpage.

EXPECTED PAPER:
{json.dumps(paper_info, indent=2)}

WEBPAGE CONTENT (CLEANED HTML):
{clean_html[:8000]}  # Limit to avoid token limits

Please extract:
1. Full abstract (the complete text, not a summary)
2. All keywords/tags (author keywords, subject classifications)
3. DOI if present
4. Any additional metadata (publication date, volume, issue, pages)

IMPORTANT:
- Extract the FULL abstract text, not a summary
- Verify this matches the expected paper (title, authors)
- If keywords are not explicitly listed, DO NOT generate them
- Return structured data only

Return ONLY a JSON object:
{{
    "matches_expected_paper": true/false,
    "confidence": 0.0-1.0,
    "abstract": "Full abstract text...",
    "keywords": ["keyword1", "keyword2", ...],
    "doi": "10.xxxx/yyyy",
    "additional_metadata": {{
        "publication_date": "2025-01-01",
        "volume": "42",
        "issue": "8",
        "pages": "085003",
        "other_fields": "..."
    }},
    "extraction_notes": "Any relevant notes"
}}"""

        try:
            response = self.llm_manager.generate(
                prompt,
                temperature=0.1,  # Low temperature for accurate extraction
                max_tokens=2000
            )
            
            if not response:
                return None
            
            # Parse JSON response
            json_text = response.strip()
            if '```json' in json_text:
                json_text = json_text.split('```json')[1].split('```')[0].strip()
            elif '```' in json_text:
                json_text = json_text.split('```')[1].split('```')[0].strip()
            
            result = json.loads(json_text)
            
            # Check if paper matches
            if not result.get('matches_expected_paper', False):
                if self.verbose >= 2:
                    self.display.warning("LLM extraction: Paper mismatch detected")
                return None
            
            # Format result
            extracted = {
                'abstract': result.get('abstract'),
                'keywords': result.get('keywords', []),
                'doi': result.get('doi'),
                'extraction_method': 'llm',
                'extraction_confidence': result.get('confidence', 0.7),
                'publisher': publisher,
                'additional_metadata': result.get('additional_metadata', {})
            }
            
            if self.verbose >= 3 and extracted['abstract']:
                self.display.info("LLM extraction", 
                                f"Extracted abstract ({len(extracted['abstract'])} chars), "
                                f"{len(extracted['keywords'])} keywords - "
                                f"confidence: {extracted['extraction_confidence']:.2f}",
                                Icons.SPARKLES)
            
            return extracted
            
        except Exception as e:
            if self.verbose >= 2:
                self.display.warning(f"LLM extraction failed: {e}")
            return None
    
    def _clean_html_for_llm(self, html: str) -> str:
        """Clean HTML for LLM processing."""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "noscript", "header", "footer", "nav"]):
            script.decompose()
        
        # Get text with some structure preserved
        text = soup.get_text(separator='\n', strip=True)
        
        # Clean up excessive whitespace
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        text = '\n'.join(lines)
        
        # Limit length and focus on relevant sections
        # Try to find abstract section
        abstract_start = -1
        for i, line in enumerate(lines):
            if any(word in line.lower() for word in ['abstract', 'summary']):
                abstract_start = i
                break
        
        if abstract_start >= 0:
            # Return content starting from abstract
            relevant_text = '\n'.join(lines[abstract_start:abstract_start+200])
        else:
            # Return first part of content
            relevant_text = '\n'.join(lines[:200])
        
        return relevant_text
    
    def _validate_extraction(self, result: Dict[str, Any], 
                           paper_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate extracted content against known paper data."""
        validation_passed = True
        validation_notes = []
        
        # Check abstract length
        abstract = result.get('abstract', '')
        if abstract:
            if len(abstract) < 100:
                validation_notes.append("Abstract seems too short")
                validation_passed = False
            elif len(abstract) > 5000:
                validation_notes.append("Abstract seems too long")
                # Truncate to reasonable length
                result['abstract'] = abstract[:5000] + "..."
        
        # Validate keywords
        keywords = result.get('keywords', [])
        if keywords:
            # Remove duplicates and clean
            cleaned_keywords = []
            seen = set()
            for kw in keywords:
                kw_clean = kw.strip().lower()
                if kw_clean and kw_clean not in seen and len(kw_clean) < 100:
                    cleaned_keywords.append(kw.strip())
                    seen.add(kw_clean)
            result['keywords'] = cleaned_keywords[:30]  # Limit to 30 keywords
        
        # Validate DOI format
        doi = result.get('doi', '')
        if doi:
            doi_match = re.match(r'^10\.\d{4,}/[-._;()/:\w]+$', doi)
            if not doi_match:
                validation_notes.append("Invalid DOI format")
                result['doi'] = None
        
        # Add validation metadata
        result['validation_passed'] = validation_passed
        result['validation_notes'] = validation_notes
        
        return result
    
    def _make_cache_key(self, url: str) -> str:
        """Create cache key from URL."""
        return hashlib.md5(url.encode('utf-8')).hexdigest()
    
    def _format_author_name(self, author: Any) -> str:
        """Format author name."""
        if isinstance(author, dict):
            return f"{author.get('firstName', '')} {author.get('lastName', '')}".strip()
        return str(author)
    
    def _identify_publisher_from_url(self, url: str) -> Optional[str]:
        """Identify publisher from URL patterns."""
        url_lower = url.lower()
        
        # Common publisher URL patterns
        publisher_patterns = {
            'sciencedirect.com': 'Elsevier',
            'elsevier.com': 'Elsevier',
            'springer.com': 'Springer',
            'springerlink.com': 'Springer',
            'nature.com': 'Nature Publishing Group',
            'wiley.com': 'Wiley',
            'onlinelibrary.wiley.com': 'Wiley',
            'ieee.org': 'IEEE',
            'ieeexplore.ieee.org': 'IEEE',
            'iopscience.iop.org': 'IOP Publishing',
            'aps.org': 'American Physical Society',
            'taylor': 'Taylor & Francis',
            'tandfonline.com': 'Taylor & Francis',
            'sagepub.com': 'SAGE Publications',
            'cambridge.org': 'Cambridge University Press',
            'oxford': 'Oxford University Press',
            'plos.org': 'PLOS',
            'frontiersin.org': 'Frontiers',
            'mdpi.com': 'MDPI',
            'bmj.com': 'BMJ'
        }
        
        for pattern, publisher in publisher_patterns.items():
            if pattern in url_lower:
                return publisher
        
        return None
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty extraction result."""
        return {
            'abstract': None,
            'keywords': [],
            'doi': None,
            'extraction_method': 'none',
            'extraction_confidence': 0.0,
            'publisher': None,
            'additional_metadata': {},
            'validation_passed': False,
            'validation_notes': ['No extraction performed'],
            'timestamp': time.time()
        }
    
    def get_statistics(self) -> Dict[str, int]:
        """Get extraction statistics."""
        return self.stats.copy()
    
    async def validate_and_enhance_keywords(self, keywords: List[str], 
                                          abstract: str,
                                          paper_title: str) -> List[str]:
        """Validate and potentially enhance keywords using LLM.
        
        Args:
            keywords: Existing keywords
            abstract: Paper abstract
            paper_title: Paper title
            
        Returns:
            Validated and enhanced keyword list
        """
        if not self.llm_manager.is_available() or not abstract:
            return keywords
        
        prompt = f"""Validate and enhance these keywords for an academic paper.

PAPER TITLE: {paper_title}

ABSTRACT: {abstract[:1000]}...

EXISTING KEYWORDS: {', '.join(keywords) if keywords else 'None found'}

Please:
1. Validate if existing keywords are appropriate
2. Remove any that don't fit the paper content
3. Add 3-5 important keywords if less than 5 exist
4. Ensure keywords are specific and relevant to the field

Return ONLY a JSON object:
{{
    "validated_keywords": ["keyword1", "keyword2", ...],
    "removed_keywords": ["keyword1", ...],
    "added_keywords": ["keyword1", ...],
    "total_keywords": 8
}}"""

        try:
            response = self.llm_manager.generate(
                prompt,
                temperature=0.2,
                max_tokens=500
            )
            
            if not response:
                return keywords
            
            # Parse response
            json_text = response.strip()
            if '```json' in json_text:
                json_text = json_text.split('```json')[1].split('```')[0].strip()
            elif '```' in json_text:
                json_text = json_text.split('```')[1].split('```')[0].strip()
            
            result = json.loads(json_text)
            validated = result.get('validated_keywords', keywords)
            
            if self.verbose >= 3:
                removed = result.get('removed_keywords', [])
                added = result.get('added_keywords', [])
                if removed or added:
                    self.display.info("Keyword validation",
                                    f"Removed: {len(removed)}, Added: {len(added)}",
                                    Icons.SPARKLES)
            
            return validated[:15]  # Limit to 15 keywords
            
        except Exception as e:
            if self.verbose >= 2:
                self.display.warning(f"Keyword validation failed: {e}")
            return keywords
    
    async def _extract_from_arxiv(self, url: str, paper_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract metadata from ArXiv using their API.
        
        Args:
            url: ArXiv URL (e.g., https://arxiv.org/abs/2401.12345)
            paper_data: Known paper data for validation
            
        Returns:
            Extracted metadata dictionary or None
        """
        try:
            # Extract ArXiv ID from URL
            arxiv_match = re.search(r'arxiv\.org/abs/(\d+\.\d+)', url)
            if not arxiv_match:
                if self.verbose >= 2:
                    self.display.warning(f"Could not extract ArXiv ID from URL: {url}")
                return None
            
            arxiv_id = arxiv_match.group(1)
            
            if self.verbose >= 3:
                self.display.info("ArXiv extraction", f"Fetching metadata for {arxiv_id}", Icons.GLOBE)
            
            # Use ArXiv API
            api_url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
            
            # Get timeout from config
            timeout = self.config.get('extraction.arxiv_api_timeout', 10)
            
            response = requests.get(api_url, timeout=timeout)
            response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(response.text)
            
            # Define namespaces
            ns = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }
            
            # Find the entry
            entry = root.find('atom:entry', ns)
            if entry is None:
                if self.verbose >= 2:
                    self.display.warning(f"No entry found in ArXiv response for {arxiv_id}")
                return None
            
            # Extract metadata
            result = {
                'abstract': None,
                'keywords': [],
                'doi': None,
                'extraction_method': 'arxiv_api',
                'extraction_confidence': 0.95,
                'publisher': 'ArXiv',
                'additional_metadata': {
                    'arxiv_id': arxiv_id,
                    'arxiv_url': url
                }
            }
            
            # Extract title
            title_elem = entry.find('atom:title', ns)
            if title_elem is not None:
                extracted_title = title_elem.text.strip().replace('\n', ' ')
                result['additional_metadata']['extracted_title'] = extracted_title
            
            # Extract abstract
            summary_elem = entry.find('atom:summary', ns)
            if summary_elem is not None:
                abstract_text = summary_elem.text.strip()
                # Clean up abstract (remove extra whitespace and newlines)
                abstract_text = ' '.join(abstract_text.split())
                result['abstract'] = abstract_text
                
                if self.verbose >= 3:
                    self.display.info("ArXiv abstract", f"Extracted {len(abstract_text)} characters", Icons.CHECK)
            
            # Extract authors
            authors = []
            for author_elem in entry.findall('atom:author', ns):
                name_elem = author_elem.find('atom:name', ns)
                if name_elem is not None:
                    authors.append(name_elem.text.strip())
            if authors:
                result['additional_metadata']['extracted_authors'] = authors
            
            # Extract categories (can be used as keywords)
            categories = []
            for category_elem in entry.findall('arxiv:primary_category', ns):
                term = category_elem.get('term')
                if term:
                    categories.append(term)
            for category_elem in entry.findall('atom:category', ns):
                term = category_elem.get('term')
                if term and term not in categories:
                    categories.append(term)
            
            if categories:
                # Convert ArXiv categories to readable keywords
                keyword_map = {
                    'cs.': 'Computer Science',
                    'math.': 'Mathematics',
                    'physics.': 'Physics',
                    'q-bio.': 'Quantitative Biology',
                    'q-fin.': 'Quantitative Finance',
                    'stat.': 'Statistics',
                    'eess.': 'Electrical Engineering and Systems Science',
                    'econ.': 'Economics'
                }
                
                keywords = []
                for cat in categories:
                    # Add the category itself
                    keywords.append(cat)
                    # Add a readable version if available
                    for prefix, readable in keyword_map.items():
                        if cat.startswith(prefix):
                            keywords.append(readable)
                            break
                
                result['keywords'] = list(set(keywords))  # Remove duplicates
                result['keyword_source'] = 'arxiv_categories'
            
            # Extract publication date
            published_elem = entry.find('atom:published', ns)
            if published_elem is not None:
                result['additional_metadata']['published_date'] = published_elem.text
            
            # Extract DOI if present (some ArXiv papers have DOIs)
            doi_elem = entry.find('arxiv:doi', ns)
            if doi_elem is not None:
                result['doi'] = doi_elem.text.strip()
            
            # Validate against known paper data
            validation_passed = True
            validation_notes = []
            
            # Check title similarity if we have both
            if result.get('additional_metadata', {}).get('extracted_title') and paper_data.get('title'):
                paper_title = paper_data['title'].lower()
                extracted_title = result['additional_metadata']['extracted_title'].lower()
                
                # Simple similarity check
                title_words = set(paper_title.split())
                extracted_words = set(extracted_title.split())
                if len(title_words) > 0:
                    overlap = len(title_words & extracted_words) / len(title_words)
                    if overlap < 0.5:
                        validation_notes.append("Title mismatch")
                        validation_passed = False
            
            result['validation_passed'] = validation_passed
            result['validation_notes'] = validation_notes
            
            return result
            
        except requests.exceptions.Timeout:
            if self.verbose >= 2:
                self.display.warning(f"ArXiv API timeout for {url}")
            return None
        except Exception as e:
            if self.verbose >= 2:
                self.display.warning(f"ArXiv extraction failed: {e}")
            return None
    
    async def _extract_from_hal(self, url: str, paper_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract metadata from HAL (Hyper Articles en Ligne).
        
        Args:
            url: HAL URL
            paper_data: Known paper data for validation
            
        Returns:
            Extracted metadata dictionary or None
        """
        try:
            if self.verbose >= 3:
                self.display.info("HAL extraction", f"Fetching metadata from {url[:50]}...", Icons.GLOBE)
            
            # Get timeout from config
            timeout = self.config.get('extraction.hal_timeout', 15)
            
            # Fetch HAL page
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'text/html,application/xhtml+xml',
                'Accept-Language': 'en-US,en;q=0.9'
            }
            
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            result = {
                'abstract': None,
                'keywords': [],
                'doi': None,
                'extraction_method': 'hal_scraping',
                'extraction_confidence': 0.85,
                'publisher': 'HAL',
                'additional_metadata': {
                    'hal_url': url
                }
            }
            
            # Extract HAL ID
            hal_match = re.search(r'/(hal-\d+)', url)
            if hal_match:
                result['additional_metadata']['hal_id'] = hal_match.group(1)
            
            # Try multiple strategies for abstract extraction
            abstract = None
            
            # Strategy 1: Look for meta tags
            abstract_meta = soup.find('meta', {'name': 'citation_abstract'}) or \
                          soup.find('meta', {'name': 'DC.Description'}) or \
                          soup.find('meta', {'property': 'og:description'})
            
            if abstract_meta and abstract_meta.get('content'):
                abstract = abstract_meta['content'].strip()
            
            # Strategy 2: Look for abstract in specific divs
            if not abstract:
                abstract_divs = soup.find_all('div', class_=['abstract', 'abstract-content', 'resume'])
                for div in abstract_divs:
                    text = div.get_text(strip=True)
                    if text and len(text) > 100:  # Likely to be an abstract
                        abstract = text
                        break
            
            # Strategy 3: Look for labeled sections
            if not abstract:
                # Look for "Abstract" or "Résumé" headers
                headers = soup.find_all(['h2', 'h3', 'h4', 'strong'])
                for header in headers:
                    header_text = header.get_text(strip=True).lower()
                    if header_text in ['abstract', 'résumé', 'summary']:
                        # Get the next sibling or parent's text
                        next_elem = header.find_next_sibling()
                        if next_elem:
                            abstract = next_elem.get_text(strip=True)
                        else:
                            parent = header.parent
                            if parent:
                                abstract = parent.get_text(strip=True)
                                # Remove the header text itself
                                abstract = abstract.replace(header.get_text(strip=True), '', 1).strip()
                        if abstract and len(abstract) > 100:
                            break
            
            if abstract:
                # Clean up the abstract - remove any HTML tags that might have been included
                if '<' in abstract and '>' in abstract:
                    soup_text = BeautifulSoup(abstract, 'html.parser')
                    abstract = soup_text.get_text(strip=True)
                # Clean up whitespace
                abstract = ' '.join(abstract.split())
                result['abstract'] = abstract
                
                if self.verbose >= 3:
                    self.display.info("HAL abstract", f"Extracted {len(abstract)} characters", Icons.CHECK)
            
            # Extract keywords
            keywords = []
            
            # Look for keywords in meta tags
            keyword_metas = soup.find_all('meta', {'name': ['citation_keywords', 'keywords', 'DC.Subject']})
            for meta in keyword_metas:
                content = meta.get('content', '')
                if content:
                    # Split by comma or semicolon
                    kws = re.split(r'[,;]', content)
                    keywords.extend([kw.strip() for kw in kws if kw.strip()])
            
            # Look for keyword lists in the page
            keyword_sections = soup.find_all(['div', 'span'], class_=['keywords', 'tags', 'subjects'])
            for section in keyword_sections:
                # Extract individual keywords
                kw_items = section.find_all(['a', 'span', 'li'])
                for item in kw_items:
                    kw = item.get_text(strip=True)
                    if kw and len(kw) < 50:  # Reasonable keyword length
                        keywords.append(kw)
            
            if keywords:
                # Remove duplicates and clean - HAL often has duplicates
                cleaned_keywords = []
                seen = set()
                for kw in keywords:
                    # Normalize for comparison (lowercase, strip)
                    kw_normalized = kw.strip().lower()
                    if kw_normalized and kw_normalized not in seen and len(kw.strip()) < 50:
                        cleaned_keywords.append(kw.strip())
                        seen.add(kw_normalized)
                
                result['keywords'] = cleaned_keywords[:20]  # Limit to 20 keywords
                result['keyword_source'] = 'hal_metadata'
            
            # Extract DOI if present
            doi_meta = soup.find('meta', {'name': 'citation_doi'})
            if doi_meta and doi_meta.get('content'):
                result['doi'] = doi_meta['content'].strip()
            else:
                # Look for DOI in links or text
                doi_pattern = re.compile(r'10\.\d{4,}/[-._;()/:\w]+')
                doi_text = soup.get_text()
                doi_match = doi_pattern.search(doi_text)
                if doi_match:
                    result['doi'] = doi_match.group(0)
            
            # Extract title for validation
            title_meta = soup.find('meta', {'name': 'citation_title'}) or \
                        soup.find('meta', {'property': 'og:title'})
            if title_meta and title_meta.get('content'):
                result['additional_metadata']['extracted_title'] = title_meta['content'].strip()
            
            # Extract authors
            authors = []
            author_metas = soup.find_all('meta', {'name': 'citation_author'})
            for meta in author_metas:
                author = meta.get('content', '').strip()
                if author:
                    authors.append(author)
            if authors:
                result['additional_metadata']['extracted_authors'] = authors
            
            # Extract publication date
            date_meta = soup.find('meta', {'name': 'citation_publication_date'}) or \
                       soup.find('meta', {'name': 'DC.Date'})
            if date_meta and date_meta.get('content'):
                result['additional_metadata']['publication_date'] = date_meta['content']
            
            # Validate extraction
            validation_passed = True
            validation_notes = []
            
            if not abstract:
                validation_notes.append("No abstract found")
                validation_passed = False
            
            # Check title similarity if available
            if result.get('additional_metadata', {}).get('extracted_title') and paper_data.get('title'):
                paper_title = paper_data['title'].lower()
                extracted_title = result['additional_metadata']['extracted_title'].lower()
                
                # Simple similarity check
                title_words = set(paper_title.split())
                extracted_words = set(extracted_title.split())
                if len(title_words) > 0:
                    overlap = len(title_words & extracted_words) / len(title_words)
                    if overlap < 0.5:
                        validation_notes.append("Title mismatch")
                        validation_passed = False
            
            result['validation_passed'] = validation_passed
            result['validation_notes'] = validation_notes
            
            return result
            
        except requests.exceptions.Timeout:
            if self.verbose >= 2:
                self.display.warning(f"HAL timeout for {url}")
            return None
        except Exception as e:
            if self.verbose >= 2:
                self.display.warning(f"HAL extraction failed: {e}")
            return None