# -*- coding: utf-8 -*-
"""
Publisher identification module using LLM.

This module uses LLM to identify which publisher publishes a given journal,
enabling more targeted web searches for paper metadata.

Author: Enhanced by Claude
Date: 2025-01-05
"""

import json
import re
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any
import time

# Import LLM provider
from .llm_provider import LLMProviderManager
from .config_manager import ConfigManager
from .display_utils import TerminalDisplay, Icons


class PublisherIdentifier:
    """Identifies publishers for academic journals using LLM with caching."""
    
    # Known publisher patterns for common cases
    KNOWN_PUBLISHERS = {
        # Physics journals
        'phys. rev.': 'American Physical Society',
        'physical review': 'American Physical Society',
        'rev. mod. phys.': 'American Physical Society',
        'j. phys.': 'IOP Publishing',
        'classical and quantum gravity': 'IOP Publishing',
        'class. quantum gravity': 'IOP Publishing',
        'new j. phys.': 'IOP Publishing',
        'eur. phys. j.': 'Springer',
        'european physical journal': 'Springer',
        
        # Math/Computer Science
        'math. comput. simul.': 'Elsevier',
        'mathematics and computers in simulation': 'Elsevier',
        'j. comput. phys.': 'Elsevier',
        'comput. phys. commun.': 'Elsevier',
        
        # Earth Sciences
        'earth planet. sci. lett.': 'Elsevier',
        'earth and planetary science letters': 'Elsevier',
        
        # Engineering
        'case stud. therm. eng.': 'Elsevier',
        'case studies in thermal engineering': 'Elsevier',
        
        # General Science
        'nature': 'Nature Publishing Group',
        'science': 'American Association for the Advancement of Science',
        'pnas': 'National Academy of Sciences',
        'proceedings of the national academy': 'National Academy of Sciences',
        
        # Publishers with many journals
        'springer': 'Springer',
        'elsevier': 'Elsevier',
        'wiley': 'Wiley',
        'taylor & francis': 'Taylor & Francis',
        'sage': 'SAGE Publications',
        'oxford': 'Oxford University Press',
        'cambridge': 'Cambridge University Press',
        'ieee': 'IEEE',
    }
    
    # Publisher to search platform mapping
    PUBLISHER_PLATFORMS = {
        'Elsevier': 'ScienceDirect',
        'Springer': 'SpringerLink',
        'Nature Publishing Group': 'Nature.com',
        'Wiley': 'Wiley Online Library',
        'Taylor & Francis': 'Taylor & Francis Online',
        'IEEE': 'IEEE Xplore',
        'American Physical Society': 'APS Journals',
        'IOP Publishing': 'IOPscience',
        'SAGE Publications': 'SAGE Journals',
        'Oxford University Press': 'Oxford Academic',
        'Cambridge University Press': 'Cambridge Core',
        'American Chemical Society': 'ACS Publications',
        'Royal Society of Chemistry': 'RSC Publishing',
        'American Association for the Advancement of Science': 'Science.org',
        'National Academy of Sciences': 'PNAS.org'
    }
    
    # DOI prefix to publisher mapping
    DOI_PUBLISHERS = {
        '10.1007': 'Springer',
        '10.1016': 'Elsevier',
        '10.1002': 'Wiley',
        '10.1038': 'Nature Publishing Group',
        '10.1126': 'American Association for the Advancement of Science',
        '10.1073': 'National Academy of Sciences',
        '10.1103': 'American Physical Society',
        '10.1088': 'IOP Publishing',
        '10.1093': 'Oxford University Press',
        '10.1017': 'Cambridge University Press',
        '10.1021': 'American Chemical Society',
        '10.1039': 'Royal Society of Chemistry',
        '10.1080': 'Taylor & Francis',
        '10.1177': 'SAGE Publications',
        '10.1109': 'IEEE',
        '10.1145': 'ACM',
        '10.1137': 'SIAM',
        '10.1140': 'Springer', # European Physical Journal
        '10.1051': 'EDP Sciences',
        '10.1063': 'AIP Publishing',
        '10.1086': 'University of Chicago Press',
        '10.1111': 'Wiley',
        '10.1142': 'World Scientific',
        '10.1155': 'Hindawi',
        '10.1186': 'BioMed Central (Springer)',
        '10.1371': 'PLOS',
        '10.3390': 'MDPI',
        '10.4230': 'Schloss Dagstuhl',
        '10.5194': 'Copernicus Publications',
        '10.7554': 'eLife Sciences',
    }
    
    def __init__(self, config: ConfigManager, cache_file: Optional[Path] = None, 
                 verbose: int = 2):
        """Initialize the publisher identifier.
        
        Args:
            config: Configuration manager
            cache_file: Path to cache file for publisher mappings
            verbose: Verbosity level (0-3)
        """
        self.config = config
        self.verbose = verbose
        self.display = TerminalDisplay(verbose=verbose)
        
        # Initialize LLM provider manager
        self.llm_manager = LLMProviderManager(config)
        
        # Set up cache
        if cache_file is None:
            cache_dir = Path(__file__).parent.parent / "cache"
            cache_dir.mkdir(exist_ok=True)
            cache_file = cache_dir / ".publisher_cache.json"
        self.cache_file = cache_file
        self.cache = self._load_cache()
        
        # Track statistics
        self.stats = {
            'cache_hits': 0,
            'known_pattern_hits': 0,
            'doi_hits': 0,
            'llm_queries': 0,
            'llm_failures': 0
        }
    
    def _load_cache(self) -> Dict[str, Dict[str, Any]]:
        """Load cache from file."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                if self.verbose >= 2:
                    self.display.warning(f"Failed to load publisher cache: {e}")
                return {}
        return {}
    
    def _save_cache(self):
        """Save cache to file."""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            if self.verbose >= 2:
                self.display.warning(f"Failed to save publisher cache: {e}")
    
    def identify_publisher(self, journal_name: str, doi: Optional[str] = None,
                         use_llm: bool = True) -> Tuple[Optional[str], Optional[str], float]:
        """Identify the publisher of a journal.
        
        Args:
            journal_name: Name of the journal (can be abbreviated)
            doi: Optional DOI of the paper
            use_llm: Whether to use LLM for identification
            
        Returns:
            Tuple of (publisher_name, search_platform, confidence_score)
        """
        if not journal_name:
            return None, None, 0.0
        
        # Normalize journal name for matching
        normalized_journal = journal_name.lower().strip()
        
        # Check cache first
        cache_key = f"{normalized_journal}|{doi or ''}"
        if cache_key in self.cache:
            self.stats['cache_hits'] += 1
            cached = self.cache[cache_key]
            return cached['publisher'], cached['platform'], cached['confidence']
        
        # Try DOI-based identification first (highest confidence)
        if doi:
            publisher, platform, confidence = self._identify_by_doi(doi)
            if publisher:
                self.stats['doi_hits'] += 1
                self._cache_result(cache_key, publisher, platform, confidence)
                return publisher, platform, confidence
        
        # Try known patterns (high confidence)
        publisher, platform, confidence = self._identify_by_pattern(normalized_journal)
        if publisher:
            self.stats['known_pattern_hits'] += 1
            self._cache_result(cache_key, publisher, platform, confidence)
            return publisher, platform, confidence
        
        # Use LLM for identification (medium confidence)
        if use_llm and self.llm_manager.is_available():
            publisher, platform, confidence = self._identify_by_llm(journal_name)
            if publisher:
                self.stats['llm_queries'] += 1
                self._cache_result(cache_key, publisher, platform, confidence)
                return publisher, platform, confidence
            else:
                self.stats['llm_failures'] += 1
        
        return None, None, 0.0
    
    def _identify_by_doi(self, doi: str) -> Tuple[Optional[str], Optional[str], float]:
        """Identify publisher by DOI prefix."""
        # Extract DOI prefix
        doi_match = re.match(r'(10\.\d{4,5})', doi)
        if not doi_match:
            return None, None, 0.0
        
        prefix = doi_match.group(1)
        
        # Check known DOI publishers
        for doi_prefix, publisher in self.DOI_PUBLISHERS.items():
            if prefix.startswith(doi_prefix):
                platform = self.PUBLISHER_PLATFORMS.get(publisher, publisher)
                return publisher, platform, 0.95  # High confidence
        
        return None, None, 0.0
    
    def _identify_by_pattern(self, normalized_journal: str) -> Tuple[Optional[str], Optional[str], float]:
        """Identify publisher by known journal patterns."""
        # Check exact and partial matches
        for pattern, publisher in self.KNOWN_PUBLISHERS.items():
            if pattern in normalized_journal:
                platform = self.PUBLISHER_PLATFORMS.get(publisher, publisher)
                # Higher confidence for longer matches
                confidence = 0.9 if len(pattern) > 10 else 0.85
                return publisher, platform, confidence
        
        return None, None, 0.0
    
    def _identify_by_llm(self, journal_name: str) -> Tuple[Optional[str], Optional[str], float]:
        """Use LLM to identify the publisher."""
        prompt = f"""Identify the academic publisher for this journal: "{journal_name}"

Please analyze the journal name and determine:
1. The publisher (e.g., Elsevier, Springer, Wiley, IEEE, Nature Publishing Group, etc.)
2. The online platform where papers are found (e.g., ScienceDirect for Elsevier, SpringerLink for Springer)
3. Your confidence level (0.0-1.0)

Consider common abbreviations:
- "Phys. Rev." = Physical Review (APS)
- "J." = Journal
- "Int." = International
- "Comput." = Computer/Computational
- "Sci." = Science/Scientific
- "Eng." = Engineering
- "Lett." = Letters

Return ONLY a JSON object with this exact format:
{{
    "publisher": "Publisher Name",
    "platform": "Platform Name",
    "confidence": 0.8,
    "reasoning": "Brief explanation"
}}

If you cannot identify the publisher with reasonable confidence, return:
{{
    "publisher": null,
    "platform": null,
    "confidence": 0.0,
    "reasoning": "Unable to identify publisher"
}}"""

        try:
            # Use LLM with fallback
            response = self.llm_manager.generate(
                prompt,
                temperature=0.1,  # Low temperature for factual response
                max_tokens=512
            )
            
            if not response:
                return None, None, 0.0
            
            # Parse JSON response
            # Extract JSON from response (handle markdown code blocks)
            json_text = response.strip()
            if '```json' in json_text:
                json_text = json_text.split('```json')[1].split('```')[0].strip()
            elif '```' in json_text:
                json_text = json_text.split('```')[1].split('```')[0].strip()
            
            result = json.loads(json_text)
            
            # Validate response
            if result.get('publisher') and result.get('confidence', 0) > 0.5:
                publisher = result['publisher']
                platform = result.get('platform') or self.PUBLISHER_PLATFORMS.get(publisher, publisher)
                confidence = min(result.get('confidence', 0.7), 0.85)  # Cap LLM confidence
                
                if self.verbose >= 3:
                    self.display.info("LLM Publisher ID", 
                                    f"{publisher} ({platform}) - confidence: {confidence:.2f}",
                                    Icons.SPARKLES)
                    if result.get('reasoning'):
                        print(f"       Reasoning: {result['reasoning']}")
                
                return publisher, platform, confidence
            
        except Exception as e:
            if self.verbose >= 2:
                self.display.warning(f"LLM publisher identification failed: {e}")
        
        return None, None, 0.0
    
    def _cache_result(self, cache_key: str, publisher: str, platform: str, confidence: float):
        """Cache a publisher identification result."""
        self.cache[cache_key] = {
            'publisher': publisher,
            'platform': platform,
            'confidence': confidence,
            'timestamp': time.time()
        }
        self._save_cache()
    
    def get_search_strategies(self, publisher: str, platform: str, 
                            paper_title: str, authors: List[str]) -> List[Dict[str, Any]]:
        """Get publisher-specific search strategies.
        
        Args:
            publisher: Publisher name
            platform: Search platform name
            paper_title: Title of the paper
            authors: List of author names
            
        Returns:
            List of search strategies with queries and expected URL patterns
        """
        strategies = []
        
        # Clean up title for search
        clean_title = re.sub(r'[^\w\s-]', ' ', paper_title).strip()
        title_words = clean_title.split()[:5]  # First 5 words
        short_title = ' '.join(title_words)
        
        # First author last name
        first_author = ""
        if authors and len(authors) > 0:
            first_author = authors[0].split()[-1] if isinstance(authors[0], str) else authors[0].get('lastName', '')
        
        # Publisher-specific strategies
        if publisher == 'Elsevier':
            strategies.extend([
                {
                    'query': f'site:sciencedirect.com "{paper_title}"',
                    'platform': 'ScienceDirect',
                    'url_pattern': r'sciencedirect\.com/science/article'
                },
                {
                    'query': f'site:sciencedirect.com {short_title} {first_author}',
                    'platform': 'ScienceDirect',
                    'url_pattern': r'sciencedirect\.com/science/article'
                }
            ])
        
        elif publisher == 'Springer':
            strategies.extend([
                {
                    'query': f'site:link.springer.com "{paper_title}"',
                    'platform': 'SpringerLink',
                    'url_pattern': r'link\.springer\.com/article'
                },
                {
                    'query': f'site:springer.com {short_title} {first_author}',
                    'platform': 'SpringerLink',
                    'url_pattern': r'link\.springer\.com/article'
                }
            ])
        
        elif publisher == 'Wiley':
            strategies.extend([
                {
                    'query': f'site:onlinelibrary.wiley.com "{paper_title}"',
                    'platform': 'Wiley Online Library',
                    'url_pattern': r'onlinelibrary\.wiley\.com/doi'
                }
            ])
        
        elif publisher == 'Nature Publishing Group':
            strategies.extend([
                {
                    'query': f'site:nature.com "{paper_title}"',
                    'platform': 'Nature.com',
                    'url_pattern': r'nature\.com/articles'
                }
            ])
        
        elif publisher == 'American Physical Society':
            strategies.extend([
                {
                    'query': f'site:journals.aps.org "{paper_title}"',
                    'platform': 'APS Journals',
                    'url_pattern': r'journals\.aps\.org/.*/PhysRev'
                }
            ])
        
        elif publisher == 'IOP Publishing':
            strategies.extend([
                {
                    'query': f'site:iopscience.iop.org "{paper_title}"',
                    'platform': 'IOPscience',
                    'url_pattern': r'iopscience\.iop\.org/article'
                }
            ])
        
        elif publisher == 'Taylor & Francis':
            strategies.extend([
                {
                    'query': f'site:tandfonline.com "{paper_title}"',
                    'platform': 'Taylor & Francis Online',
                    'url_pattern': r'tandfonline\.com/doi'
                }
            ])
        
        elif publisher == 'IEEE':
            strategies.extend([
                {
                    'query': f'site:ieeexplore.ieee.org "{paper_title}"',
                    'platform': 'IEEE Xplore',
                    'url_pattern': r'ieeexplore\.ieee\.org/document'
                }
            ])
        
        # Add generic strategy as fallback
        strategies.append({
            'query': f'"{paper_title}" {first_author} {publisher}',
            'platform': 'General Search',
            'url_pattern': None
        })
        
        return strategies
    
    def get_statistics(self) -> Dict[str, int]:
        """Get usage statistics."""
        return self.stats.copy()
    
    def validate_publisher_url(self, url: str, publisher: str) -> bool:
        """Validate if a URL belongs to the expected publisher.
        
        Args:
            url: URL to validate
            publisher: Expected publisher
            
        Returns:
            True if URL matches expected publisher patterns
        """
        if not url or not publisher:
            return False
        
        url_lower = url.lower()
        
        # Publisher-specific URL patterns
        publisher_urls = {
            'Elsevier': ['sciencedirect.com', 'elsevier.com'],
            'Springer': ['springer.com', 'link.springer.com', 'springerlink.com'],
            'Wiley': ['wiley.com', 'onlinelibrary.wiley.com'],
            'Nature Publishing Group': ['nature.com'],
            'American Physical Society': ['aps.org', 'journals.aps.org'],
            'IOP Publishing': ['iop.org', 'iopscience.iop.org'],
            'Taylor & Francis': ['tandfonline.com', 'taylorandfrancis.com'],
            'IEEE': ['ieee.org', 'ieeexplore.ieee.org'],
            'Oxford University Press': ['oxford.com', 'academic.oup.com'],
            'Cambridge University Press': ['cambridge.org'],
            'SAGE Publications': ['sagepub.com'],
            'American Chemical Society': ['acs.org', 'pubs.acs.org'],
            'Royal Society of Chemistry': ['rsc.org', 'pubs.rsc.org'],
            'American Association for the Advancement of Science': ['science.org', 'sciencemag.org'],
            'National Academy of Sciences': ['pnas.org']
        }
        
        expected_domains = publisher_urls.get(publisher, [])
        return any(domain in url_lower for domain in expected_domains)