# -*- coding: utf-8 -*-
"""
Paper web search module with publisher-specific strategies.

This module searches for academic papers using Google Custom Search API
with publisher-specific search strategies and LLM-based result analysis.

Author: Enhanced by Claude
Date: 2025-01-05
"""

import re
import json
import time
import requests
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor

from .config_manager import ConfigManager
from .llm_provider import LLMProviderManager
from .publisher_identifier import PublisherIdentifier
from .display_utils import TerminalDisplay, Icons


class PaperWebSearch:
    """Search for academic papers with publisher-specific strategies."""
    
    def __init__(self, config: ConfigManager, publisher_identifier: PublisherIdentifier,
                 cache_file: Optional[Path] = None, verbose: int = 2):
        """Initialize the paper search module.
        
        Args:
            config: Configuration manager
            publisher_identifier: Publisher identification module
            cache_file: Path to cache file for search results
            verbose: Verbosity level (0-3)
        """
        self.config = config
        self.publisher_id = publisher_identifier
        self.verbose = verbose
        self.display = TerminalDisplay(verbose=verbose)
        
        # Initialize LLM provider
        self.llm_manager = LLMProviderManager(config)
        
        # Get API keys
        self.google_api_key = config.get_api_key('google')
        self.google_cx = config.get('api_keys.google_cx')
        
        # Check if Google Custom Search is available
        self.has_google_search = bool(self.google_api_key and self.google_cx)
        if not self.has_google_search and verbose >= 2:
            self.display.warning("Google Custom Search not configured. Web search will be limited.")
        
        # Set up cache
        if cache_file is None:
            cache_dir = Path(__file__).parent.parent / "cache"
            cache_dir.mkdir(exist_ok=True)
            cache_file = cache_dir / ".paper_search_cache.json"
        self.cache_file = cache_file
        self.cache = self._load_cache()
        
        # Search configuration
        self.max_results_per_query = config.get('paper_search_settings.max_results_per_query', 10)
        self.use_llm_validation = config.get('paper_search_settings.use_llm_validation', True)
        self.use_smart_search = config.get('paper_search_settings.use_smart_single_search', True)
        self.blocked_publishers = config.get('paper_search_settings.blocked_publishers', [])
        self.domain_priority = config.get('paper_search_settings.domain_priority', {})
        self.trusted_domains = config.get('paper_search_settings.trusted_domains', [])
        
        # Statistics
        self.stats = {
            'searches_performed': 0,
            'cache_hits': 0,
            'urls_found': 0,
            'urls_validated': 0,
            'llm_validations': 0
        }
    
    def _load_cache(self) -> Dict[str, Any]:
        """Load cache from file."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                if self.verbose >= 2:
                    self.display.warning(f"Failed to load search cache: {e}")
                return {}
        return {}
    
    def _save_cache(self):
        """Save cache to file."""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            if self.verbose >= 2:
                self.display.warning(f"Failed to save search cache: {e}")
    
    async def search_paper(self, paper_data: Dict[str, Any], 
                          use_publisher_strategy: bool = True) -> Dict[str, Any]:
        """Search for a paper online.
        
        Args:
            paper_data: Dictionary containing paper metadata (title, authors, journal, doi, etc.)
            use_publisher_strategy: Whether to use publisher-specific search strategies
            
        Returns:
            Dictionary with search results including:
            - publisher_url: Direct URL to the paper on publisher's site
            - alternative_urls: Other URLs found (preprints, etc.)
            - search_confidence: Confidence score (0-1)
            - metadata_found: Any additional metadata found
        """
        title = paper_data.get('title', '')
        authors = paper_data.get('authors', [])
        journal = paper_data.get('journal', '')
        doi = paper_data.get('doi', '')
        year = paper_data.get('year', '')
        
        if not title:
            return self._empty_result()
        
        # Check cache
        cache_key = self._make_cache_key(title, doi)
        if cache_key in self.cache:
            self.stats['cache_hits'] += 1
            if self.verbose >= 3:
                self.display.info("Cache hit", f"Found cached results for: {title[:50]}...", Icons.CHECK)
            return self.cache[cache_key]
        
        # Identify publisher if using publisher strategy
        publisher = None
        platform = None
        publisher_confidence = 0.0
        
        if use_publisher_strategy and journal:
            publisher, platform, publisher_confidence = self.publisher_id.identify_publisher(journal, doi)
            if self.verbose >= 2 and publisher:
                self.display.info("Publisher identified", 
                                f"{publisher} ({platform}) - confidence: {publisher_confidence:.2f}",
                                Icons.SPARKLES)
        
        # Perform searches
        results = await self._perform_searches(paper_data, publisher, platform)
        
        # Cache and return results
        self.cache[cache_key] = results
        self._save_cache()
        
        return results
    
    async def _perform_searches(self, paper_data: Dict[str, Any], 
                              publisher: Optional[str], 
                              platform: Optional[str]) -> Dict[str, Any]:
        """Perform web searches for the paper."""
        self.stats['searches_performed'] += 1
        
        title = paper_data.get('title', '')
        authors = paper_data.get('authors', [])
        doi = paper_data.get('doi', '')
        year = paper_data.get('year', '')
        
        # Prepare search strategies
        strategies = []
        
        if self.use_smart_search:
            # Smart single-search strategy
            if doi:
                # If we have DOI, just search for it
                strategies.append({
                    'query': f'"{doi}"',
                    'platform': 'DOI Search',
                    'priority': 1.0
                })
            else:
                # Build comprehensive general search query
                query_parts = [f'"{title}"']
                
                # Add first author's last name for disambiguation
                if authors:
                    first_author = authors[0]
                    if isinstance(first_author, dict):
                        last_name = first_author.get('lastName', '')
                    else:
                        # Handle string format
                        last_name = str(first_author).split()[-1] if first_author else ''
                    if last_name:
                        query_parts.append(last_name)
                
                # Add year if available
                if year:
                    query_parts.append(str(year))
                
                strategies.append({
                    'query': ' '.join(query_parts),
                    'platform': 'General Search',
                    'priority': 1.0
                })
        else:
            # Old multi-search strategy (fallback)
            # Add DOI search if available
            if doi:
                strategies.append({
                    'query': f'"{doi}"',
                    'platform': 'DOI Search',
                    'priority': 1.0
                })
            
            # Add publisher-specific strategies
            if publisher and platform and publisher not in self.blocked_publishers:
                publisher_strategies = self.publisher_id.get_search_strategies(
                    publisher, platform, title, 
                    [self._format_author_name(a) for a in authors]
                )
                for strat in publisher_strategies:
                    strat['priority'] = 0.9 if 'site:' in strat['query'] else 0.7
                strategies.extend(publisher_strategies)
            
            # Add general academic search
            strategies.append({
                'query': f'"{title}" site:scholar.google.com',
                'platform': 'Google Scholar',
                'priority': 0.6
            })
        
        # Execute searches
        all_results = []
        if self.has_google_search:
            # Use asyncio for concurrent searches
            search_tasks = []
            for strategy in strategies[:3]:  # Limit concurrent searches
                if self.verbose >= 3:
                    self.display.info("Searching", f"{strategy['platform']}: {strategy['query'][:60]}...", Icons.SEARCH)
                search_tasks.append(self._execute_search(strategy))
            
            # Wait for all searches to complete
            search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            for results in search_results:
                if isinstance(results, Exception):
                    if self.verbose >= 2:
                        self.display.warning(f"Search failed: {results}")
                elif results:
                    all_results.extend(results)
        
        # Process and validate results
        processed_results = await self._process_search_results(
            all_results, paper_data, publisher
        )
        
        return processed_results
    
    async def _execute_search(self, strategy: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute a single search query."""
        if not self.has_google_search:
            return []
        
        query = strategy['query']
        
        # Use Google Custom Search API
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            'key': self.google_api_key,
            'cx': self.google_cx,
            'q': query,
            'num': self.max_results_per_query
        }
        
        try:
            if self.verbose >= 3:
                self.display.info("Executing search", f"{strategy['platform']}: {query[:60]}...", Icons.SEARCH)
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            items = data.get('items', [])
            
            if self.verbose >= 3:
                self.display.info("Search results", f"Found {len(items)} results", Icons.CHECK)
            
            results = []
            for item in items:
                results.append({
                    'url': item.get('link', ''),
                    'title': item.get('title', ''),
                    'snippet': item.get('snippet', ''),
                    'platform': strategy['platform'],
                    'priority': strategy.get('priority', 0.5)
                })
            
            return results
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                if self.verbose >= 1:
                    self.display.warning("Google API quota exceeded. Try again tomorrow.")
            else:
                if self.verbose >= 2:
                    self.display.warning(f"HTTP error {e.response.status_code}: {e}")
            return []
        except Exception as e:
            if self.verbose >= 2:
                self.display.warning(f"Search error: {e}")
            return []
    
    async def _process_search_results(self, results: List[Dict[str, Any]], 
                                    paper_data: Dict[str, Any],
                                    expected_publisher: Optional[str]) -> Dict[str, Any]:
        """Process and validate search results."""
        if not results:
            return self._empty_result()
        
        # Apply domain-based ranking
        for result in results:
            url = result.get('url', '')
            result['domain_score'] = self._calculate_domain_score(url)
        
        # Sort by combined priority and domain score
        results.sort(key=lambda x: x.get('priority', 0) * x.get('domain_score', 0.5), reverse=True)
        
        # Find best publisher URL
        publisher_url = None
        publisher_url_confidence = 0.0
        alternative_urls = []
        
        for result in results:
            url = result.get('url', '')
            if not url:
                continue
            
            self.stats['urls_found'] += 1
            
            # Check if this is a publisher URL
            is_publisher_url = False
            if expected_publisher:
                is_publisher_url = self.publisher_id.validate_publisher_url(url, expected_publisher)
            
            # Check if this is a trusted domain first
            is_trusted = self._is_trusted_domain(url)
            
            # Skip validation for highly trusted domains or use simplified validation
            if is_trusted and self._basic_url_validation(url, result.get('title', ''), paper_data):
                url_type = self._classify_url_type(url)
                confidence = 0.9 if is_trusted else 0.7
                
                if is_publisher_url and not publisher_url:
                    publisher_url = url
                    publisher_url_confidence = confidence
                else:
                    alternative_urls.append({
                        'url': url,
                        'type': url_type,
                        'confidence': confidence
                    })
                self.stats['urls_validated'] += 1
            elif self.use_llm_validation and self.llm_manager.is_available() and not is_trusted:
                # Use LLM validation only for non-trusted domains
                validation = await self._validate_url_with_llm(
                    url, result.get('title', ''), result.get('snippet', ''),
                    paper_data, is_publisher_url
                )
                
                if validation['is_correct_paper']:
                    if is_publisher_url and not publisher_url:
                        publisher_url = url
                        publisher_url_confidence = validation['confidence']
                    else:
                        alternative_urls.append({
                            'url': url,
                            'type': validation.get('url_type', 'unknown'),
                            'confidence': validation['confidence']
                        })
                    self.stats['urls_validated'] += 1
            else:
                # Basic validation without LLM
                if self._basic_url_validation(url, result.get('title', ''), paper_data):
                    if is_publisher_url and not publisher_url:
                        publisher_url = url
                        publisher_url_confidence = 0.7
                    else:
                        alternative_urls.append({
                            'url': url,
                            'type': 'search_result',
                            'confidence': 0.5
                        })
        
        # Calculate overall search confidence
        search_confidence = self._calculate_search_confidence(
            publisher_url, publisher_url_confidence, alternative_urls
        )
        
        return {
            'publisher_url': publisher_url,
            'publisher_url_confidence': publisher_url_confidence,
            'alternative_urls': alternative_urls[:5],  # Limit to top 5
            'search_confidence': search_confidence,
            'publisher_identified': expected_publisher,
            'search_strategy_used': 'publisher_specific' if expected_publisher else 'general',
            'total_results_found': len(results),
            'timestamp': time.time()
        }
    
    async def _validate_url_with_llm(self, url: str, title: str, snippet: str,
                                   paper_data: Dict[str, Any], 
                                   is_publisher_url: bool) -> Dict[str, Any]:
        """Use LLM to validate if a URL corresponds to the paper."""
        self.stats['llm_validations'] += 1
        
        paper_title = paper_data.get('title', '')
        paper_authors = paper_data.get('authors', [])
        paper_year = paper_data.get('year', '')
        
        # Format authors for comparison
        author_names = []
        for author in paper_authors[:3]:  # First 3 authors
            if isinstance(author, dict):
                name = f"{author.get('firstName', '')} {author.get('lastName', '')}".strip()
            else:
                name = str(author)
            if name:
                author_names.append(name)
        
        prompt = f"""Validate if this search result corresponds to the target academic paper.

TARGET PAPER:
Title: {paper_title}
Authors: {', '.join(author_names)} {' et al.' if len(paper_authors) > 3 else ''}
Year: {paper_year}

SEARCH RESULT:
URL: {url}
Title: {title}
Snippet: {snippet}
Is Publisher URL: {is_publisher_url}

Analyze if this is the same paper by comparing:
1. Title similarity (consider minor variations, formatting differences)
2. Author overlap (at least first author should match)
3. Year if mentioned
4. URL type (publisher site, preprint, repository, etc.)

Return ONLY a JSON object:
{{
    "is_correct_paper": true/false,
    "confidence": 0.0-1.0,
    "url_type": "publisher|preprint|repository|other",
    "title_match": "exact|close|partial|none",
    "author_match": "full|partial|none",
    "reasoning": "Brief explanation"
}}"""

        try:
            response = self.llm_manager.generate(
                prompt,
                temperature=0.1,
                max_tokens=512
            )
            
            if not response:
                return {'is_correct_paper': False, 'confidence': 0.0}
            
            # Parse JSON response
            json_text = response.strip()
            if '```json' in json_text:
                json_text = json_text.split('```json')[1].split('```')[0].strip()
            elif '```' in json_text:
                json_text = json_text.split('```')[1].split('```')[0].strip()
            
            result = json.loads(json_text)
            
            if self.verbose >= 3 and result.get('is_correct_paper'):
                self.display.info("URL validated", 
                                f"{result.get('url_type', 'unknown')} - confidence: {result.get('confidence', 0):.2f}",
                                Icons.CHECK)
                if result.get('reasoning'):
                    print(f"       Reasoning: {result['reasoning']}")
            
            return result
            
        except Exception as e:
            if self.verbose >= 2:
                self.display.warning(f"LLM validation failed: {e}")
            return {'is_correct_paper': False, 'confidence': 0.0}
    
    def _basic_url_validation(self, url: str, title: str, paper_data: Dict[str, Any]) -> bool:
        """Basic URL validation without LLM."""
        paper_title = paper_data.get('title', '').lower()
        search_title = title.lower()
        
        # Check for title similarity
        # Remove special characters for comparison
        clean_paper_title = re.sub(r'[^\w\s]', '', paper_title)
        clean_search_title = re.sub(r'[^\w\s]', '', search_title)
        
        # Check if most words match
        paper_words = set(clean_paper_title.split())
        search_words = set(clean_search_title.split())
        
        if len(paper_words) > 0:
            overlap = len(paper_words & search_words) / len(paper_words)
            return overlap > 0.5
        
        return False
    
    def _calculate_search_confidence(self, publisher_url: Optional[str],
                                   publisher_confidence: float,
                                   alternative_urls: List[Dict[str, Any]]) -> float:
        """Calculate overall search confidence."""
        if publisher_url and publisher_confidence > 0.8:
            return 0.95
        elif publisher_url and publisher_confidence > 0.6:
            return 0.85
        elif alternative_urls and any(u['confidence'] > 0.7 for u in alternative_urls):
            return 0.75
        elif alternative_urls:
            return 0.60
        else:
            return 0.0
    
    def _make_cache_key(self, title: str, doi: str) -> str:
        """Create cache key from paper identifiers."""
        # Normalize title
        normalized_title = re.sub(r'[^\w\s]', '', title.lower()).strip()
        return f"{doi}|{normalized_title}" if doi else normalized_title
    
    def _format_author_name(self, author: Any) -> str:
        """Format author name for search."""
        if isinstance(author, dict):
            return f"{author.get('firstName', '')} {author.get('lastName', '')}".strip()
        return str(author)
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty search result."""
        return {
            'publisher_url': None,
            'publisher_url_confidence': 0.0,
            'alternative_urls': [],
            'search_confidence': 0.0,
            'publisher_identified': None,
            'search_strategy_used': 'none',
            'total_results_found': 0,
            'timestamp': time.time()
        }
    
    def _calculate_domain_score(self, url: str) -> float:
        """Calculate domain priority score."""
        url_lower = url.lower()
        
        # Check against trusted domains
        for domain in self.trusted_domains:
            if domain in url_lower:
                # Determine type and return appropriate score
                if any(pub in url_lower for pub in ['sciencedirect.com', 'springer.com', 'nature.com', 'wiley.com', 'ieee.org']):
                    return self.domain_priority.get('publisher', 1.0)
                elif any(pre in url_lower for pre in ['arxiv.org', 'hal.science', 'hal.archives-ouvertes.fr']):
                    return self.domain_priority.get('preprint', 0.9)
                elif any(repo in url_lower for repo in ['researchgate.net', 'academia.edu']):
                    return self.domain_priority.get('repository', 0.8)
        
        return self.domain_priority.get('other', 0.5)
    
    def _is_trusted_domain(self, url: str) -> bool:
        """Check if URL is from a trusted domain."""
        url_lower = url.lower()
        return any(domain in url_lower for domain in self.trusted_domains)
    
    def _classify_url_type(self, url: str) -> str:
        """Classify URL type based on domain."""
        url_lower = url.lower()
        
        if any(pub in url_lower for pub in ['sciencedirect.com', 'springer.com', 'nature.com', 'wiley.com', 'ieee.org', 'iopscience.iop.org']):
            return 'publisher'
        elif any(pre in url_lower for pre in ['arxiv.org', 'hal.science', 'hal.archives-ouvertes.fr']):
            return 'preprint'
        elif any(repo in url_lower for repo in ['researchgate.net', 'academia.edu']):
            return 'repository'
        else:
            return 'other'
    
    def get_statistics(self) -> Dict[str, int]:
        """Get search statistics."""
        return self.stats.copy()
    
    async def extract_metadata_from_url(self, url: str, paper_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from a paper URL using LLM.
        
        Args:
            url: URL to extract metadata from
            paper_data: Known paper data for validation
            
        Returns:
            Dictionary with extracted metadata
        """
        if not url or not self.llm_manager.is_available():
            return {}
        
        # This will be implemented as part of the content extractor module
        # For now, return empty dict
        return {}