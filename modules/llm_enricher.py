# -*- coding: utf-8 -*-
"""
LLM-based metadata enricher using Claude API.

This module provides intelligent metadata extraction from web pages and
generation of keywords using Claude's language understanding capabilities.

Author: Dr. Denys Dutykh (Khalifa University of Science and Technology, Abu Dhabi, UAE)
Date: 2025-01-01
"""

import json
import hashlib
import re
import asyncio
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote_plus

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


class LLMMetadataEnricher:
    """Enricher for bibliography metadata using Claude LLM."""
    
    # Claude model to use
    MODEL = "claude-3-haiku-20240307"  # Fast and cost-effective for extraction tasks
    
    # System prompts for different tasks
    SEARCH_PROMPT = """You are an expert at finding academic publications online. Given publication details, generate optimal search queries and analyze search results to find the correct publication.

When generating search queries:
1. Include the full title in quotes for exact matching
2. Add key author names (last names)
3. Include publication year
4. Add publisher or book title if relevant
5. Generate multiple query variations for better results

When analyzing results:
1. Compare titles carefully (accounting for minor variations)
2. Check author names match (at least primary authors)
3. Verify publication year (might be off by 1-2 years)
4. Consider publisher/venue information
5. Return confidence score (0-1) for each match"""

    EXTRACT_PROMPT = """You are an expert at extracting academic metadata from HTML pages. Extract ALL available metadata from the provided HTML content.

Extract the following if available:
- DOI (Digital Object Identifier) - both chapter DOI and book DOI if present
- Abstract (full text, clean up formatting)
- Keywords (as array of strings) - look for all keyword sections
- Publisher URL (the canonical URL for this publication)
- ISBN (both print and electronic if available)
- Publication date (online and print dates)
- Author affiliations (institution, department, country)
- Citation information (how to cite this work)
- Book series information (series name, volume, ISSN)
- Chapter information (chapter number, part)
- Editor information
- Page numbers (start and end)
- Copyright information
- License information
- Funding/acknowledgments
- Related content URLs
- Download statistics if shown
- Any subject classifications (MSC, PACS, etc.)

Important:
1. Look EVERYWHERE: meta tags, JSON-LD, visible text, script tags, data attributes
2. Extract from multiple keyword sections: Keywords, Key Terms, Index Terms, Subject, Topics
3. Clean up extracted text (remove HTML tags, fix spacing, decode entities)
4. For Springer chapters, check for "Keywords" section after abstract
5. Return comprehensive metadata - we want EVERYTHING available
6. Return ONLY valid JSON with all found fields"""

    KEYWORD_GEN_PROMPT = """You are an expert at generating academic keywords for publications. Based on the title and abstract provided, generate relevant keywords that would help others find this publication.

Guidelines:
1. Generate 5-8 keywords or short phrases
2. Include both specific terms and broader concepts
3. Use standard academic terminology
4. Consider the field and subfield
5. Include methodologies, applications, and key concepts
6. Make keywords useful for indexing and discovery

Return keywords as a JSON array of strings."""
    
    def __init__(self, config, cache_file: Path = None, verbose: int = 2):
        """
        Initialize the LLM enricher.
        
        Args:
            config: Configuration manager instance
            cache_file: Path to cache file (optional)
            verbose: Verbosity level (0-3)
        """
        self.config = config
        self.verbose = verbose
        self.cache_file = cache_file or Path(".llm_enricher_cache.json")
        self.cache = self._load_cache()
        
        # Initialize Claude client if available
        self.client = None
        self.enabled = False
        
        if ANTHROPIC_AVAILABLE and config:
            api_key = config.get_api_key('anthropic')
            if api_key:
                self.client = Anthropic(api_key=api_key)
                self.enabled = config.get('preferences.use_llm_enricher', True)
                
                if self.verbose >= 2:
                    print("[LLM_ENRICHER] Claude API initialized for metadata extraction")
            elif self.verbose >= 1:
                print("[LLM_ENRICHER] No Anthropic API key found")
        elif not ANTHROPIC_AVAILABLE and self.verbose >= 1:
            print("[LLM_ENRICHER] Anthropic library not installed")
    
    def _load_cache(self) -> Dict[str, Any]:
        """Load cache from file."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
                    if self.verbose >= 3:
                        print(f"[LLM_ENRICHER] Loaded cache with {len(cache)} entries")
                    return cache
            except Exception as e:
                if self.verbose >= 1:
                    print(f"[LLM_ENRICHER] Error loading cache: {e}")
        return {}
    
    def _save_cache(self) -> None:
        """Save cache to file."""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
                if self.verbose >= 3:
                    print(f"[LLM_ENRICHER] Saved cache with {len(self.cache)} entries")
        except Exception as e:
            if self.verbose >= 1:
                print(f"[LLM_ENRICHER] Error saving cache: {e}")
    
    def _get_cache_key(self, operation: str, content: str) -> str:
        """Generate cache key from operation and content."""
        combined = f"{operation}:{content}"
        return hashlib.md5(combined.encode('utf-8')).hexdigest()
    
    def _is_cache_valid(self, cache_entry: Dict) -> bool:
        """Check if cache entry is still valid."""
        if 'timestamp' not in cache_entry:
            return False
        
        cache_days = self.config.get('preferences.llm_enricher_cache_days', 30) if self.config else 30
        cache_time = datetime.fromisoformat(cache_entry['timestamp'])
        expiry_time = cache_time + timedelta(days=cache_days)
        
        return datetime.now() < expiry_time
    
    async def generate_search_queries(self, entry: Dict[str, Any]) -> List[str]:
        """
        Generate optimal search queries for finding a publication.
        
        Args:
            entry: Bibliography entry with title, authors, etc.
            
        Returns:
            List of search queries to try
        """
        if not self.enabled or not self.client:
            # Fallback to simple query generation
            title = entry.get('title', '')
            authors = entry.get('authors', [])
            year = entry.get('year', '')
            
            queries = []
            if title:
                # Basic query with title
                queries.append(f'"{title}"')
                
                # Add author and year if available
                if authors and year:
                    first_author = authors[0].get('lastName', '')
                    queries.append(f'"{title}" {first_author} {year}')
                elif authors:
                    first_author = authors[0].get('lastName', '')
                    queries.append(f'"{title}" {first_author}')
            
            return queries
        
        # Check cache
        cache_key = self._get_cache_key('search_queries', str(entry))
        if cache_key in self.cache:
            cache_entry = self.cache[cache_key]
            if self._is_cache_valid(cache_entry):
                if self.verbose >= 3:
                    print("[LLM_ENRICHER] Using cached search queries")
                return cache_entry['result']
        
        # Prepare prompt
        entry_info = {
            'title': entry.get('title', ''),
            'authors': [f"{a.get('firstName', '')} {a.get('lastName', '')}".strip() 
                       for a in entry.get('authors', [])],
            'year': entry.get('year', ''),
            'book_title': entry.get('book_title', ''),
            'publisher': entry.get('publisher', {}).get('name', ''),
            'series': entry.get('series', '')
        }
        
        prompt = f"""Generate search queries for this publication:
{json.dumps(entry_info, indent=2)}

Return a JSON array of 3-5 search query strings, ordered from most to least specific."""
        
        try:
            response = self.client.messages.create(
                model=self.MODEL,
                max_tokens=500,
                temperature=0,
                system=self.SEARCH_PROMPT,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Extract JSON from response
            json_text = response.content[0].text.strip()
            if json_text.startswith("```json"):
                json_text = json_text[7:]
            if json_text.endswith("```"):
                json_text = json_text[:-3]
            
            try:
                queries = json.loads(json_text.strip())
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract array from the text
                import re
                array_match = re.search(r'\[.*?\]', json_text, re.DOTALL)
                if array_match:
                    queries = json.loads(array_match.group(0))
                else:
                    raise
            
            # Cache the result
            self.cache[cache_key] = {
                'result': queries,
                'timestamp': datetime.now().isoformat()
            }
            self._save_cache()
            
            if self.verbose >= 2:
                print(f"[LLM_ENRICHER] Generated {len(queries)} search queries")
                if self.verbose >= 3:
                    for i, q in enumerate(queries[:3]):
                        print(f"  Query {i+1}: {q}")
            
            return queries
            
        except Exception as e:
            if self.verbose >= 1:
                print(f"[LLM_ENRICHER] Error generating search queries: {e}")
            # Return fallback queries
            return [f'"{entry.get("title", "")}"'] if entry.get('title') else []
    
    async def extract_metadata_from_html(self, html: str, url: str = "") -> Dict[str, Any]:
        """
        Extract metadata from HTML content using LLM.
        
        Args:
            html: HTML content of the page
            url: URL of the page (for context)
            
        Returns:
            Dictionary of extracted metadata
        """
        if not self.enabled or not self.client:
            return {}
        
        # Truncate HTML if too long (keep important parts)
        max_length = 50000
        if len(html) > max_length:
            # Try to keep head and main content
            head_match = re.search(r'<head.*?</head>', html, re.DOTALL | re.IGNORECASE)
            head_content = head_match.group(0) if head_match else ""
            
            # Look for main content areas
            content_markers = ['abstract', 'keywords', 'article', 'main', 'content']
            body_content = ""
            for marker in content_markers:
                pattern = rf'<[^>]*(?:class|id)="[^"]*{marker}[^"]*"[^>]*>.*?</\w+>'
                matches = re.findall(pattern, html, re.DOTALL | re.IGNORECASE)
                body_content += " ".join(matches[:2])  # Take first 2 matches
            
            html = head_content + body_content
            if len(html) > max_length:
                html = html[:max_length]
        
        # Check cache
        cache_key = self._get_cache_key('extract_html', f"{url}:{html[:1000]}")
        if cache_key in self.cache:
            cache_entry = self.cache[cache_key]
            if self._is_cache_valid(cache_entry):
                if self.verbose >= 3:
                    print("[LLM_ENRICHER] Using cached HTML extraction")
                return cache_entry['result']
        
        prompt = f"""Extract metadata from this HTML page:

URL: {url}

HTML:
{html}

Focus on academic metadata. Return valid JSON only."""
        
        try:
            if self.verbose >= 3:
                print("[LLM_ENRICHER] Sending HTML to Claude for extraction")
            
            response = self.client.messages.create(
                model=self.MODEL,
                max_tokens=1500,
                temperature=0,
                system=self.EXTRACT_PROMPT,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Extract JSON from response
            json_text = response.content[0].text.strip()
            if json_text.startswith("```json"):
                json_text = json_text[7:]
            if json_text.endswith("```"):
                json_text = json_text[:-3]
            
            metadata = json.loads(json_text.strip())
            
            # Normalize and extract all metadata
            result = {
                'doi': metadata.get('doi', metadata.get('DOI', '')),
                'abstract': metadata.get('abstract', ''),
                'keywords': metadata.get('keywords', []),
                'publisher_url': metadata.get('publisher_url', url),
                'isbn': metadata.get('isbn', metadata.get('ISBN', '')),
                'keyword_source': 'publisher' if metadata.get('keywords') else 'none',
                # Additional metadata fields
                'book_doi': metadata.get('book_doi', ''),
                'electronic_isbn': metadata.get('electronic_isbn', ''),
                'print_isbn': metadata.get('print_isbn', ''),
                'publication_date': metadata.get('publication_date', ''),
                'online_date': metadata.get('online_date', ''),
                'author_affiliations': metadata.get('author_affiliations', []),
                'citation_text': metadata.get('citation_information', ''),
                'series_issn': metadata.get('series_issn', ''),
                'chapter_number': metadata.get('chapter_number', ''),
                'editor_info': metadata.get('editor_information', []),
                'copyright': metadata.get('copyright_information', ''),
                'license': metadata.get('license_information', ''),
                'funding': metadata.get('funding', ''),
                'related_content': metadata.get('related_content_urls', []),
                'download_stats': metadata.get('download_statistics', ''),
                'subject_classifications': metadata.get('subject_classifications', []),
                'book_series_info': metadata.get('book_series_information', {})
            }
            
            # Clean up empty fields
            result = {k: v for k, v in result.items() if v}
            
            # Cache the result
            self.cache[cache_key] = {
                'result': result,
                'timestamp': datetime.now().isoformat()
            }
            self._save_cache()
            
            if self.verbose >= 2:
                print("[LLM_ENRICHER] Successfully extracted metadata")
                if result.get('doi'):
                    print(f"  - DOI: {result['doi']}")
                if result.get('abstract'):
                    print(f"  - Abstract: {len(result['abstract'])} chars")
                if result.get('keywords'):
                    print(f"  - Keywords: {len(result['keywords'])} found")
            
            return result
            
        except Exception as e:
            if self.verbose >= 1:
                print(f"[LLM_ENRICHER] Error extracting metadata: {e}")
            return {}
    
    async def match_search_results(self, entry: Dict[str, Any], 
                                  search_results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Use LLM to find the best match from search results.
        
        Args:
            entry: Original bibliography entry
            search_results: List of search results to analyze
            
        Returns:
            Best matching result with confidence score, or None
        """
        if not self.enabled or not self.client or not search_results:
            return None
        
        # Prepare entry info
        entry_info = {
            'title': entry.get('title', ''),
            'authors': [f"{a.get('firstName', '')} {a.get('lastName', '')}".strip() 
                       for a in entry.get('authors', [])],
            'year': entry.get('year', ''),
            'book_title': entry.get('book_title', ''),
            'publisher': entry.get('publisher', {}).get('name', '')
        }
        
        prompt = f"""Find the best match for this publication:
{json.dumps(entry_info, indent=2)}

From these search results:
{json.dumps(search_results, indent=2)}

Return JSON with:
- best_match_index: index of best match (0-based) or -1 if no good match
- confidence: confidence score (0-1)
- reason: brief explanation"""
        
        try:
            response = self.client.messages.create(
                model=self.MODEL,
                max_tokens=500,
                temperature=0,
                system=self.SEARCH_PROMPT,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Extract JSON from response
            json_text = response.content[0].text.strip()
            if json_text.startswith("```json"):
                json_text = json_text[7:]
            if json_text.endswith("```"):
                json_text = json_text[:-3]
            
            match_info = json.loads(json_text.strip())
            
            if match_info['best_match_index'] >= 0 and match_info['confidence'] > 0.7:
                best_match = search_results[match_info['best_match_index']]
                best_match['match_confidence'] = match_info['confidence']
                
                if self.verbose >= 2:
                    print(f"[LLM_ENRICHER] Found match with {match_info['confidence']:.2f} confidence")
                    if self.verbose >= 3:
                        print(f"  Reason: {match_info['reason']}")
                
                return best_match
            
            return None
            
        except Exception as e:
            if self.verbose >= 1:
                print(f"[LLM_ENRICHER] Error matching results: {e}")
            return None
    
    async def generate_keywords(self, title: str, abstract: str = "") -> List[str]:
        """
        Generate keywords from title and abstract using LLM.
        
        Args:
            title: Publication title
            abstract: Publication abstract (optional)
            
        Returns:
            List of generated keywords
        """
        if not self.enabled or not self.client or not title:
            return []
        
        # Check cache
        cache_content = f"{title}:{abstract[:500] if abstract else ''}"
        cache_key = self._get_cache_key('keywords', cache_content)
        if cache_key in self.cache:
            cache_entry = self.cache[cache_key]
            if self._is_cache_valid(cache_entry):
                if self.verbose >= 3:
                    print("[LLM_ENRICHER] Using cached keywords")
                return cache_entry['result']
        
        prompt = f"""Generate keywords for this publication:

Title: {title}

Abstract: {abstract if abstract else "Not available"}

Generate appropriate academic keywords."""
        
        try:
            if self.verbose >= 3:
                print("[LLM_ENRICHER] Generating keywords with Claude")
            
            response = self.client.messages.create(
                model=self.MODEL,
                max_tokens=300,
                temperature=0.3,  # Slight creativity for keyword variety
                system=self.KEYWORD_GEN_PROMPT,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Extract JSON from response
            json_text = response.content[0].text.strip()
            if json_text.startswith("```json"):
                json_text = json_text[7:]
            if json_text.endswith("```"):
                json_text = json_text[:-3]
            
            keywords = json.loads(json_text.strip())
            
            # Ensure it's a list of strings
            if isinstance(keywords, list):
                keywords = [str(k).strip() for k in keywords if k]
            else:
                keywords = []
            
            # Cache the result
            self.cache[cache_key] = {
                'result': keywords,
                'timestamp': datetime.now().isoformat()
            }
            self._save_cache()
            
            if self.verbose >= 2:
                print(f"[LLM_ENRICHER] Generated {len(keywords)} keywords")
                if self.verbose >= 3 and keywords:
                    print(f"  Keywords: {', '.join(keywords[:5])}")
                    if len(keywords) > 5:
                        print(f"  ... and {len(keywords) - 5} more")
            
            return keywords
            
        except Exception as e:
            if self.verbose >= 1:
                print(f"[LLM_ENRICHER] Error generating keywords: {e}")
            return []
    
    async def search_and_extract(self, entry: Dict[str, Any], 
                               search_func, fetch_func) -> Dict[str, Any]:
        """
        Complete workflow: search, match, and extract metadata.
        
        Args:
            entry: Bibliography entry
            search_func: Async function to perform search
            fetch_func: Async function to fetch page content
            
        Returns:
            Extracted metadata dictionary
        """
        metadata = {
            'doi': '',
            'abstract': '',
            'keywords': [],
            'publisher_url': '',
            'isbn': '',
            'keyword_source': 'none'
        }
        
        try:
            # Generate search queries
            queries = await self.generate_search_queries(entry)
            
            # Try each query
            for query in queries:
                if self.verbose >= 3:
                    print(f"[LLM_ENRICHER] Trying query: {query}")
                
                # Perform search
                search_results = await search_func(query)
                
                if not search_results:
                    if self.verbose >= 3:
                        print(f"[LLM_ENRICHER] No results for query: {query[:50]}...")
                    continue
                
                # Find best match
                best_match = await self.match_search_results(entry, search_results)
                
                if best_match:
                    # First, extract DOI directly from search result if available
                    if best_match.get('doi') and not metadata.get('doi'):
                        metadata['doi'] = best_match['doi']
                        if self.verbose >= 2:
                            print(f"[LLM_ENRICHER] Found DOI from search: {metadata['doi']}")
                    
                    # Determine URL to fetch
                    url_to_fetch = best_match.get('url', '')
                    
                    # If we have DOI but no URL or a generic URL, convert DOI to publisher URL
                    if metadata.get('doi'):
                        if not url_to_fetch or not url_to_fetch.startswith('http') or 'doi.org' in url_to_fetch:
                            # Convert DOI to publisher URL directly
                            publisher_url = self._convert_doi_to_publisher_url(metadata['doi'])
                            if publisher_url:
                                url_to_fetch = publisher_url
                                if self.verbose >= 3:
                                    print(f"[LLM_ENRICHER] Converted DOI to publisher URL: {url_to_fetch}")
                    
                    # Fetch and extract metadata if we have a URL
                    if url_to_fetch and url_to_fetch.startswith('http'):
                        # Always fetch to try to get abstract
                        if self.verbose >= 3:
                            print(f"[LLM_ENRICHER] Fetching page: {url_to_fetch}")
                        
                        try:
                            html = await fetch_func(url_to_fetch)
                            
                            if html:
                                if self.verbose >= 3:
                                    print(f"[LLM_ENRICHER] Got HTML ({len(html)} chars), extracting metadata...")
                                
                                extracted = await self.extract_metadata_from_html(html, url_to_fetch)
                                
                                # Update metadata with all extracted data
                                if extracted:
                                    # Core fields
                                    for key in ['doi', 'abstract', 'keywords', 'publisher_url', 'isbn']:
                                        if extracted.get(key):
                                            if key == 'keywords' and metadata.get('keywords'):
                                                # Don't overwrite existing keywords unless from publisher
                                                if extracted.get('keyword_source') == 'publisher':
                                                    metadata[key] = extracted[key]
                                                    metadata['keyword_source'] = 'publisher'
                                            elif not metadata.get(key):
                                                metadata[key] = extracted[key]
                                    
                                    # Additional metadata fields - always update if found
                                    additional_fields = [
                                        'book_doi', 'electronic_isbn', 'print_isbn',
                                        'publication_date', 'online_date', 'author_affiliations',
                                        'citation_text', 'series_issn', 'chapter_number',
                                        'editor_info', 'copyright', 'license', 'funding',
                                        'related_content', 'download_stats', 'subject_classifications',
                                        'book_series_info'
                                    ]
                                    
                                    for field in additional_fields:
                                        if field in extracted:
                                            metadata[field] = extracted[field]
                                    
                                    # Update keyword source if we got publisher keywords
                                    if extracted.get('keywords') and extracted.get('keyword_source') == 'publisher':
                                        metadata['keyword_source'] = 'publisher'
                                        
                                    if self.verbose >= 2:
                                        if extracted.get('abstract'):
                                            print(f"[LLM_ENRICHER] Extracted abstract: {len(extracted['abstract'])} chars")
                                        if extracted.get('keywords'):
                                            print(f"[LLM_ENRICHER] Extracted {len(extracted['keywords'])} keywords from publisher")
                                        # Report other extracted fields
                                        other_fields = [f for f in additional_fields if f in extracted]
                                        if other_fields:
                                            print(f"[LLM_ENRICHER] Also extracted: {', '.join(other_fields)}")
                        except Exception as e:
                            if self.verbose >= 2:
                                print(f"[LLM_ENRICHER] Error fetching/extracting from {url_to_fetch}: {e}")
                    
                    # Store publisher URL
                    if url_to_fetch and not metadata.get('publisher_url'):
                        metadata['publisher_url'] = url_to_fetch
                    
                    # Continue searching if we still need abstract
                    if metadata['doi'] and metadata['abstract'] and metadata['keywords']:
                        break  # We have everything
                    elif metadata['doi'] and metadata['keywords'] and not metadata['abstract']:
                        # We have DOI and keywords but no abstract, continue to try to get abstract
                        if self.verbose >= 3:
                            print("[LLM_ENRICHER] Have DOI and keywords, continuing to search for abstract...")
                        continue
            
            # Generate keywords if not found
            if not metadata['keywords'] and (entry.get('title') or metadata.get('abstract')):
                generated_keywords = await self.generate_keywords(
                    entry.get('title', ''),
                    metadata.get('abstract', entry.get('abstract', ''))
                )
                
                if generated_keywords:
                    metadata['keywords'] = generated_keywords
                    metadata['keyword_source'] = 'generated'
            
        except Exception as e:
            if self.verbose >= 1:
                print(f"[LLM_ENRICHER] Error in search and extract: {e}")
        
        return metadata
    
    def clear_cache(self) -> None:
        """Clear the enricher cache."""
        self.cache = {}
        self._save_cache()
        if self.verbose >= 2:
            print("[LLM_ENRICHER] Cache cleared")
    
    def _convert_doi_to_publisher_url(self, doi: str) -> Optional[str]:
        """Convert DOI to publisher-specific URL."""
        if doi.startswith('10.1016/'):
            # Elsevier/ScienceDirect
            return f"https://www.sciencedirect.com/science/article/pii/{doi.split('/')[-1]}"
        elif doi.startswith('10.1007/'):
            # Springer
            return f"https://link.springer.com/chapter/{doi}"
        elif doi.startswith('10.1002/'):
            # Wiley
            return f"https://onlinelibrary.wiley.com/doi/{doi}"
        elif doi.startswith('10.1109/'):
            # IEEE
            return f"https://ieeexplore.ieee.org/document/{doi.split('/')[-1]}"
        elif doi.startswith('10.1145/'):
            # ACM
            return f"https://dl.acm.org/doi/{doi}"
        
        # Default to DOI.org resolver
        return f"https://doi.org/{doi}"
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        stats = {
            'total_entries': len(self.cache),
            'search_queries': 0,
            'html_extractions': 0,
            'keyword_generations': 0
        }
        
        for key in self.cache:
            if key.startswith(hashlib.md5(b'search_queries:').hexdigest()[:8]):
                stats['search_queries'] += 1
            elif key.startswith(hashlib.md5(b'extract_html:').hexdigest()[:8]):
                stats['html_extractions'] += 1
            elif key.startswith(hashlib.md5(b'keywords:').hexdigest()[:8]):
                stats['keyword_generations'] += 1
        
        return stats