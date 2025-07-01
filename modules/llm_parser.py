# -*- coding: utf-8 -*-
"""
LLM-based LaTeX bibliography parser using Claude API.

This module provides intelligent parsing of LaTeX bibliography entries using
Claude's language understanding capabilities, with fallback to regex parsing.

Author: Dr. Denys Dutykh (Khalifa University of Science and Technology, Abu Dhabi, UAE)
Date: 2025-01-01
"""

import json
import hashlib
import re
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


class LLMBibliographyParser:
    """Parser for LaTeX bibliography using Claude LLM."""
    
    # Claude model to use
    MODEL = "claude-3-haiku-20240307"  # Fast and cost-effective for parsing tasks
    
    # System prompt for bibliography parsing
    SYSTEM_PROMPT = """You are an expert at parsing LaTeX bibliography entries. Extract structured information from LaTeX entries and return it as JSON.

For each entry, extract:
- authors: Array of {firstName, lastName} objects
- title: The chapter/article title (without LaTeX formatting)
- editors: Array of {firstName, lastName} objects (if present)
- book_title: The book title (if it's a book chapter)
- series: The series name (e.g., "Lecture Notes in Physics", "Springer Water")
- volume: The volume number (integer)
- pages: Object with {start, end} page numbers
- publisher: Object with {name, location}
- year: Publication year (integer)
- url: Any URL in the entry
- doi: DOI if present
- chapter_number: Chapter number if specified
- abstract: Empty string (to be filled later)
- keywords: Empty array (to be filled later)

Important parsing rules:
1. For names like "D.~Dutykh" or "Ph.~\\textsc{Goubersville}", extract firstName and lastName
2. Remove LaTeX formatting like \\textbf{}, \\textit{}, \\textsc{}
3. Handle various separators: "In:", "in", or just comma after title
4. Series can appear as "Series Name (volume)" or "Series Name, Vol. X"
5. Pages can be "pp. 47--64" or just "47--64"
6. Publishers can be "Springer, Singapore" or "Springer Singapore"
7. Convert LaTeX special characters (---, --, ~) appropriately

Return ONLY valid JSON, no explanations."""
    
    def __init__(self, config, cache_file: Path = None, verbose: int = 2):
        """
        Initialize the LLM parser.
        
        Args:
            config: Configuration manager instance
            cache_file: Path to cache file (optional)
            verbose: Verbosity level (0-3)
        """
        self.config = config
        self.verbose = verbose
        self.cache_file = cache_file or Path(".llm_parse_cache.json")
        self.cache = self._load_cache()
        
        # Initialize Claude client if available
        self.client = None
        self.enabled = False
        
        if ANTHROPIC_AVAILABLE and config:
            api_key = config.get_api_key('anthropic')
            if api_key:
                self.client = Anthropic(api_key=api_key)
                self.enabled = config.get('preferences.use_llm_parser', True)
                
                if self.verbose >= 2:
                    print("[LLM_PARSER] Claude API initialized for parsing")
            elif self.verbose >= 1:
                print("[LLM_PARSER] No Anthropic API key found")
        elif not ANTHROPIC_AVAILABLE and self.verbose >= 1:
            print("[LLM_PARSER] Anthropic library not installed")
    
    def _load_cache(self) -> Dict[str, Any]:
        """Load cache from file."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
                    if self.verbose >= 3:
                        print(f"[LLM_PARSER] Loaded cache with {len(cache)} entries")
                    return cache
            except Exception as e:
                if self.verbose >= 1:
                    print(f"[LLM_PARSER] Error loading cache: {e}")
        return {}
    
    def _save_cache(self) -> None:
        """Save cache to file."""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
                if self.verbose >= 3:
                    print(f"[LLM_PARSER] Saved cache with {len(self.cache)} entries")
        except Exception as e:
            if self.verbose >= 1:
                print(f"[LLM_PARSER] Error saving cache: {e}")
    
    def _get_cache_key(self, content: str) -> str:
        """Generate cache key from content."""
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def _is_cache_valid(self, cache_entry: Dict) -> bool:
        """Check if cache entry is still valid."""
        if 'timestamp' not in cache_entry:
            return False
        
        cache_days = self.config.get('preferences.llm_parser_cache_days', 30) if self.config else 30
        cache_time = datetime.fromisoformat(cache_entry['timestamp'])
        expiry_time = cache_time + timedelta(days=cache_days)
        
        return datetime.now() < expiry_time
    
    def parse_entry(self, latex_content: str) -> Optional[Dict[str, Any]]:
        """
        Parse a single LaTeX bibliography entry using LLM.
        
        Args:
            latex_content: The LaTeX entry content
            
        Returns:
            Parsed entry dictionary or None if parsing fails
        """
        if not self.enabled or not self.client:
            return None
        
        # Check cache first
        cache_key = self._get_cache_key(latex_content)
        if cache_key in self.cache:
            cache_entry = self.cache[cache_key]
            if self._is_cache_valid(cache_entry):
                if self.verbose >= 3:
                    print("[LLM_PARSER] Using cached result")
                elif self.verbose >= 2:
                    # Show cached result info
                    cached_result = cache_entry['result']
                    title = cached_result.get('title', 'Unknown title')[:60]
                    if len(cached_result.get('title', '')) > 60:
                        title += "..."
                    authors = cached_result.get('authors', [])
                    if authors:
                        first_author = f"{authors[0].get('firstName', '')} {authors[0].get('lastName', '')}".strip()
                        author_info = f"{first_author}" + (f" et al." if len(authors) > 1 else "")
                    else:
                        author_info = "Unknown authors"
                    
                    print(f"[LLM_PARSER] Using cached: {author_info} - {title}")
                return cache_entry['result']
        
        # Prepare the prompt
        prompt = f"Parse this LaTeX bibliography entry and return JSON:\n\n{latex_content}"
        
        try:
            if self.verbose >= 3:
                print(f"[LLM_PARSER] Sending entry to Claude for parsing")
            
            # Call Claude API
            response = self.client.messages.create(
                model=self.MODEL,
                max_tokens=1000,
                temperature=0,
                system=self.SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract JSON from response
            json_text = response.content[0].text.strip()
            
            # Clean up the response if needed
            if json_text.startswith("```json"):
                json_text = json_text[7:]
            if json_text.endswith("```"):
                json_text = json_text[:-3]
            json_text = json_text.strip()
            
            # Parse JSON
            result = json.loads(json_text)
            
            # Cache the result
            self.cache[cache_key] = {
                'result': result,
                'timestamp': datetime.now().isoformat()
            }
            self._save_cache()
            
            if self.verbose >= 2:
                # Extract basic info for display
                title = result.get('title', 'Unknown title')[:60]
                if len(result.get('title', '')) > 60:
                    title += "..."
                authors = result.get('authors', [])
                if authors:
                    first_author = f"{authors[0].get('firstName', '')} {authors[0].get('lastName', '')}".strip()
                    author_info = f"{first_author}" + (f" et al." if len(authors) > 1 else "")
                else:
                    author_info = "Unknown authors"
                
                print(f"[LLM_PARSER] Successfully parsed: {author_info} - {title}")
            
            return result
            
        except json.JSONDecodeError as e:
            if self.verbose >= 1:
                print(f"[LLM_PARSER] Failed to parse Claude response as JSON: {e}")
                if self.verbose >= 3:
                    print(f"[LLM_PARSER] Response was: {json_text}")
            return None
        except Exception as e:
            if self.verbose >= 1:
                print(f"[LLM_PARSER] Error calling Claude API: {e}")
            return None
    
    def parse_entries(self, entries_content: List[str]) -> List[Dict[str, Any]]:
        """
        Parse multiple LaTeX entries.
        
        Args:
            entries_content: List of LaTeX entry strings
            
        Returns:
            List of parsed entries
        """
        results = []
        
        for i, content in enumerate(entries_content):
            if self.verbose >= 2:
                print(f"[LLM_PARSER] Parsing entry {i+1}/{len(entries_content)}")
            
            result = self.parse_entry(content)
            if result:
                results.append(result)
            else:
                if self.verbose >= 1:
                    print(f"[LLM_PARSER] Failed to parse entry {i+1}")
        
        return results
    
    def clear_cache(self) -> None:
        """Clear the parse cache."""
        self.cache = {}
        self._save_cache()
        if self.verbose >= 2:
            print("[LLM_PARSER] Cache cleared")