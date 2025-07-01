# -*- coding: utf-8 -*-
"""
Cache manager module for storing API responses and generated keywords.

This module provides caching functionality with expiry management to minimize
redundant API calls and store generated content.

Author: Dr. Denys Dutykh (Khalifa University of Science and Technology, Abu Dhabi, UAE)
Date: 2025-07-01
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta


class CacheManager:
    """Manages caching for API responses and generated keywords."""
    
    def __init__(self, api_cache_file: Path, keyword_cache_file: Path, 
                 expiry_days: int = 7, verbose: int = 2):
        """
        Initialize the cache manager.
        
        Args:
            api_cache_file: Path to API response cache file
            keyword_cache_file: Path to keyword cache file
            expiry_days: Number of days before cache entries expire
            verbose: Verbosity level (0-3)
        """
        self.api_cache_file = api_cache_file
        self.keyword_cache_file = keyword_cache_file
        self.expiry_days = expiry_days
        self.verbose = verbose
        
        # Load existing caches
        self.api_cache = self._load_cache(api_cache_file)
        self.keyword_cache = self._load_cache(keyword_cache_file)
    
    def _load_cache(self, cache_file: Path) -> Dict[str, Any]:
        """Load cache from file or create empty cache."""
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache = json.load(f)
                    if self.verbose >= 2:
                        print(f"[CACHE] Loaded {len(cache)} entries from {cache_file.name}")
                    return cache
            except json.JSONDecodeError:
                if self.verbose >= 1:
                    print(f"[CACHE] Warning: Cache file {cache_file.name} is corrupted. Starting fresh.")
        return {}
    
    def save_caches(self) -> None:
        """Save both caches to their respective files."""
        self._save_cache(self.api_cache, self.api_cache_file)
        self._save_cache(self.keyword_cache, self.keyword_cache_file)
    
    def _save_cache(self, cache: Dict[str, Any], cache_file: Path) -> None:
        """Save a cache to file."""
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
            if self.verbose >= 3:
                print(f"[CACHE] Saved {len(cache)} entries to {cache_file.name}")
        except Exception as e:
            if self.verbose >= 1:
                print(f"[CACHE] Error saving cache to {cache_file.name}: {e}")
    
    def get_api_response(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get cached API response if not expired.
        
        Args:
            key: Cache key (usually URL or identifier)
            
        Returns:
            Cached response or None if not found/expired
        """
        if key in self.api_cache:
            entry = self.api_cache[key]
            if self._is_valid_entry(entry):
                if self.verbose >= 3:
                    print(f"[CACHE] API cache hit for: {key}")
                return entry['data']
            else:
                # Remove expired entry
                del self.api_cache[key]
                if self.verbose >= 3:
                    print(f"[CACHE] Removed expired API cache entry: {key}")
        return None
    
    def set_api_response(self, key: str, data: Dict[str, Any]) -> None:
        """
        Cache an API response.
        
        Args:
            key: Cache key
            data: Response data to cache
        """
        self.api_cache[key] = {
            'data': data,
            'timestamp': datetime.now().isoformat(),
            'expiry': (datetime.now() + timedelta(days=self.expiry_days)).isoformat()
        }
        if self.verbose >= 3:
            print(f"[CACHE] Cached API response for: {key}")
    
    def get_keywords(self, key: str) -> Optional[List[str]]:
        """
        Get cached keywords (no expiry for keywords).
        
        Args:
            key: Cache key (usually title or identifier)
            
        Returns:
            Cached keywords or None if not found
        """
        if key in self.keyword_cache:
            if self.verbose >= 3:
                print(f"[CACHE] Keyword cache hit for: {key}")
            return self.keyword_cache[key]['keywords']
        return None
    
    def set_keywords(self, key: str, keywords: List[str], source: str = "generated") -> None:
        """
        Cache keywords.
        
        Args:
            key: Cache key
            keywords: List of keywords
            source: Source of keywords (e.g., "generated", "publisher")
        """
        self.keyword_cache[key] = {
            'keywords': keywords,
            'source': source,
            'timestamp': datetime.now().isoformat()
        }
        if self.verbose >= 3:
            print(f"[CACHE] Cached {len(keywords)} keywords for: {key}")
    
    def _is_valid_entry(self, entry: Dict[str, Any]) -> bool:
        """Check if a cache entry is still valid (not expired)."""
        if 'expiry' not in entry:
            return True  # No expiry set
        
        expiry_date = datetime.fromisoformat(entry['expiry'])
        return datetime.now() < expiry_date
    
    def clear_expired(self) -> Tuple[int, int]:
        """
        Remove all expired entries from API cache.
        
        Returns:
            Tuple of (removed_count, remaining_count)
        """
        expired_keys = [
            key for key, entry in self.api_cache.items()
            if not self._is_valid_entry(entry)
        ]
        
        for key in expired_keys:
            del self.api_cache[key]
        
        removed_count = len(expired_keys)
        remaining_count = len(self.api_cache)
        
        if self.verbose >= 2 and removed_count > 0:
            print(f"[CACHE] Removed {removed_count} expired entries, {remaining_count} remaining")
        
        return removed_count, remaining_count
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the caches."""
        api_expired = sum(1 for entry in self.api_cache.values() if not self._is_valid_entry(entry))
        
        return {
            'api_cache': {
                'total_entries': len(self.api_cache),
                'expired_entries': api_expired,
                'valid_entries': len(self.api_cache) - api_expired
            },
            'keyword_cache': {
                'total_entries': len(self.keyword_cache),
                'generated': sum(1 for e in self.keyword_cache.values() if e.get('source') == 'generated'),
                'publisher': sum(1 for e in self.keyword_cache.values() if e.get('source') == 'publisher')
            }
        }
    
    def create_entry_key(self, entry: Dict[str, Any]) -> str:
        """
        Create a unique cache key for a bibliography entry.
        
        Args:
            entry: Bibliography entry dictionary
            
        Returns:
            Unique key string
        """
        # Use combination of first author, year, and first words of title
        authors = entry.get('authors', [])
        first_author = authors[0]['lastName'] if authors else 'Unknown'
        year = entry.get('year', 'NoYear')
        title_words = entry.get('title', '').split()[:3]
        title_part = '_'.join(title_words)
        
        return f"{first_author}_{year}_{title_part}".lower().replace(' ', '_')