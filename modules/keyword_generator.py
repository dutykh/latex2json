# -*- coding: utf-8 -*-
"""
Keyword generator module using Claude API.

This module generates academic keywords for book chapters when they are not
available from publisher metadata.

Author: Dr. Denys Dutykh (Khalifa University of Science and Technology, Abu Dhabi, UAE)
Date: 2025-07-01
"""

from typing import List, Dict, Any, Optional
from anthropic import AsyncAnthropic

from .cache_manager import CacheManager
from .config_manager import ConfigManager


class KeywordGenerator:
    """Generates keywords for academic content using Claude API."""
    
    def __init__(self, config: ConfigManager, cache: CacheManager, verbose: int = 2):
        """
        Initialize the keyword generator.
        
        Args:
            config: Configuration manager instance
            cache: Cache manager instance
            verbose: Verbosity level (0-3)
        """
        self.config = config
        self.cache = cache
        self.verbose = verbose
        
        # Initialize Claude client if API key is available
        api_key = config.get_api_key('anthropic')
        self.client = AsyncAnthropic(api_key=api_key) if api_key else None
        self.enabled = bool(api_key) and config.get('preferences.enable_keyword_generation', True)
        
        if self.verbose >= 2:
            if self.enabled:
                print("[KEYWORDS] Claude API initialized for keyword generation")
            else:
                print("[KEYWORDS] Keyword generation disabled or API key not found")
    
    async def generate_keywords_for_entries(self, entries: List[Dict[str, Any]]) -> None:
        """
        Generate keywords for entries that don't have them.
        
        Args:
            entries: List of bibliography entries (modified in place)
        """
        if not self.enabled:
            return
        
        # Only generate keywords for entries that:
        # 1. Don't have keywords from publishers
        # 2. Have an abstract (required for accurate keyword generation)
        entries_needing_keywords = [
            entry for entry in entries 
            if (not entry.get('keywords') or entry.get('keyword_source') == 'none') 
            and entry.get('keyword_source') != 'publisher'
            and entry.get('abstract')  # Must have abstract for keyword generation
        ]
        
        # Count entries skipped due to missing abstract
        entries_without_abstract = [
            entry for entry in entries 
            if (not entry.get('keywords') or entry.get('keyword_source') == 'none') 
            and entry.get('keyword_source') != 'publisher'
            and not entry.get('abstract')
        ]
        
        if not entries_needing_keywords:
            if self.verbose >= 2:
                if entries_without_abstract:
                    print(f"[KEYWORDS] Skipped {len(entries_without_abstract)} entries without abstracts")
                else:
                    print("[KEYWORDS] All entries already have keywords")
            return
        
        if self.verbose >= 1:
            print(f"\n[KEYWORDS] Generating keywords for {len(entries_needing_keywords)} entries")
        
        for i, entry in enumerate(entries_needing_keywords, 1):
            if self.verbose >= 2:
                print(f"\n[KEYWORDS] Processing entry {i}/{len(entries_needing_keywords)}")
                print(f"[KEYWORDS] Title: {entry.get('title', 'Unknown')[:60]}...")
                
                # Show what metadata we have to work with
                if self.verbose >= 3:
                    print("[KEYWORDS] Available context:")
                    if entry.get('book_title'):
                        print(f"  - Book: {entry['book_title'][:50]}...")
                    if entry.get('abstract'):
                        print(f"  - Abstract: {len(entry['abstract'])} chars available")
                    if entry.get('series'):
                        print(f"  - Series: {entry['series']}")
                    if not entry.get('abstract'):
                        print("  - No abstract available (using title and book info only)")
            
            keywords = await self._generate_keywords_for_entry(entry)
            
            if keywords:
                entry['keywords'] = keywords
                entry['keyword_source'] = 'generated'
                
                # Cache the keywords
                cache_key = self.cache.create_entry_key(entry)
                self.cache.set_keywords(cache_key, keywords, source='generated')
                
                if self.verbose >= 2:
                    print(f"[SUCCESS] Generated {len(keywords)} keywords:")
                    for kw in keywords:
                        print(f"  • {kw}")
            else:
                if self.verbose >= 2:
                    print("[WARNING] Failed to generate keywords")
    
    async def _generate_keywords_for_entry(self, entry: Dict[str, Any]) -> Optional[List[str]]:
        """Generate keywords for a single entry."""
        # Check cache first
        cache_key = self.cache.create_entry_key(entry)
        cached_keywords = self.cache.get_keywords(cache_key)
        
        if cached_keywords:
            if self.verbose >= 3:
                print("[KEYWORDS] Using cached keywords")
            return cached_keywords
        
        # Build context for keyword generation
        context = self._build_context(entry)
        
        if not context:
            if self.verbose >= 3:
                print("[KEYWORDS] Insufficient context for keyword generation")
            return None
        
        # Generate keywords using Claude
        try:
            keywords = await self._call_claude_api(context)
            return keywords
        except Exception as e:
            if self.verbose >= 2:
                print(f"[ERROR] Claude API error: {e}")
            return None
    
    def _build_context(self, entry: Dict[str, Any]) -> Optional[str]:
        """Build context string for keyword generation."""
        parts = []
        
        # Title is required
        title = entry.get('title', '').strip()
        if not title:
            return None
        
        parts.append(f"Title: {title}")
        
        # Add book title if available
        book_title = entry.get('book_title', '').strip()
        if book_title:
            parts.append(f"Book: {book_title}")
        
        # Add abstract if available
        abstract = entry.get('abstract', '').strip()
        if abstract:
            # Limit abstract length to avoid token limits
            if len(abstract) > 500:
                abstract = abstract[:500] + "..."
            parts.append(f"Abstract: {abstract}")
        
        # Add series information if available
        series = entry.get('series', '').strip()
        if series:
            parts.append(f"Series: {series}")
        
        return '\n\n'.join(parts)
    
    async def _call_claude_api(self, context: str) -> List[str]:
        """Call Claude API to generate keywords."""
        if not self.client:
            return []
        
        prompt = f"""You are an academic librarian helping to catalog book chapters. 
Based on the following information about a book chapter, generate 5-8 relevant academic keywords 
that would help researchers find this content. Keywords should be specific, technical terms 
that capture the main topics and methods discussed.

{context}

Instructions:
- Focus on technical terms, methodologies, and specific concepts mentioned
- If an abstract is provided, prioritize keywords from its content
- Avoid generic terms; be as specific as possible
- Include both theoretical concepts and practical applications mentioned

Provide only the keywords as a comma-separated list, without any explanation or numbering.
Example format: keyword1, keyword2, keyword3, keyword4, keyword5"""

        try:
            # Use Claude 3 Haiku for cost-effectiveness
            response = await self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=150,
                temperature=0.3,  # Lower temperature for more focused keywords
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Parse the response
            keywords_text = response.content[0].text.strip()
            
            # Split and clean keywords
            keywords = [
                kw.strip() 
                for kw in keywords_text.split(',') 
                if kw.strip()
            ]
            
            # Validate keywords (remove too short or too long)
            keywords = [
                kw for kw in keywords 
                if 2 <= len(kw) <= 50 and not kw.isdigit()
            ]
            
            # Limit to 8 keywords
            return keywords[:8]
            
        except Exception as e:
            if self.verbose >= 3:
                print(f"[KEYWORDS] API call failed: {e}")
            raise
    
    def validate_keywords(self, keywords: List[str]) -> List[str]:
        """
        Validate and clean generated keywords.
        
        Args:
            keywords: List of keywords to validate
            
        Returns:
            Cleaned list of keywords
        """
        validated = []
        
        for keyword in keywords:
            # Clean whitespace
            keyword = keyword.strip()
            
            # Skip empty or too short
            if len(keyword) < 2:
                continue
            
            # Skip if just numbers
            if keyword.isdigit():
                continue
            
            # Skip if too long (probably a phrase)
            if len(keyword) > 50:
                continue
            
            # Remove quotes if present
            keyword = keyword.strip('"\'')
            
            # Add to validated list
            validated.append(keyword)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for kw in validated:
            kw_lower = kw.lower()
            if kw_lower not in seen:
                seen.add(kw_lower)
                unique_keywords.append(kw)
        
        return unique_keywords