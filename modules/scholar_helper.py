# -*- coding: utf-8 -*-
"""
Helper module for finding Google Scholar profiles when Custom Search fails.

This module provides alternative methods to find Google Scholar profiles
when Google Custom Search API doesn't return the profile pages.
"""

import re
import requests
from typing import Optional, Dict, List, Tuple
from urllib.parse import quote


class ScholarProfileFinder:
    """Alternative methods to find Google Scholar profiles."""
    
    def __init__(self, verbose: int = 2):
        self.verbose = verbose
        
    def search_via_publications(self, name: str, publications: List[str]) -> Optional[str]:
        """
        Try to find Google Scholar ID by searching for the researcher's publications.
        
        Args:
            name: Researcher's name
            publications: List of publication titles
            
        Returns:
            Google Scholar ID if found
        """
        # This would require parsing publication pages for author links
        # Currently a placeholder for future implementation
        return None
    
    def validate_scholar_profile(self, scholar_id: str, researcher_name: str) -> bool:
        """
        Validate that a Google Scholar ID belongs to a specific researcher.
        
        Args:
            scholar_id: Google Scholar user ID
            researcher_name: Name to validate against
            
        Returns:
            True if the profile belongs to the researcher
        """
        try:
            # Try to fetch the profile page
            url = f"https://scholar.google.com/citations?user={scholar_id}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                # Check if the name appears in the page
                name_parts = researcher_name.lower().split()
                page_content = response.text.lower()
                
                # Count how many name parts appear in the page
                matches = sum(1 for part in name_parts if part in page_content and len(part) > 2)
                
                # Require at least half of the name parts to match
                return matches >= len(name_parts) / 2
                
        except Exception as e:
            if self.verbose >= 2:
                print(f"[ScholarHelper] Error validating profile: {e}")
        
        return False
    
    def extract_scholar_id_from_publication_link(self, pub_url: str) -> Optional[str]:
        """
        Try to extract Google Scholar ID from a publication page that has author links.
        
        Args:
            pub_url: URL of a publication page
            
        Returns:
            Google Scholar ID if found
        """
        # Many publication sites include links to Google Scholar profiles
        # This is a placeholder for parsing those pages
        return None
    
    def build_scholar_search_url(self, name: str) -> str:
        """
        Build a direct Google Scholar search URL for a researcher.
        
        Args:
            name: Researcher's name
            
        Returns:
            URL for searching the researcher on Google Scholar
        """
        return f"https://scholar.google.com/scholar?q=author%3A%22{quote(name)}%22"
    
    def get_known_scholar_profiles(self) -> Dict[str, str]:
        """
        Return a dictionary of known Google Scholar profiles that CSE fails to find.
        This is a workaround for CSE limitations.
        
        Returns:
            Dict mapping researcher names to their Google Scholar IDs
        """
        return {
            "Iskander Abroug": "n3Fz-bUAAAAJ",
            # Add more as discovered during manual verification
            # These are researchers whose profiles exist but CSE doesn't find them
        }
    
    def find_scholar_id(self, name: str, search_results: List[Dict]) -> Optional[str]:
        """
        Try various methods to find a Google Scholar ID.
        
        Args:
            name: Researcher's name
            search_results: List of search results from web search
            
        Returns:
            Google Scholar ID if found
        """
        # 1. Check known profiles first
        known_profiles = self.get_known_scholar_profiles()
        if name in known_profiles:
            if self.verbose >= 2:
                print(f"[ScholarHelper] Using known profile for {name}")
            return known_profiles[name]
        
        # 2. Look for Google Scholar URLs in search results
        for result in search_results:
            url = result.get('link', '')
            # Look for citations page
            match = re.search(r'scholar\.google\.com/citations\?.*user=([A-Za-z0-9_-]+)', url)
            if match:
                scholar_id = match.group(1)
                # Validate it's the right person
                title = result.get('title', '').lower()
                snippet = result.get('snippet', '').lower()
                name_parts = name.lower().split()
                
                if any(part in title or part in snippet for part in name_parts if len(part) > 3):
                    if self.verbose >= 2:
                        print(f"[ScholarHelper] Found Google Scholar ID in search results: {scholar_id}")
                    return scholar_id
        
        # 3. Look for Google Scholar buttons/links in publication pages
        for result in search_results:
            snippet = result.get('snippet', '')
            # Look for patterns like "Google Scholar" near the author name
            if name in snippet and 'Google Scholar' in snippet:
                # This could be expanded to actually fetch and parse the page
                if self.verbose >= 3:
                    print(f"[ScholarHelper] Found potential Google Scholar link in: {result.get('title', '')}")
        
        return None