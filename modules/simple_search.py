# -*- coding: utf-8 -*-
"""
Simple web search module using direct URL construction.

This module provides a simplified approach to finding researcher information
by constructing direct URLs to known academic platforms.

Author: Dr. Denys Dutykh (Khalifa University of Science and Technology, Abu Dhabi, UAE)
Date: 2025-01-04
"""

import re
import requests
import time
from typing import Dict, List, Optional, Any
from urllib.parse import quote_plus


class SimpleResearcherSearch:
    """Simple researcher search using direct platform URLs."""
    
    # Headers for requests
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate',
        'DNT': '1',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    }
    
    def __init__(self, verbose: int = 2):
        """Initialize the simple search client."""
        self.verbose = verbose
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
    
    def search_researcher(self, name: str, affiliation: str, 
                         country: str = None) -> Dict[str, Any]:
        """
        Search for researcher information using direct platform queries.
        
        Args:
            name: Researcher's full name
            affiliation: Institution/affiliation
            country: Country (optional)
            
        Returns:
            Dictionary with found information
        """
        results = {
            'name': name,
            'affiliation': affiliation,
            'homepage': None,
            'email': None,
            'title': None,
            'profiles': {
                'orcid': None,
                'google_scholar': None,
                'researchgate': None,
                'linkedin': None,
                'github': None,
                'twitter': None,
                'scopus': None
            },
            'academic_metrics': {
                'h_index': None,
                'citations': None,
                'i10_index': None,
                'publications_count': None
            },
            'research_interests': [],
            'recent_publications': [],
            'additional_info': {},
            'search_confidence': 0.0
        }
        
        # Search different platforms
        platforms_searched = 0
        platforms_found = 0
        
        # 1. Try ORCID
        orcid_result = self._search_orcid(name, affiliation)
        if orcid_result:
            results['profiles']['orcid'] = orcid_result.get('orcid')
            if orcid_result.get('email'):
                results['email'] = orcid_result['email']
            platforms_found += 1
        platforms_searched += 1
        
        # 2. Try Google Scholar
        scholar_result = self._search_google_scholar(name, affiliation)
        if scholar_result:
            results['profiles']['google_scholar'] = scholar_result.get('profile_id')
            if scholar_result.get('metrics'):
                results['academic_metrics'].update(scholar_result['metrics'])
            if scholar_result.get('interests'):
                results['research_interests'] = scholar_result['interests']
            platforms_found += 1
        platforms_searched += 1
        
        # 3. Try ResearchGate
        rg_result = self._search_researchgate(name, affiliation)
        if rg_result:
            results['profiles']['researchgate'] = rg_result.get('profile')
            if rg_result.get('homepage'):
                results['homepage'] = rg_result['homepage']
            platforms_found += 1
        platforms_searched += 1
        
        # 4. Try institutional search
        inst_result = self._search_institutional(name, affiliation)
        if inst_result:
            if inst_result.get('homepage') and not results['homepage']:
                results['homepage'] = inst_result['homepage']
            if inst_result.get('email') and not results['email']:
                results['email'] = inst_result['email']
            if inst_result.get('title'):
                results['title'] = inst_result['title']
            platforms_found += 0.5  # Partial credit
        platforms_searched += 1
        
        # Calculate confidence
        results['search_confidence'] = platforms_found / platforms_searched if platforms_searched > 0 else 0.0
        
        return results
    
    def _search_orcid(self, name: str, affiliation: str) -> Optional[Dict[str, Any]]:
        """Search ORCID registry."""
        try:
            # Use ORCID public API
            query = f"{name} {affiliation}"
            url = f"https://pub.orcid.org/v3.0/search/?q={quote_plus(query)}&rows=5"
            
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                # Simple extraction - would need proper XML parsing in production
                content = response.text
                orcid_match = re.search(r'<common:orcid-identifier.*?>([\d-]+)</common:orcid-identifier>', content)
                if orcid_match:
                    return {'orcid': orcid_match.group(1)}
        except Exception as e:
            if self.verbose >= 3:
                print(f"[SimpleSearch] ORCID search error: {e}")
        
        return None
    
    def _search_google_scholar(self, name: str, affiliation: str) -> Optional[Dict[str, Any]]:
        """Search Google Scholar (simplified approach)."""
        try:
            # Note: This is a simplified approach - Google Scholar requires more sophisticated handling
            # This shows the structure but won't work without proper scraping setup
            
            # Return placeholder for now
            if self.verbose >= 3:
                print("[SimpleSearch] Google Scholar search would require Selenium or API access")
            
            # Could integrate with scholarly library here
            return None
            
        except Exception as e:
            if self.verbose >= 3:
                print(f"[SimpleSearch] Scholar search error: {e}")
        
        return None
    
    def _search_researchgate(self, name: str, affiliation: str) -> Optional[Dict[str, Any]]:
        """Search ResearchGate (simplified)."""
        try:
            # ResearchGate blocks automated access
            # This shows the structure
            if self.verbose >= 3:
                print("[SimpleSearch] ResearchGate search would require authentication")
            
            return None
            
        except Exception as e:
            if self.verbose >= 3:
                print(f"[SimpleSearch] ResearchGate search error: {e}")
        
        return None
    
    def _search_institutional(self, name: str, affiliation: str) -> Optional[Dict[str, Any]]:
        """Try to find institutional pages."""
        results = {}
        
        # Extract domain hints from affiliation
        domain_hints = self._extract_domain_hints(affiliation)
        
        # Common university directory patterns
        name_parts = name.lower().split()
        if len(name_parts) >= 2:
            first_name = name_parts[0]
            last_name = name_parts[-1]
            
            # Common patterns
            patterns = [
                f"{first_name}.{last_name}",
                f"{first_name[0]}{last_name}",
                f"{last_name}",
                f"{first_name}-{last_name}",
                f"{last_name}-{first_name}"
            ]
            
            # Try common institutional URL patterns
            for domain in domain_hints:
                for pattern in patterns[:2]:  # Try first 2 patterns
                    urls = [
                        f"https://www.{domain}/people/{pattern}",
                        f"https://www.{domain}/~{pattern}",
                        f"https://www.{domain}/faculty/{pattern}",
                        f"https://www.{domain}/staff/{pattern}",
                        f"https://{domain}/people/{pattern}",
                    ]
                    
                    for url in urls[:2]:  # Limit attempts
                        try:
                            response = self.session.head(url, timeout=5, allow_redirects=True)
                            if response.status_code == 200:
                                results['homepage'] = url
                                if self.verbose >= 2:
                                    print(f"[SimpleSearch] Found institutional page: {url}")
                                return results
                        except (requests.RequestException, requests.Timeout):
                            pass
                        
                        time.sleep(0.1)  # Be polite
        
        return results if results else None
    
    def _extract_domain_hints(self, affiliation: str) -> List[str]:
        """Extract possible domains from affiliation."""
        domains = []
        
        # Common university domains
        affiliation_lower = affiliation.lower()
        
        # Try to extract existing domains
        domain_match = re.findall(r'([a-z0-9]+(?:-[a-z0-9]+)*\.(?:edu|ac\.[a-z]{2}|fr|de|it|es|org))', affiliation_lower)
        if domain_match:
            domains.extend(domain_match)
        
        # Common patterns
        if 'university' in affiliation_lower:
            # Extract university name
            uni_match = re.search(r'university\s+(?:of\s+)?([a-z]+)', affiliation_lower)
            if uni_match:
                uni_name = uni_match.group(1)
                domains.extend([
                    f"{uni_name}.edu",
                    f"{uni_name}.ac.uk",
                    f"www.{uni_name}.edu"
                ])
        
        return list(set(domains))[:3]  # Return top 3 unique domains