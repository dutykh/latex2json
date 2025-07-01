# -*- coding: utf-8 -*-
"""
Alternative search engines for finding academic publications.

This module provides various search methods beyond Google Scholar
for finding publisher URLs and metadata.

Author: Dr. Denys Dutykh (Khalifa University of Science and Technology, Abu Dhabi, UAE)
Date: 2025-01-01
"""

import asyncio
import aiohttp
import re
from typing import List, Dict, Any, Optional
from urllib.parse import quote_plus, urlencode
import json


class AcademicSearchEngines:
    """Collection of search methods for academic publications."""
    
    def __init__(self, session: aiohttp.ClientSession, verbose: int = 2):
        """Initialize search engines."""
        self.session = session
        self.verbose = verbose
        
    async def search_crossref(self, entry: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Search CrossRef for publication metadata.
        
        CrossRef is very reliable and has good API limits.
        """
        try:
            title = entry.get('title', '')
            if not title:
                return []
            
            # Build query
            params = {
                'query.bibliographic': title,
                'rows': 5,
                'select': 'DOI,title,author,published-print,container-title,link,abstract'
            }
            
            # Add author if available
            if entry.get('authors'):
                first_author = entry['authors'][0].get('lastName', '')
                if first_author:
                    params['query.author'] = first_author
            
            url = f"https://api.crossref.org/works?{urlencode(params)}"
            
            if self.verbose >= 3:
                print(f"[SEARCH] CrossRef query: {title[:50]}...")
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    items = data.get('message', {}).get('items', [])
                    
                    results = []
                    for item in items:
                        # Extract authors
                        authors = []
                        for author in item.get('author', []):
                            name = f"{author.get('given', '')} {author.get('family', '')}".strip()
                            if name:
                                authors.append(name)
                        
                        # Extract URL - prefer publisher link
                        url = ''
                        links = item.get('link', [])
                        for link in links:
                            if link.get('content-type') == 'text/html':
                                url = link.get('URL', '')
                                break
                        
                        result = {
                            'title': item.get('title', [''])[0] if item.get('title') else '',
                            'authors': authors,
                            'year': str(item.get('published-print', {}).get('date-parts', [[None]])[0][0] or ''),
                            'doi': item.get('DOI', ''),
                            'url': url,
                            'venue': item.get('container-title', [''])[0] if item.get('container-title') else '',
                            'abstract': item.get('abstract', ''),
                            'source': 'crossref'
                        }
                        results.append(result)
                    
                    if self.verbose >= 3:
                        print(f"[SEARCH] CrossRef found {len(results)} results")
                    
                    return results
                    
        except Exception as e:
            if self.verbose >= 2:
                print(f"[SEARCH] CrossRef error: {e}")
        
        return []
    
    async def search_semantic_scholar(self, entry: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Search Semantic Scholar API.
        
        Free, no API key required, good for CS and AI papers.
        """
        try:
            title = entry.get('title', '')
            if not title:
                return []
            
            # Build search query
            query = quote_plus(title)
            url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&limit=5&fields=title,authors,year,abstract,url,externalIds,venue"
            
            if self.verbose >= 3:
                print(f"[SEARCH] Semantic Scholar query: {title[:50]}...")
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    papers = data.get('data', [])
                    
                    results = []
                    for paper in papers:
                        # Extract authors
                        authors = [author['name'] for author in paper.get('authors', [])]
                        
                        # Extract DOI from external IDs
                        doi = ''
                        external_ids = paper.get('externalIds', {})
                        if 'DOI' in external_ids:
                            doi = external_ids['DOI']
                        
                        result = {
                            'title': paper.get('title', ''),
                            'authors': authors,
                            'year': str(paper.get('year', '')),
                            'doi': doi,
                            'url': paper.get('url', ''),
                            'venue': paper.get('venue', ''),
                            'abstract': paper.get('abstract', ''),
                            'source': 'semantic_scholar'
                        }
                        results.append(result)
                    
                    if self.verbose >= 3:
                        print(f"[SEARCH] Semantic Scholar found {len(results)} results")
                    
                    return results
                    
        except Exception as e:
            if self.verbose >= 2:
                print(f"[SEARCH] Semantic Scholar error: {e}")
        
        return []
    
    async def search_unpaywall(self, doi: str) -> Optional[Dict[str, Any]]:
        """
        Search Unpaywall for open access version and metadata.
        
        Great for finding free versions and publisher links.
        """
        if not doi:
            return None
            
        try:
            # Unpaywall requires email in query
            email = "metadata@example.com"  # Generic email for metadata extraction
            url = f"https://api.unpaywall.org/v2/{doi}?email={email}"
            
            if self.verbose >= 3:
                print(f"[SEARCH] Unpaywall query for DOI: {doi}")
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Get best OA location or publisher URL
                    best_location = data.get('best_oa_location', {})
                    publisher_url = best_location.get('url_for_landing_page', '') or \
                                   best_location.get('url', '') or \
                                   data.get('doi_url', '')
                    
                    result = {
                        'title': data.get('title', ''),
                        'doi': data.get('doi', ''),
                        'url': publisher_url,
                        'year': str(data.get('year', '')),
                        'publisher': data.get('publisher', ''),
                        'is_oa': data.get('is_oa', False),
                        'source': 'unpaywall'
                    }
                    
                    if self.verbose >= 3:
                        print(f"[SEARCH] Unpaywall found: {result['url']}")
                    
                    return result
                    
        except Exception as e:
            if self.verbose >= 3:
                print(f"[SEARCH] Unpaywall error: {e}")
        
        return None
    
    async def search_microsoft_academic(self, entry: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Search using Bing Web Search for academic content.
        
        More reliable than Google Scholar for programmatic access.
        """
        try:
            # Build academic search query
            title = entry.get('title', '')
            authors = entry.get('authors', [])
            year = entry.get('year', '')
            
            # Construct search query
            query_parts = [f'"{title}"']
            if authors:
                first_author = authors[0].get('lastName', '')
                if first_author:
                    query_parts.append(first_author)
            if year:
                query_parts.append(str(year))
            
            # Add academic sites
            query_parts.append('(site:springer.com OR site:sciencedirect.com OR site:ieee.org OR site:acm.org OR site:wiley.com)')
            
            query = ' '.join(query_parts)
            
            # Use DuckDuckGo HTML search (no API key needed)
            url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
            
            if self.verbose >= 3:
                print(f"[SEARCH] Web search query: {query[:100]}...")
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    html = await response.text()
                    
                    # Extract results using regex (simple HTML parsing)
                    results = []
                    
                    # Find result links
                    link_pattern = r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>([^<]+)</a>'
                    matches = re.finditer(link_pattern, html, re.IGNORECASE)
                    
                    for i, match in enumerate(matches):
                        if i >= 5:  # Limit to 5 results
                            break
                        
                        result_url = match.group(1)
                        result_title = match.group(2)
                        
                        # Filter for academic publishers
                        if any(domain in result_url for domain in ['springer.com', 'sciencedirect.com', 'ieee.org', 'acm.org', 'wiley.com']):
                            result = {
                                'title': result_title,
                                'url': result_url,
                                'source': 'web_search'
                            }
                            results.append(result)
                    
                    if self.verbose >= 3:
                        print(f"[SEARCH] Web search found {len(results)} academic results")
                    
                    return results
                    
        except Exception as e:
            if self.verbose >= 2:
                print(f"[SEARCH] Web search error: {e}")
        
        return []
    
    async def direct_publisher_search(self, entry: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Search directly on publisher websites based on publisher name.
        """
        publisher = entry.get('publisher', {}).get('name', '').lower()
        title = entry.get('title', '')
        
        if not publisher or not title:
            return []
        
        results = []
        
        try:
            # Springer
            if 'springer' in publisher:
                search_url = f"https://link.springer.com/search?query={quote_plus(title)}"
                results.append({
                    'title': title,
                    'url': search_url,
                    'source': 'springer_direct',
                    'search_url': True
                })
            
            # Elsevier
            elif 'elsevier' in publisher:
                search_url = f"https://www.sciencedirect.com/search?qs={quote_plus(title)}"
                results.append({
                    'title': title,
                    'url': search_url,
                    'source': 'elsevier_direct',
                    'search_url': True
                })
            
            # Wiley
            elif 'wiley' in publisher:
                search_url = f"https://onlinelibrary.wiley.com/action/doSearch?AllField={quote_plus(title)}"
                results.append({
                    'title': title,
                    'url': search_url,
                    'source': 'wiley_direct',
                    'search_url': True
                })
            
            if results and self.verbose >= 3:
                print(f"[SEARCH] Direct publisher search URL generated")
            
        except Exception as e:
            if self.verbose >= 2:
                print(f"[SEARCH] Direct publisher search error: {e}")
        
        return results
    
    async def search_all(self, entry: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run all search methods in parallel and return combined results.
        """
        # Create tasks for all search methods
        tasks = [
            ('crossref', self.search_crossref(entry)),
            ('semantic_scholar', self.search_semantic_scholar(entry)),
            ('web_search', self.search_microsoft_academic(entry)),
            ('direct_publisher', self.direct_publisher_search(entry))
        ]
        
        # Run all searches in parallel
        results = {}
        search_results = await asyncio.gather(
            *[task[1] for task in tasks],
            return_exceptions=True
        )
        
        # Collect results
        for (name, _), result in zip(tasks, search_results):
            if isinstance(result, Exception):
                if self.verbose >= 2:
                    print(f"[SEARCH] {name} failed: {result}")
                results[name] = []
            else:
                results[name] = result
        
        # If we found a DOI from any source, also try Unpaywall
        for source_results in results.values():
            for result in source_results:
                if result.get('doi'):
                    unpaywall_result = await self.search_unpaywall(result['doi'])
                    if unpaywall_result:
                        results['unpaywall'] = [unpaywall_result]
                    break
            if 'unpaywall' in results:
                break
        
        return results