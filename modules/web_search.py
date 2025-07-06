# -*- coding: utf-8 -*-
"""
Enhanced web search module for finding researcher information.

This module uses Google Custom Search API to find comprehensive information
about researchers including homepages, profiles, and academic metrics.

Author: Dr. Denys Dutykh (Khalifa University of Science and Technology, Abu Dhabi, UAE)
Date: 2025-01-04
"""

import re
import json
import time
import asyncio
from typing import Dict, List, Optional, Any, Tuple

try:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    GOOGLE_API_AVAILABLE = True
except ImportError:
    GOOGLE_API_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Import for config and LLM management
from pathlib import Path
import hashlib
from .config_manager import ConfigManager
from .llm_provider import LLMProviderManager
from .scholar_helper import ScholarProfileFinder


class ResearcherWebSearch:
    """Enhanced web search for finding researcher information."""
    
    # Academic profile patterns
    PROFILE_PATTERNS = {
        'orcid': r'orcid\.org/(\d{4}-\d{4}-\d{4}-\d{3}[0-9X])',
        'google_scholar': r'scholar\.google\.com/citations\?.*user=([A-Za-z0-9_-]+)',
        'researchgate': r'researchgate\.net/profile/([A-Za-z0-9_-]+)',
        'linkedin': r'linkedin\.com/in/([A-Za-z0-9_-]+)',
        'github': r'github\.com/([A-Za-z0-9_-]+)',
        'twitter': r'twitter\.com/([A-Za-z0-9_]+)|x\.com/([A-Za-z0-9_]+)',
        'scopus': r'scopus\.com/authid/detail\.uri\?authorId=(\d+)',
        'loop': r'loop\.frontiersin\.org/people/(\d+)',
    }
    
    # Email patterns
    EMAIL_PATTERN = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    
    @staticmethod
    def _repair_json(json_str: str) -> str:
        """
        Attempt to repair common JSON formatting issues.
        
        Args:
            json_str: Potentially malformed JSON string
            
        Returns:
            Repaired JSON string
        """
        # Remove any markdown code blocks
        json_str = re.sub(r'^```(?:json)?\s*\n?', '', json_str, flags=re.MULTILINE)
        json_str = re.sub(r'\n?```\s*$', '', json_str, flags=re.MULTILINE)
        
        # Replace single quotes with double quotes (but not within strings)
        # This is a simple approach - may need refinement for complex cases
        json_str = re.sub(r"(?<![\"'])(\w+)':", r'"\1":', json_str)  # Keys with single quotes
        json_str = re.sub(r":\s*'([^']*)'", r': "\1"', json_str)  # Values with single quotes
        
        # Remove trailing commas
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        # Fix unquoted keys (simple cases)
        json_str = re.sub(r'(\{|\,)\s*([a-zA-Z_]\w*)\s*:', r'\1 "\2":', json_str)
        
        # Remove comments (JavaScript style)
        json_str = re.sub(r'//.*$', '', json_str, flags=re.MULTILINE)
        json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
        
        # Ensure the string ends properly
        json_str = json_str.strip()
        
        # Try to extract just the JSON object if there's extra text
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', json_str)
        if json_match:
            return json_match.group(0)
        
        return json_str
    
    # Academic title patterns
    TITLE_PATTERNS = [
        'Professor', 'Associate Professor', 'Assistant Professor',
        'Research Scientist', 'Postdoctoral', 'Research Fellow',
        'Lecturer', 'Senior Lecturer', 'Reader', 'Chair',
        'Director', 'Dean', 'Head of Department'
    ]
    
    # LLM prompt for extracting profiles from search results
    SEARCH_EXTRACTION_PROMPT = """You are an expert at extracting researcher profile information from Google search results.
    
Given search results for a researcher, extract ONLY verified profile information. We prefer MISSING data over INCORRECT data.

CRITICAL RULES:
- Only extract profiles that you are CERTAIN belong to the researcher
- If unsure, set the profile to null
- Check that names match in titles, URLs, and snippets
- Be extremely careful with ResearchGate profiles - verify the name matches

For each search result item, look for:

1. **Profile URLs and IDs**:
   - Google Scholar: Extract the user ID from URLs like scholar.google.com/citations?user=ABC123
     Pattern to extract: user=([A-Za-z0-9_-]+)
     Example: from "scholar.google.com/citations?user=UdW9A9EAAAAJ" extract "UdW9A9EAAAAJ"
     IMPORTANT: Set to null if no profile URL found. Never use "found" as a value.
   - ORCID: Only extract if the researcher's name is clearly mentioned with the ORCID
   - ResearchGate: CRITICAL - The profile ID must contain the researcher's name
     * For "Jean-Paul Chehab", REJECT "Brachet-Matthieu" or any other person's profile
     * For "Benoît Colange", REJECT "Denys-Dutykh" or profiles with your name
     * The profile should match: First-Last, Last-First, or variations with the actual name
   - LinkedIn, GitHub: Only if name clearly matches

2. **Contact Information**:
   - Email: ONLY if it clearly belongs to THIS researcher
     * Must contain parts of the researcher's name
     * REJECT emails mentioned in co-authored papers that belong to others
   - Homepage: Only institutional pages, not profile sites

3. **Academic Information**:
   - Title: Only if clearly stated for this researcher
   - Research interests: Clean and separate properly

4. **Validation Process**:
   For EACH profile found, ask yourself:
   - Does the name in the profile match the researcher's name?
   - Is this profile mentioned in context with the researcher's affiliation?
   - Could this profile belong to someone else mentioned in the same search result?
   
   If ANY doubt exists, set the profile to null.

EXAMPLES OF WHAT TO REJECT:
- A ResearchGate profile "Brachet-Matthieu" for researcher "Jean-Paul Chehab" 
- An ORCID mentioned in a paper co-authored by multiple people (unless clearly attributed)
- Any profile where the name doesn't match the researcher
     * but REJECT "Shatha-Aldalain" or completely different names
   - If a profile clearly belongs to a different person, set it to null
   - Use affiliation to confirm when names are similar
   - Lower confidence if name matching is uncertain

Return ONLY a valid JSON object with all extracted information. 

CRITICAL JSON FORMATTING RULES:
- Use ONLY double quotes (") for strings, NEVER single quotes (')
- Do NOT include comments in the JSON
- Do NOT add trailing commas
- Return ONLY the JSON object, no additional text before or after
- Ensure all keys are quoted with double quotes
- Use null for missing values, not "null" or undefined

Example of CORRECT response (copy this structure exactly):
{
  "profiles": {
    "google_scholar": "UdW9A9EAAAAJ",
    "orcid": "0000-0000-0000-0000",
    "researchgate": "Profile-Name",
    "linkedin": null,
    "github": null,
    "loop": "1609629"
  },
  "email": "researcher@university.edu",
  "homepage": "https://university.edu/~researcher",
  "title": "Associate Professor",
  "research_interests": ["machine learning", "data science"],
  "metrics": {
    "h_index": 25,
    "citations": 1500
  },
  "extraction_confidence": 0.95,
  "notes": "Found verified institutional email and multiple profiles"
}

Remember: Return ONLY the JSON object above, nothing else!"""
    
    def __init__(self, api_key: str, cx: Optional[str] = None, verbose: int = 2, 
                 use_llm_extraction: bool = True, model: str = "gemini-2.5-flash",
                 config: Optional[ConfigManager] = None):
        """
        Initialize the web search client.
        
        Args:
            api_key: Google API key
            cx: Custom Search Engine ID (optional)
            verbose: Verbosity level
            use_llm_extraction: Whether to use LLM for extracting profiles
            model: Gemini model to use for LLM extraction (deprecated, use config)
            config: Configuration manager (optional, for LLM fallback support)
        """
        self.api_key = api_key
        self.cx = cx  # Custom Search Engine ID
        self.verbose = verbose
        self.use_llm_extraction = use_llm_extraction
        self.model = model
        self.service = None
        self.gemini_model = None
        self.llm_manager = None
        self.config = config
        
        # Initialize caches for LLM operations
        cache_folder = Path("cache")
        cache_folder.mkdir(exist_ok=True)
        self.llm_cache_file = cache_folder / ".llm_websearch_cache.json"
        self.llm_cache = {}
        if self.llm_cache_file.exists():
            try:
                with open(self.llm_cache_file, 'r') as f:
                    self.llm_cache = json.load(f)
            except Exception:
                self.llm_cache = {}
        
        if GOOGLE_API_AVAILABLE and api_key:
            try:
                self.service = build('customsearch', 'v1', developerKey=api_key)
            except Exception as e:
                if self.verbose >= 1:
                    print(f"[WebSearch] Failed to initialize Google API: {str(e)}")
        
        # Initialize LLM for extraction if requested
        if use_llm_extraction:
            # If config is provided, use LLMProviderManager for fallback support
            if config:
                try:
                    self.llm_manager = LLMProviderManager(config)
                    if self.verbose >= 2:
                        print("[WebSearch] Initialized LLM provider with fallback support")
                except Exception as e:
                    if self.verbose >= 1:
                        print(f"[WebSearch] Failed to initialize LLM provider: {str(e)}")
                    self.llm_manager = None
            
            # Fallback to direct Gemini initialization (for backward compatibility)
            if not self.llm_manager and GEMINI_AVAILABLE and api_key:
                try:
                    genai.configure(api_key=api_key)
                    self.gemini_model = genai.GenerativeModel(model)
                    if self.verbose >= 2:
                        print("[WebSearch] Initialized Gemini directly (no fallback)")
                except Exception as e:
                    if self.verbose >= 1:
                        print(f"[WebSearch] Failed to initialize Gemini: {str(e)}")
                    self.gemini_model = None
        
        # Initialize Scholar helper
        self.scholar_helper = ScholarProfileFinder(verbose=verbose)
    
    def search_researcher(self, name: str, affiliation: str, 
                         country: str = None,
                         quality_mode: bool = True,
                         strict_validation: bool = True) -> Dict[str, Any]:
        """
        Comprehensive search for researcher information.
        
        Args:
            name: Researcher's full name
            affiliation: Institution/affiliation
            country: Country (optional)
            
        Returns:
            Dictionary with all found information
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
                'scopus': None,
                'loop': None
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
        
        if not self.service:
            return results
        
        # Perform multiple targeted searches
        if quality_mode:
            # Use comprehensive targeted queries
            search_queries = self._generate_targeted_search_queries(name, affiliation, country, quality_mode=True)
            results_by_category = {}
            
            # Track what we've found
            found_profiles = set()
            
            for query, category in search_queries:
                # Skip if we already found this profile type
                if category in found_profiles and category not in ['general', 'email']:
                    if self.verbose >= 3:
                        print(f"[WebSearch] Skipping {category} query - already found")
                    continue
                
                try:
                    if self.verbose >= 3:
                        print(f"[WebSearch] Executing {category} query: {query}")
                    
                    search_results = self._perform_search(query)
                    
                    # Store results by category
                    if category not in results_by_category:
                        results_by_category[category] = []
                    results_by_category[category].extend(search_results)
                    
                    # Check if we found the profile
                    if category in ['google_scholar', 'researchgate', 'orcid', 'linkedin'] and search_results:
                        # Quick check if we found a relevant result
                        for result in search_results[:3]:  # Check top 3 results
                            if name.lower() in result.get('title', '').lower() or name.lower() in result.get('snippet', '').lower():
                                found_profiles.add(category)
                                break
                    
                    # Longer delay for quality mode
                    time.sleep(1.5)  # 1.5 seconds between queries
                    
                except Exception as e:
                    if self.verbose >= 2:
                        print(f"[WebSearch] Search error for '{query}': {str(e)}")
            
            # Flatten results for processing
            all_search_results = []
            for cat_results in results_by_category.values():
                all_search_results.extend(cat_results)
                
        else:
            # Use standard queries for faster mode
            search_queries = self._generate_search_queries(name, affiliation, country)
            all_search_results = []
            
            for query in search_queries[:4]:  # Use first 4 queries
                try:
                    search_results = self._perform_search(query)
                    all_search_results.extend(search_results)
                    time.sleep(1.0)  # Respect rate limits (1 second between queries)
                except Exception as e:
                    if self.verbose >= 2:
                        print(f"[WebSearch] Search error for '{query}': {str(e)}")
        
        # Process all results
        if all_search_results:
            # First run regex extraction to get base results
            results = self._process_search_results(results, all_search_results, name, affiliation)
            
            # Then try LLM extraction which can override/correct regex results
            if self.use_llm_extraction and (self.llm_manager or self.gemini_model):
                if self.verbose >= 2:
                    print(f"[WebSearch] Attempting LLM extraction for {name}")
                llm_extracted = self._extract_with_llm(all_search_results, name, affiliation)
                if llm_extracted:
                    # Merge LLM results with base results - LLM takes precedence
                    results = self._merge_llm_results(results, llm_extracted)
                    if self.verbose >= 2:
                        print("[WebSearch] LLM extraction successful")
                    
                    # Additional LLM validation step for critical profiles
                    if results['profiles'] and any(results['profiles'].values()):
                        if self.verbose >= 2:
                            print("[WebSearch] Running LLM validation on extracted profiles")
                        validated_profiles = self._validate_profiles_with_llm(
                            results['profiles'], all_search_results, name, affiliation
                        )
                        if validated_profiles is not None:
                            results['profiles'] = validated_profiles
                            if self.verbose >= 2:
                                print("[WebSearch] LLM validation complete")
                else:
                    if self.verbose >= 2:
                        print("[WebSearch] LLM extraction failed, using regex-extracted results as fallback")
            elif self.verbose >= 3:
                print(f"[WebSearch] LLM extraction skipped: use_llm={self.use_llm_extraction}, llm_available={bool(self.llm_manager or self.gemini_model)}")
        
        # Post-process to ensure homepage is valid after LLM validation
        # If the LLM rejected profiles, we should also reject any homepage from those sites
        if results.get('homepage'):
            homepage_lower = results['homepage'].lower()
            
            # Check if homepage is from a profile site that was rejected
            profile_sites = {
                'researchgate.net': 'researchgate',
                'orcid.org': 'orcid',
                'scholar.google': 'google_scholar',
                'linkedin.com': 'linkedin',
                'loop.frontiersin.org': 'loop'
            }
            
            for site_domain, profile_type in profile_sites.items():
                if site_domain in homepage_lower:
                    # If this profile type was rejected (set to None), clear the homepage too
                    if results['profiles'].get(profile_type) is None:
                        if self.verbose >= 2:
                            print(f"[WebSearch] Clearing homepage {results['homepage']} - profile was rejected")
                        results['homepage'] = None
                        break
            
            # Additional check: if LLM notes mention the profile belongs to someone else
            if results.get('additional_info', {}).get('llm_notes'):
                llm_notes = results['additional_info']['llm_notes'].lower()
                if any(phrase in llm_notes for phrase in ['belonged to', 'belongs to', 'not ' + name.split()[0].lower()]):
                    # Clear any ResearchGate or similar profile homepages
                    for site_domain in profile_sites.keys():
                        if site_domain in homepage_lower:
                            if self.verbose >= 2:
                                print(f"[WebSearch] Clearing homepage based on LLM notes: {results['homepage']}")
                            results['homepage'] = None
                            break
        
        # Try to get Google Scholar metrics if profile found
        if results['profiles']['google_scholar']:
            scholar_data = self._get_scholar_metrics(results['profiles']['google_scholar'])
            if scholar_data:
                results['academic_metrics'].update(scholar_data)
        
        # Final email validation - ensure email belongs to researcher
        if results.get('email'):
            if not self._validate_email_for_researcher(results['email'], name):
                if self.verbose >= 2:
                    print(f"[WebSearch] Removing invalid email '{results['email']}' for {name}")
                    # Check if LLM notes mention this email belongs to someone else
                    if results.get('additional_info', {}).get('llm_notes'):
                        llm_notes = results['additional_info']['llm_notes']
                        if 'email' in llm_notes.lower():
                            print(f"[WebSearch] LLM notes: {llm_notes}")
                results['email'] = None
        
        # Calculate confidence score
        results['search_confidence'] = self._calculate_confidence(results)
        
        return results
    
    def _parse_researcher_name(self, name: str) -> Dict[str, Any]:
        """
        Parse researcher name into components and generate variations.
        
        Args:
            name: Full name string
            
        Returns:
            Dictionary with name components and variations
        """
        # Clean and split name
        name_parts = name.strip().split()
        
        result = {
            'full': name,
            'first': '',
            'last': '',
            'middle': [],
            'initials': '',
            'variations': []
        }
        
        if not name_parts:
            return result
        
        # Extract components
        if len(name_parts) == 1:
            result['last'] = name_parts[0]
        elif len(name_parts) == 2:
            result['first'] = name_parts[0]
            result['last'] = name_parts[1]
        else:
            result['first'] = name_parts[0]
            result['last'] = name_parts[-1]
            result['middle'] = name_parts[1:-1]
        
        # Generate initials
        if result['first']:
            result['initials'] = result['first'][0].upper() + '.'
            if result['middle']:
                for m in result['middle']:
                    result['initials'] += ' ' + m[0].upper() + '.'
        
        # Generate variations
        variations = [f'"{name}"']  # Full name in quotes
        
        if result['first'] and result['last']:
            # Last, First format
            variations.append(f'"{result["last"]}, {result["first"]}"')
            
            # F. Last format
            variations.append(f'"{result["first"][0]}. {result["last"]}"')
            
            # First Last (without quotes for broader search)
            variations.append(f'{result["first"]} {result["last"]}')
            
            # With middle initials if present
            if result['middle']:
                middle_initials = ' '.join(m[0].upper() + '.' for m in result['middle'])
                variations.append(f'"{result["first"]} {middle_initials} {result["last"]}"')
        
        result['variations'] = variations
        return result
    
    def _extract_email_domains(self, affiliation: str) -> List[str]:
        """
        Extract likely email domains from affiliation string.
        
        Args:
            affiliation: Institution/affiliation string
            
        Returns:
            List of likely email domains
        """
        domains = []
        affiliation_lower = affiliation.lower()
        
        # Known university to domain mappings
        known_mappings = {
            'tallinn university of technology': ['taltech.ee', 'ttu.ee'],
            'newcastle university': ['ncl.ac.uk', 'newcastle.ac.uk'],
            'university of cambridge': ['cam.ac.uk'],
            'university of oxford': ['ox.ac.uk'],
            'harvard university': ['harvard.edu'],
            'mit': ['mit.edu'],
            'stanford university': ['stanford.edu'],
            'eth zurich': ['ethz.ch'],
            'sorbonne': ['sorbonne-universite.fr', 'upmc.fr'],
            'khalifa university': ['ku.ac.ae'],
            'universiti malaysia': ['umt.edu.my', 'um.edu.my'],
            'université de caen': ['unicaen.fr'],
        }
        
        # Check known mappings
        for uni_name, uni_domains in known_mappings.items():
            if uni_name in affiliation_lower:
                domains.extend(uni_domains)
        
        # Generic patterns
        # Extract existing domains from affiliation
        import re
        domain_pattern = r'([a-z0-9]+(?:-[a-z0-9]+)*\.(?:edu|ac\.[a-z]{2}|edu\.[a-z]{2}|fr|de|it|es|org|ch|ae|my|ee|dk|se|no|fi))'
        found_domains = re.findall(domain_pattern, affiliation_lower)
        domains.extend(found_domains)
        
        # Try to construct domains from university names
        if not domains:
            # Extract key words
            if 'university' in affiliation_lower:
                # Try to extract university name
                uni_match = re.search(r'university\s+(?:of\s+)?([a-z]+)', affiliation_lower)
                if uni_match:
                    uni_name = uni_match.group(1)
                    # Common patterns
                    domains.extend([
                        f'{uni_name}.edu',
                        f'{uni_name}.ac.uk',
                        f'{uni_name}.edu.au'
                    ])
            
            # For institutes
            if 'institute' in affiliation_lower:
                inst_match = re.search(r'([a-z]+)\s+institute', affiliation_lower)
                if inst_match:
                    inst_name = inst_match.group(1)
                    domains.append(f'{inst_name}.edu')
        
        # Remove duplicates and limit to top 3
        unique_domains = list(dict.fromkeys(domains))
        return unique_domains[:3]
    
    def _generate_search_queries(self, name: str, affiliation: str, 
                                country: str = None) -> List[str]:
        """Generate multiple search queries for better coverage."""
        queries = []
        
        # Primary query for general info
        queries.append(f'"{name}" {affiliation}')
        
        # Targeted profile searches - these are most likely to find profiles
        queries.append(f'"{name}" site:scholar.google.com')
        queries.append(f'"{name}" site:orcid.org')
        queries.append(f'"{name}" site:researchgate.net')
        
        # Homepage and institutional page
        queries.append(f'"{name}" {affiliation} homepage')
        
        # With country if provided
        if country:
            queries.append(f'"{name}" {affiliation} {country}')
        
        # Research interests
        queries.append(f'"{name}" {affiliation} "research interests"')
        
        # Combined profile search
        queries.append(f'"{name}" (ORCID OR "Google Scholar" OR ResearchGate)')
        
        return queries
    
    def _generate_targeted_search_queries(self, name: str, affiliation: str,
                                        country: str = None,
                                        quality_mode: bool = True) -> List[Tuple[str, str]]:
        """
        Generate comprehensive targeted search queries with categories.
        
        Args:
            name: Researcher's full name
            affiliation: Institution/affiliation
            country: Country (optional)
            quality_mode: If True, generate more comprehensive queries
            
        Returns:
            List of (query, category) tuples
        """
        queries = []
        
        # Parse name for variations
        name_data = self._parse_researcher_name(name)
        name_variations = name_data['variations']
        
        # Extract email domains
        email_domains = self._extract_email_domains(affiliation)
        
        if quality_mode:
            # Comprehensive query set for quality mode
            
            # Google Scholar searches (highest priority)
            queries.extend([
                (f'{name_variations[0]} site:scholar.google.com', 'google_scholar'),
                (f'{name_variations[0]} "google scholar" profile', 'google_scholar'),
            ])
            if len(name_variations) > 1:
                queries.append((f'{name_variations[1]} scholar.google.com', 'google_scholar'))
            
            # ResearchGate searches
            queries.extend([
                (f'{name_variations[0]} site:researchgate.net', 'researchgate'),
                (f'{name_variations[0]} ResearchGate profile', 'researchgate'),
            ])
            if len(name_variations) > 3:
                queries.append((f'{name_variations[3]} researchgate', 'researchgate'))
            
            # LinkedIn searches
            queries.extend([
                (f'{name_variations[0]} site:linkedin.com/in', 'linkedin'),
                (f'{name_variations[0]} "{affiliation}" LinkedIn', 'linkedin'),
            ])
            
            # ORCID searches
            queries.extend([
                (f'{name_variations[0]} ORCID', 'orcid'),
                (f'{name_variations[0]} site:orcid.org', 'orcid'),
            ])
            
            # Email searches
            queries.append((f'{name_variations[0]} email contact', 'email'))
            if email_domains:
                queries.append((f'{name_variations[0]} "@{email_domains[0]}"', 'email'))
                queries.append((f'"{name}" email {affiliation}', 'email'))
            
            # Homepage searches
            queries.append((f'{name_variations[0]} {affiliation} homepage', 'homepage'))
            if email_domains:
                queries.append((f'{name_variations[0]} site:{email_domains[0]}', 'homepage'))
            
            # General searches
            queries.append((f'{name_variations[0]} {affiliation}', 'general'))
            if country:
                queries.append((f'{name_variations[0]} {affiliation} {country}', 'general'))
            
            # Additional academic profiles
            queries.extend([
                (f'{name_variations[0]} Scopus author', 'scopus'),
                (f'{name_variations[0]} Loop profile', 'loop'),
                (f'{name_variations[0]} Academia.edu', 'academia'),
            ])
            
        else:
            # Faster mode with fewer queries
            queries.extend([
                (f'"{name}" site:scholar.google.com', 'google_scholar'),
                (f'"{name}" site:researchgate.net', 'researchgate'),
                (f'"{name}" site:linkedin.com/in', 'linkedin'),
                (f'"{name}" ORCID', 'orcid'),
                (f'"{name}" email contact', 'email'),
                (f'"{name}" {affiliation}', 'general'),
            ])
        
        return queries
    
    def _perform_search(self, query: str) -> List[Dict[str, Any]]:
        """Perform a single Google search."""
        if not self.service:
            if self.verbose >= 2:
                print("[WebSearch] Service not initialized")
            return []
        
        try:
            # Use custom search engine if provided
            if self.cx:
                if self.verbose >= 3:
                    print(f"[WebSearch] Searching with query: '{query}'")
                result = self.service.cse().list(q=query, cx=self.cx, num=10).execute()
                items = result.get('items', [])
                if self.verbose >= 3:
                    print(f"[WebSearch] Found {len(items)} results")
                return items
            else:
                if self.verbose >= 1:
                    print("[WebSearch] No Custom Search Engine ID (cx) configured. Set 'google_cx' in config.")
                return []
        
        except HttpError as e:
            if self.verbose >= 1:
                error_content = getattr(e, 'content', str(e))
                if "Invalid Value" in str(e) or "invalid" in str(e).lower():
                    print("[WebSearch] Invalid Custom Search Engine ID. Please check your 'google_cx' configuration.")
                elif "quotaExceeded" in str(error_content):
                    print("[WebSearch] API quota exceeded. Try again tomorrow.")
                else:
                    print(f"[WebSearch] HTTP error: {e}")
            return []
        except Exception as e:
            if self.verbose >= 1:
                print(f"[WebSearch] Search error: {e}")
            return []
    
    def _process_search_results(self, results: Dict[str, Any], 
                               search_results: List[Dict[str, Any]], 
                               name: str, affiliation: str) -> Dict[str, Any]:
        """Process search results to extract information."""
        name_parts = name.lower().split()
        affiliation_parts = affiliation.lower().split()
        
        for item in search_results:
            url = item.get('link', '')
            title = item.get('title', '').lower()
            snippet = item.get('snippet', '').lower()
            
            # Check if this result is relevant
            relevance_score = self._calculate_relevance(
                title + ' ' + snippet, name_parts, affiliation_parts
            )
            
            if relevance_score < 0.3:  # Skip low relevance results
                continue
            
            # Extract homepage
            if not results['homepage'] and self._is_likely_homepage(url, title, snippet):
                results['homepage'] = url
            
            # Extract profiles with validation
            for profile_type, pattern in self.PROFILE_PATTERNS.items():
                if not results['profiles'][profile_type]:
                    match = re.search(pattern, url)
                    if match:
                        profile_id = match.group(1)
                        # Validate that this profile belongs to the researcher
                        if self._validate_profile_ownership(profile_type, profile_id, name, title, snippet):
                            results['profiles'][profile_type] = profile_id
                        elif self.verbose >= 3:
                            print(f"[WebSearch] Rejected {profile_type} profile '{profile_id}' - doesn't match researcher '{name}'")
            
            # Extract email
            if not results['email']:
                email_match = re.search(self.EMAIL_PATTERN, snippet)
                if email_match:
                    potential_email = email_match.group(0)
                    # Validate email belongs to the researcher
                    if self._validate_email_for_researcher(potential_email, name):
                        results['email'] = potential_email
                    elif self.verbose >= 3:
                        print(f"[WebSearch] Rejected email '{potential_email}' - doesn't match researcher name '{name}'")
            
            # Extract title
            if not results['title']:
                for title_keyword in self.TITLE_PATTERNS:
                    if title_keyword.lower() in snippet:
                        results['title'] = title_keyword
                        break
            
            # Extract research interests from snippet
            if 'research interests' in snippet or 'research areas' in snippet or 'research topics' in snippet:
                interests = self._extract_research_interests(snippet)
                results['research_interests'].extend(interests)
        
        # Deduplicate research interests
        results['research_interests'] = list(set(results['research_interests']))[:10]
        
        return results
    
    def _calculate_relevance(self, text: str, name_parts: List[str], 
                           affiliation_parts: List[str]) -> float:
        """Calculate relevance score for a search result."""
        score = 0.0
        text = text.lower()
        
        # Check for name parts
        name_matches = sum(1 for part in name_parts if part in text)
        score += name_matches / len(name_parts) * 0.5
        
        # Check for affiliation parts
        aff_matches = sum(1 for part in affiliation_parts if len(part) > 3 and part in text)
        score += aff_matches / len(affiliation_parts) * 0.3
        
        # Bonus for academic keywords
        academic_keywords = ['professor', 'research', 'university', 'department', 'faculty']
        academic_matches = sum(1 for keyword in academic_keywords if keyword in text)
        score += min(academic_matches * 0.05, 0.2)
        
        return min(score, 1.0)
    
    def _is_likely_homepage(self, url: str, title: str, snippet: str) -> bool:
        """Determine if a URL is likely a personal homepage."""
        # First, exclude known research profile sites that should NOT be treated as homepages
        excluded_domains = [
            'researchgate.net',
            'orcid.org',
            'scholar.google',
            'scopus.com',
            'linkedin.com',
            'github.com',
            'twitter.com',
            'loop.frontiersin.org',
            'publons.com',
            'webofscience.com',
            'academia.edu'
        ]
        
        url_lower = url.lower()
        for domain in excluded_domains:
            if domain in url_lower:
                return False
        
        # Check URL patterns for actual homepages
        homepage_patterns = [
            r'/~\w+/?$',  # Unix-style personal pages
            r'/people/\w+',
            r'/faculty/\w+',
            r'/staff/\w+',
            r'/profile/\w+',  # Keep this but exclude research sites above
            r'homepage',
            r'personal.*page'
        ]
        
        for pattern in homepage_patterns:
            if re.search(pattern, url, re.I):
                return True
        
        # Check title/snippet for homepage indicators
        homepage_keywords = ['homepage', 'personal page', 'faculty page', 'staff profile']
        text = (title + ' ' + snippet).lower()
        
        # But exclude if it mentions it's someone else's page
        exclude_keywords = ['directory', 'list of', 'department of', 'research group']
        if any(keyword in text for keyword in exclude_keywords):
            return False
        
        return any(keyword in text for keyword in homepage_keywords)
    
    def _clean_research_interests(self, interests: List[str]) -> List[str]:
        """
        Clean up research interests that may have missing spaces.
        
        Args:
            interests: List of potentially malformed interest strings
            
        Returns:
            Cleaned list of research interests
            
        Examples:
            "Tsunamisolitary waveswave runupFreak waves" 
            → ["Tsunami", "solitary waves", "wave runup", "Freak waves"]
        """
        cleaned = []
        
        # Known problematic concatenations
        known_fixes = {
            'tsunamisolitary': 'tsunami, solitary',
            'waveswave': 'waves, wave',
            'runupfreak': 'runup, freak',
            'runupFreak': 'runup, Freak',
            'wavesstorm': 'waves, storm',
            'dynamicscoastal': 'dynamics, coastal',
        }
        
        for interest in interests:
            # Apply all known fixes in sequence
            for problem, fix in known_fixes.items():
                if problem in interest:
                    interest = interest.replace(problem, fix)
                elif problem in interest.lower():
                    interest = re.sub(re.escape(problem), fix, interest, flags=re.I)
            
            # Check if it looks like concatenated words (no spaces but mixed case)
            if ' ' not in interest and len(interest) > 10 and any(c.isupper() for c in interest[1:]):
                # Split on lowercase followed by uppercase
                parts = re.sub(r'([a-z])([A-Z])', r'\1, \2', interest)
                # Split on common word endings followed by lowercase
                parts = re.sub(r'(waves|wave|runup|dynamics|coastal|ocean|marine)([a-z])', r'\1, \2', parts, flags=re.I)
                # Clean up and split
                items = [item.strip() for item in parts.split(',') if item.strip()]
                cleaned.extend(items)
            elif any(delim in interest for delim in [',', ';', '•', '·', '|', '/']):
                # Split by common delimiters
                items = re.split(r'[,;•·|/]', interest)
                cleaned.extend([item.strip() for item in items if item.strip() and len(item.strip()) > 2])
            else:
                # Add as-is if it looks reasonable
                if interest.strip() and len(interest.strip()) > 2:
                    cleaned.append(interest.strip())
        
        # Remove duplicates while preserving order
        seen = set()
        unique_cleaned = []
        for item in cleaned:
            item_lower = item.lower()
            if item_lower not in seen:
                seen.add(item_lower)
                unique_cleaned.append(item)
        
        return unique_cleaned
    
    def _validate_email_for_researcher(self, email: str, researcher_name: str) -> bool:
        """
        Validate if an email likely belongs to the researcher.
        
        Args:
            email: Email address to validate
            researcher_name: Full name of the researcher
            
        Returns:
            True if email likely belongs to researcher, False otherwise
        """
        if not email or not researcher_name:
            return False
            
        # Extract email username (part before @)
        email_username = email.split('@')[0].lower()
        
        # Strip numbers from email username for validation
        # This handles common cases like birth years, random numbers, etc.
        email_username_no_numbers = re.sub(r'\d+', '', email_username)
        
        # Parse researcher name - handle complex names
        name_parts = researcher_name.lower().split()
        # Remove common words that might not appear in emails
        name_parts = [p for p in name_parts if p not in ['bin', 'binti', 'ibn', 'bint', 'al', 'el', 'de', 'van', 'von', 'della', 'di', 'da']]
        
        if not name_parts:
            return False
            
        first_name = name_parts[0] if name_parts else ""
        last_name = name_parts[-1] if len(name_parts) > 1 else ""
        middle_names = name_parts[1:-1] if len(name_parts) > 2 else []
        
        # Common email patterns to check
        valid_patterns = []
        
        if first_name and last_name:
            valid_patterns.extend([
                first_name,  # john@
                last_name,   # smith@
                f"{first_name}.{last_name}",  # john.smith@
                f"{first_name}_{last_name}",  # john_smith@
                f"{first_name}-{last_name}",  # john-smith@
                f"{first_name[0]}{last_name}",  # jsmith@
                f"{first_name}{last_name[0]}",  # johns@
                f"{last_name}.{first_name}",  # smith.john@
                f"{last_name}_{first_name}",  # smith_john@
                f"{last_name}{first_name[0]}",  # smithj@
                f"{first_name[0]}.{last_name}",  # j.smith@
            ])
            
            # Add patterns with middle initials
            if middle_names:
                middle_initial = middle_names[0][0]
                valid_patterns.extend([
                    f"{first_name}{middle_initial}{last_name}",  # johnmsmith@
                    f"{first_name}.{middle_initial}.{last_name}",  # john.m.smith@
                ])
        elif first_name:  # Only first name available
            valid_patterns.append(first_name)
        elif last_name:  # Only last name available
            valid_patterns.append(last_name)
        
        # Check if any pattern matches (exact match or contained in email)
        # Check both with and without numbers
        for pattern in valid_patterns:
            if pattern and (pattern == email_username or pattern in email_username or 
                           pattern == email_username_no_numbers or pattern in email_username_no_numbers):
                return True
        
        # Check if significant parts of the name appear in email
        significant_parts = [part for part in name_parts if len(part) > 2]
        if significant_parts:
            parts_in_email = sum(1 for part in significant_parts 
                               if part in email_username or part in email_username_no_numbers)
            # Require at least 2 parts for complex names, or 1 for simple names
            required_parts = min(2, len(significant_parts))
            if parts_in_email >= required_parts:
                return True
        
        # Check for substrings of ANY name part (length 4+)
        # This catches cases like "karory" from "Alkarory", "brah" from "Ibrahim", etc.
        for name_part in name_parts:
            if len(name_part) >= 4:
                # Check all substrings of length 4 or more
                for i in range(len(name_part) - 3):
                    for j in range(i + 4, len(name_part) + 1):
                        substring = name_part[i:j]
                        if substring in email_username or substring in email_username_no_numbers:
                            if self.verbose >= 3:
                                print(f"[WebSearch] Found substring '{substring}' from '{name_part}' in email username")
                            return True
        
        # Special case: Check for initials-based patterns
        if len(name_parts) >= 2:
            # Check for patterns like "na" for "Nizar Abcha"
            initials = ''.join(part[0] for part in name_parts[:2])
            if len(initials) >= 2 and (initials in email_username or initials in email_username_no_numbers):
                return True
        
        return False
    
    def _extract_research_interests(self, text: str) -> List[str]:
        """Extract research interests from text."""
        interests = []
        
        # Look for common patterns
        patterns = [
            r'research interests?:?\s*([^.]+)',
            r'interests? include:?\s*([^.]+)',
            r'working on:?\s*([^.]+)',
            r'focuses? on:?\s*([^.]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.I)
            if match:
                interest_text = match.group(1)
                # Split by common delimiters
                items = re.split(r'[,;]|\band\b', interest_text)
                interests.extend([item.strip() for item in items if item.strip()])
        
        # Clean the extracted interests
        cleaned_interests = self._clean_research_interests(interests)
        return cleaned_interests[:5]  # Limit to 5 interests
    
    def _get_scholar_metrics(self, scholar_id: str) -> Optional[Dict[str, Any]]:
        """Get Google Scholar metrics for a researcher."""
        # Note: This would typically require scholarly library or web scraping
        # For now, returning placeholder
        return {
            'source': 'google_scholar',
            'scholar_id': scholar_id
        }
    
    def _calculate_confidence(self, results: Dict[str, Any]) -> float:
        """Calculate confidence score for the search results."""
        score = 0.0
        
        # Points for finding different types of information
        if results['homepage']:
            score += 0.2
        if results['email']:
            score += 0.15
        if results['title']:
            score += 0.1
        
        # Points for profiles
        profile_score = sum(0.1 for p in results['profiles'].values() if p)
        score += min(profile_score, 0.3)
        
        # Points for academic metrics
        if any(results['academic_metrics'].values()):
            score += 0.15
        
        # Points for research interests
        if results['research_interests']:
            score += 0.1
        
        return min(score, 1.0)
    
    async def search_multiple_researchers(self, researchers: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Search for multiple researchers concurrently.
        
        Args:
            researchers: List of dicts with 'name', 'affiliation', and optionally 'country'
            
        Returns:
            List of search results
        """
        async def search_one(researcher):
            return self.search_researcher(
                researcher['name'],
                researcher['affiliation'],
                researcher.get('country')
            )
        
        # Use asyncio for concurrent searches
        tasks = [search_one(r) for r in researchers]
        
        # Process in batches to avoid rate limits
        batch_size = 5
        all_results = []
        
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            batch_results = await asyncio.gather(*batch)
            all_results.extend(batch_results)
            
            # Rate limiting between batches
            if i + batch_size < len(tasks):
                await asyncio.sleep(2)
        
        return all_results
    
    def _validate_profile_ownership(self, profile_type: str, profile_id: str, 
                                   researcher_name: str, title: str, snippet: str) -> bool:
        """
        Strictly validate that a profile belongs to the researcher.
        
        Args:
            profile_type: Type of profile (researchgate, orcid, etc.)
            profile_id: The extracted profile ID
            researcher_name: Full name of the researcher
            title: Title of the search result
            snippet: Snippet text from search result
            
        Returns:
            True only if we're confident the profile belongs to the researcher
        """
        # Parse researcher name
        name_parts = researcher_name.lower().split()
        # Remove common particles that might not appear in profiles
        name_parts = [p for p in name_parts if p not in ['bin', 'binti', 'ibn', 'bint', 'al', 'el', 'de', 'van', 'von', 'della', 'di', 'da']]
        
        if not name_parts:
            return False
        
        first_name = name_parts[0] if name_parts else ""
        last_name = name_parts[-1] if len(name_parts) > 1 else ""
        
        # Combine title and snippet for full context
        full_context = f"{title} {snippet}".lower()
        
        # Profile-specific validation
        if profile_type == 'researchgate':
            # ResearchGate profiles should have the researcher's name in the title or URL
            # Remove numbers from profile ID for comparison
            profile_id_clean = re.sub(r'[-_]\d+$', '', profile_id).lower()
            
            # Check if profile ID contains name parts
            name_in_profile = False
            if first_name and last_name:
                # Check various combinations
                if (first_name in profile_id_clean and last_name in profile_id_clean) or \
                   (f"{first_name}-{last_name}" in profile_id_clean) or \
                   (f"{last_name}-{first_name}" in profile_id_clean):
                    name_in_profile = True
            elif first_name and first_name in profile_id_clean:
                name_in_profile = True
            elif last_name and last_name in profile_id_clean:
                name_in_profile = True
            
            # Also check if the full name appears in the title
            name_in_title = False
            if first_name and last_name:
                if (first_name in title.lower() and last_name in title.lower()) or \
                   researcher_name.lower() in title.lower():
                    name_in_title = True
            
            # For ResearchGate, require name match in profile ID or clear mention in title
            if not (name_in_profile or name_in_title):
                if self.verbose >= 3:
                    print(f"[WebSearch] ResearchGate validation failed: profile_id='{profile_id}', name='{researcher_name}'")
                return False
                
        elif profile_type == 'orcid':
            # ORCID should have the researcher's name clearly mentioned
            if not (first_name in full_context and last_name in full_context):
                return False
                
        elif profile_type == 'google_scholar':
            # Google Scholar should have clear name match
            if not (researcher_name.lower() in full_context or 
                   (first_name in full_context and last_name in full_context)):
                return False
                
        elif profile_type == 'linkedin':
            # LinkedIn profiles are often less reliable, require strong name match
            if not (researcher_name.lower() in title.lower() or
                   (first_name in full_context and last_name in full_context)):
                return False
        
        # Additional validation: Check for explicit mentions of other people
        # If the snippet mentions this profile belongs to someone else, reject it
        other_person_patterns = [
            r'co-author[ed]?\s+(?:by|with)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)',
            r'([A-Z][a-z]+\s+[A-Z][a-z]+)\s+(?:and|&)\s+' + re.escape(researcher_name),
            r'(?:by|author:)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)',
        ]
        
        for pattern in other_person_patterns:
            match = re.search(pattern, snippet)
            if match:
                other_name = match.group(1)
                # If this other name doesn't match our researcher, be cautious
                if other_name.lower() != researcher_name.lower():
                    # Check if this other person's name appears in the profile
                    other_parts = other_name.lower().split()
                    if any(part in profile_id.lower() for part in other_parts if len(part) > 3):
                        if self.verbose >= 3:
                            print(f"[WebSearch] Profile might belong to '{other_name}' not '{researcher_name}'")
                        return False
        
        # If we made it here, we have reasonable confidence
        return True
    
    def _validate_profiles_with_llm(self, profiles: Dict[str, Optional[str]], 
                                   search_results: List[Dict[str, Any]],
                                   name: str, affiliation: str) -> Optional[Dict[str, Optional[str]]]:
        """
        Use LLM to validate extracted profiles against search results.
        This is a second pass to ensure accuracy.
        
        Args:
            profiles: Currently extracted profiles
            search_results: Raw search results
            name: Researcher name
            affiliation: Researcher affiliation
            
        Returns:
            Validated profiles or None if validation fails
        """
        if not (self.llm_manager or self.gemini_model) or not profiles:
            return profiles
        
        # Only validate profiles that have values
        profiles_to_validate = {k: v for k, v in profiles.items() if v and v != 'found'}
        if not profiles_to_validate:
            return profiles
        
        validation_prompt = f"""You are a strict validator of researcher profiles. Your job is to verify that extracted profiles actually belong to the researcher.

Researcher Information:
- Name: {name}
- Affiliation: {affiliation}

Extracted Profiles to Validate:
{json.dumps(profiles_to_validate, indent=2)}

Search Results Context:
{json.dumps([{
    'title': r.get('title', ''),
    'snippet': r.get('snippet', ''),
    'link': r.get('link', '')
} for r in search_results[:10]], indent=2)}

For EACH profile listed above:
1. Check if the profile ID/username contains the researcher's name
2. Verify the search results mention this profile in connection with the researcher
3. Ensure the profile doesn't belong to a co-author or someone else

Return a JSON object with your validation results:
{{
  "validated_profiles": {{
    "profile_type": "profile_id or null if invalid",
    ...
  }},
  "validation_notes": {{
    "profile_type": "reason for rejection if applicable",
    ...
  }},
  "confidence": 0.0-1.0
}}

BE VERY STRICT: If you have ANY doubt about a profile belonging to {name}, set it to null.
Examples of what to reject:
- ResearchGate profile "Brachet-Matthieu" for "Jean-Paul Chehab"
- ORCID mentioned in a multi-author paper without clear attribution
- Any profile where the name doesn't match
"""
        
        try:
            # Use LLM manager if available (with fallback), otherwise use Gemini directly
            if self.llm_manager:
                result_text = self.llm_manager.generate(
                    validation_prompt,
                    temperature=0.1,
                    max_tokens=1024
                )
            else:
                # Add timeout for direct Gemini call
                from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
                timeout_seconds = 30
                
                def _generate():
                    return self.gemini_model.generate_content(
                        validation_prompt,
                        generation_config={
                            'temperature': 0.1,
                            'max_output_tokens': 1024,
                        }
                    )
                
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_generate)
                    try:
                        response = future.result(timeout=timeout_seconds)
                        result_text = response.text.strip()
                    except FutureTimeoutError:
                        raise Exception(f"Gemini validation timed out after {timeout_seconds} seconds")
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            
            if json_match:
                validation_result = json.loads(json_match.group(0))
                
                if self.verbose >= 3:
                    print(f"[WebSearch] LLM validation result: {json.dumps(validation_result, indent=2)}")
                
                # Update profiles based on validation
                validated = profiles.copy()
                if 'validated_profiles' in validation_result:
                    for profile_type, profile_value in validation_result['validated_profiles'].items():
                        if profile_type in validated:
                            if profile_value is None or profile_value == 'null':
                                if self.verbose >= 2 and validated[profile_type]:
                                    reason = validation_result.get('validation_notes', {}).get(profile_type, 'Failed validation')
                                    print(f"[WebSearch] LLM rejected {profile_type} '{validated[profile_type]}': {reason}")
                                validated[profile_type] = None
                            else:
                                validated[profile_type] = profile_value
                
                return validated
            
            return profiles
            
        except Exception as e:
            if self.verbose >= 2:
                print(f"[WebSearch] LLM validation error: {str(e)}")
            return profiles
    
    def _extract_with_llm(self, search_results: List[Dict[str, Any]], 
                         name: str, affiliation: str) -> Dict[str, Any]:
        """
        Use LLM to extract profile information from search results.
        Supports fallback to alternative LLM if primary fails.
        
        Args:
            search_results: Raw Google search results
            name: Researcher name
            affiliation: Researcher affiliation
            
        Returns:
            Extracted profile information
        """
        if not (self.llm_manager or self.gemini_model) or not search_results:
            return {}
        
        # Create cache key
        cache_key = hashlib.md5(f"extract:{name}:{affiliation}:{len(search_results)}".encode()).hexdigest()
        
        # Check cache
        if cache_key in self.llm_cache:
            if self.verbose >= 3:
                print("[WebSearch] LLM extraction cache hit")
            return self.llm_cache[cache_key]
        
        # Prepare search results for LLM
        formatted_results = []
        for idx, item in enumerate(search_results[:10]):  # Limit to 10 results
            formatted_results.append({
                'index': idx + 1,
                'title': item.get('title', ''),
                'link': item.get('link', ''),
                'snippet': item.get('snippet', '')
            })
        
        # Debug: show what the LLM will see
        if self.verbose >= 3:
            print("[WebSearch] Formatted results for LLM:")
            for result in formatted_results[:3]:  # Show first 3
                print(f"  #{result['index']}: {result['title'][:60]}...")
                print(f"      Link: {result['link'][:80]}...")
                print(f"      Snippet: {result['snippet'][:100]}...")
        
        # Create prompt
        prompt = f"""
{self.SEARCH_EXTRACTION_PROMPT}

Researcher Information:
- Name: {name}
- Affiliation: {affiliation}

IMPORTANT: Only extract profiles that belong to THIS specific person ({name}).
If a profile name doesn't match (e.g., completely different name), set it to null.

Google Search Results:
{json.dumps(formatted_results, indent=2)}

Extract all available profile information from these search results.
Verify each profile belongs to {name} before including it.

Return ONLY the JSON object, no other text.
"""
        
        try:
            if self.verbose >= 3:
                print(f"[WebSearch] Using LLM to extract profiles for {name}")
            
            # Use LLM manager if available (with fallback), otherwise use Gemini directly
            if self.llm_manager:
                result_text = self.llm_manager.generate(
                    prompt,
                    temperature=0.1,
                    max_tokens=2048
                )
            else:
                # Add timeout for direct Gemini calls
                from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
                timeout_seconds = 30
                
                def _generate():
                    return self.gemini_model.generate_content(
                        prompt,
                        generation_config={
                            'temperature': 0.1,
                            'max_output_tokens': 2048,
                        }
                    )
                
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_generate)
                    try:
                        response = future.result(timeout=timeout_seconds)
                        result_text = response.text.strip()
                    except FutureTimeoutError:
                        raise Exception(f"Gemini timed out after {timeout_seconds} seconds")
            
            # Extract JSON from the response
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                try:
                    extracted_data = json.loads(json_str)
                except json.JSONDecodeError as e:
                    if self.verbose >= 2:
                        print(f"[WebSearch] Initial JSON parse failed: {str(e)}")
                        print(f"[WebSearch] Raw JSON string: {json_str[:200]}...")
                    
                    # Try to repair the JSON
                    repaired_json = self._repair_json(json_str)
                    if self.verbose >= 3:
                        print(f"[WebSearch] Attempting repair. Repaired JSON: {repaired_json[:200]}...")
                    
                    try:
                        extracted_data = json.loads(repaired_json)
                        if self.verbose >= 2:
                            print("[WebSearch] JSON repair successful!")
                    except json.JSONDecodeError as e2:
                        if self.verbose >= 1:
                            print(f"[WebSearch] JSON repair failed: {str(e2)}")
                            print(f"[WebSearch] Failed JSON (first 500 chars): {repaired_json[:500]}")
                        raise e2
                
                # Verify extracted profiles match the researcher name
                if 'profiles' in extracted_data:
                    extracted_data['profiles'] = self._verify_profile_names(
                        extracted_data['profiles'], name
                    )
                
                if self.verbose >= 3:
                    print(f"[WebSearch] LLM extracted: {json.dumps(extracted_data, indent=2)}")
                
                # Fallback: If Google Scholar is "found" or None, try harder to find it
                if not extracted_data.get('profiles', {}).get('google_scholar') or extracted_data.get('profiles', {}).get('google_scholar') == 'found':
                    if self.verbose >= 2:
                        print("[WebSearch] Google Scholar not found via LLM, trying alternative extraction...")
                    
                    # First, search through all current results for Google Scholar URLs
                    scholar_found = False
                    for result in search_results:
                        url = result.get('link', '')
                        match = re.search(self.PROFILE_PATTERNS['google_scholar'], url)
                        if match:
                            scholar_id = match.group(1)
                            # Validate it's for this researcher by checking title/snippet
                            title = result.get('title', '').lower()
                            snippet = result.get('snippet', '').lower()
                            if name.lower() in title or name.lower() in snippet:
                                extracted_data['profiles']['google_scholar'] = scholar_id
                                if self.verbose >= 2:
                                    print(f"[WebSearch] Extracted Google Scholar ID from search results: {scholar_id}")
                                scholar_found = True
                                break
                    
                    # If still not found, try a broader search specifically for Google Scholar
                    if not scholar_found:
                        if self.verbose >= 2:
                            print("[WebSearch] Trying broader Google Scholar search...")
                        broader_query = f'{name} Google Scholar profile'
                        try:
                            broader_results = self._perform_search(broader_query)
                            for result in broader_results[:10]:
                                url = result.get('link', '')
                                match = re.search(self.PROFILE_PATTERNS['google_scholar'], url)
                                if match:
                                    scholar_id = match.group(1)
                                    title = result.get('title', '').lower()
                                    snippet = result.get('snippet', '').lower()
                                    # More lenient matching for broader search
                                    name_parts = name.lower().split()
                                    if any(part in title or part in snippet for part in name_parts if len(part) > 3):
                                        extracted_data['profiles']['google_scholar'] = scholar_id
                                        if self.verbose >= 2:
                                            print(f"[WebSearch] Found Google Scholar ID in broader search: {scholar_id}")
                                        scholar_found = True
                                        break
                        except Exception as e:
                            if self.verbose >= 2:
                                print(f"[WebSearch] Broader search failed: {e}")
                    
                    # If still not found, use Scholar helper as last resort
                    if not scholar_found:
                        if self.verbose >= 2:
                            print("[WebSearch] Using Scholar helper for known profiles...")
                        scholar_id = self.scholar_helper.find_scholar_id(name, search_results)
                        if scholar_id:
                            extracted_data['profiles']['google_scholar'] = scholar_id
                            if self.verbose >= 2:
                                print(f"[WebSearch] Scholar helper found ID: {scholar_id}")
                
                # Cache the result
                self.llm_cache[cache_key] = extracted_data
                self._save_llm_cache()
                
                return extracted_data
            else:
                if self.verbose >= 2:
                    print("[WebSearch] LLM response did not contain valid JSON")
                    print(f"[WebSearch] Raw LLM response (first 500 chars): {result_text[:500]}")
                return {}
                
        except Exception as e:
            if self.verbose >= 2:
                print(f"[WebSearch] LLM extraction error: {str(e)}")
                # Show the raw response if we have it
                if 'result_text' in locals():
                    print(f"[WebSearch] Raw LLM response (first 500 chars): {result_text[:500]}")
            
            # Try a simpler prompt as fallback
            if self.verbose >= 2:
                print("[WebSearch] Attempting extraction with simplified prompt...")
            
            simple_prompt = f"""Extract researcher profiles from these search results for {name}.

Search results:
{json.dumps([{'title': r['title'], 'link': r['link']} for r in formatted_results[:5]], indent=2)}

Return a simple JSON object with found profiles. Use this exact format:
{{
  "profiles": {{
    "google_scholar": null,
    "orcid": null,
    "researchgate": null,
    "linkedin": null
  }},
  "email": null,
  "homepage": null
}}

Fill in the values you find, keep null for missing data. Return ONLY the JSON."""
            
            try:
                if self.llm_manager:
                    result_text = self.llm_manager.generate(
                        simple_prompt,
                        temperature=0.1,
                        max_tokens=512
                    )
                else:
                    # Direct Gemini call with timeout
                    timeout_seconds = 30
                    def _generate():
                        return self.gemini_model.generate_content(
                            simple_prompt,
                            generation_config={
                                'temperature': 0.1,
                                'max_output_tokens': 512,
                            }
                        )
                    
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(_generate)
                        response = future.result(timeout=timeout_seconds)
                        result_text = response.text.strip()
                
                # Try to parse the simplified response
                json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                if json_match:
                    simplified_data = json.loads(self._repair_json(json_match.group(0)))
                    if self.verbose >= 2:
                        print("[WebSearch] Simplified extraction successful")
                    
                    # Cache the simplified result
                    self.llm_cache[cache_key] = simplified_data
                    self._save_llm_cache()
                    
                    return simplified_data
            except Exception as e2:
                if self.verbose >= 2:
                    print(f"[WebSearch] Simplified extraction also failed: {str(e2)}")
            
            return {}
    
    def _merge_llm_results(self, base_results: Dict[str, Any], 
                          llm_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge LLM-extracted results with base results.
        
        Args:
            base_results: Current results dictionary
            llm_results: Results extracted by LLM
            
        Returns:
            Merged results with LLM data taking precedence where available
        """
        # Update profiles - LLM results take precedence
        if 'profiles' in llm_results:
            for profile_type in base_results['profiles'].keys():
                # If LLM has a value for this profile type (including None), use it
                if profile_type in llm_results['profiles']:
                    base_results['profiles'][profile_type] = llm_results['profiles'][profile_type]
            
            # Also add any new profile types from LLM
            for profile_type, profile_id in llm_results['profiles'].items():
                if profile_type not in base_results['profiles']:
                    base_results['profiles'][profile_type] = profile_id
        
        # Update email
        if llm_results.get('email') and not base_results['email']:
            base_results['email'] = llm_results['email']
        
        # Update homepage
        if llm_results.get('homepage') and not base_results['homepage']:
            base_results['homepage'] = llm_results['homepage']
        
        # Update title
        if llm_results.get('title') and not base_results['title']:
            base_results['title'] = llm_results['title']
        
        # Update research interests
        if llm_results.get('research_interests'):
            # Clean the interests before adding
            cleaned_interests = self._clean_research_interests(llm_results['research_interests'])
            base_results['research_interests'].extend(cleaned_interests)
            # Remove duplicates while preserving order
            seen = set()
            unique_interests = []
            for interest in base_results['research_interests']:
                if interest.lower() not in seen:
                    seen.add(interest.lower())
                    unique_interests.append(interest)
            base_results['research_interests'] = unique_interests[:10]
        
        # Update metrics
        if llm_results.get('metrics'):
            for metric, value in llm_results['metrics'].items():
                if value and not base_results['academic_metrics'].get(metric):
                    base_results['academic_metrics'][metric] = value
        
        # Add LLM extraction confidence
        if llm_results.get('extraction_confidence'):
            base_results['additional_info']['llm_extraction_confidence'] = llm_results['extraction_confidence']
        
        # Add LLM notes
        if llm_results.get('notes'):
            base_results['additional_info']['llm_notes'] = llm_results['notes']
        
        return base_results
    
    def _verify_profile_names(self, profiles: Dict[str, Optional[str]], 
                             researcher_name: str) -> Dict[str, Optional[str]]:
        """
        Strictly verify that extracted profiles belong to the researcher.
        We prefer missing data over incorrect data.
        
        Args:
            profiles: Dictionary of profile types to IDs/names
            researcher_name: Full name of the researcher
            
        Returns:
            Cleaned profiles dictionary with invalid entries set to null
        """
        # Split researcher name into parts for matching
        name_parts = researcher_name.lower().split()
        # Remove common words that might not appear in profiles
        name_parts = [p for p in name_parts if p not in ['bin', 'binti', 'ibn', 'bint', 'al', 'el', 'de', 'van', 'von']]
        
        # Get significant name parts (longer than 3 characters)
        significant_parts = [p for p in name_parts if len(p) > 3]
        if not significant_parts:
            significant_parts = name_parts  # Use all parts if none are long enough
        
        verified_profiles = profiles.copy()
        
        # Check ResearchGate profiles - VERY STRICT
        if profiles.get('researchgate'):
            rg_profile = profiles['researchgate'].lower()
            # Remove trailing numbers for comparison
            rg_profile_clean = re.sub(r'[-_]\d+$', '', rg_profile)
            
            # Count how many significant name parts appear in the profile
            matching_parts = sum(1 for part in significant_parts if part in rg_profile_clean)
            
            # For ResearchGate, we need a strong match:
            # - If name has 2+ significant parts, we need at least 2 matches
            # - If name has only 1 significant part, we need that 1 match
            # - The profile should not contain names that don't belong to the researcher
            min_matches = min(2, len(significant_parts))
            
            # Check for foreign names in the profile
            profile_words = set(re.findall(r'[a-z]+', rg_profile_clean))
            researcher_words = set(significant_parts)
            foreign_words = profile_words - researcher_words - {'profile', 'research', 'gate'}
            
            # If there are significant foreign words (potential other person's name), be cautious
            suspicious_foreign = any(len(word) > 4 for word in foreign_words)
            
            if matching_parts < min_matches or suspicious_foreign:
                if self.verbose >= 2:
                    reason = f"insufficient match ({matching_parts}/{min_matches} parts)"
                    if suspicious_foreign:
                        reason += f", suspicious foreign words: {foreign_words}"
                    print(f"[WebSearch] Rejecting ResearchGate '{profiles['researchgate']}' for '{researcher_name}' - {reason}")
                verified_profiles['researchgate'] = None
        
        # Check LinkedIn profiles - require exact name match in profile ID
        if profiles.get('linkedin'):
            linkedin_profile = profiles['linkedin'].lower()
            matching_parts = sum(1 for part in significant_parts if part in linkedin_profile)
            
            # LinkedIn should have at least first or last name
            if matching_parts < 1:
                if self.verbose >= 2:
                    print(f"[WebSearch] Rejecting LinkedIn '{profiles['linkedin']}' for '{researcher_name}' - no name match")
                verified_profiles['linkedin'] = None
        
        # ORCID and Google Scholar IDs don't contain names, so we can't verify them this way
        # They should have been validated by context during extraction
        
        # For "found" values (like Google Scholar), keep them as they indicate presence without ID
        
        return verified_profiles
    
    def _save_llm_cache(self):
        """Save LLM cache to file."""
        try:
            with open(self.llm_cache_file, 'w') as f:
                json.dump(self.llm_cache, f, indent=2)
        except Exception as e:
            if self.verbose >= 2:
                print(f"[WebSearch] Failed to save LLM cache: {e}")