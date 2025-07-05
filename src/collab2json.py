# -*- coding: utf-8 -*-
"""
Enhanced script to extract collaborator information from LaTeX files using LLMs and web search.

This script parses collaborator data from LaTeX files, uses Gemini LLM for intelligent
extraction, geocodes locations, searches for researcher profiles, and saves enriched data as JSON.

Author: Dr. Denys Dutykh (Khalifa University of Science and Technology, Abu Dhabi, UAE)
Date: 2025-01-04
"""

import re
import json
import time
import argparse
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Add parent directory to path for module imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

# Import our modules
from modules.config_manager import ConfigManager
from modules.llm_provider import LLMProviderManager
from modules.gemini_client import GeminiCollaboratorParser, UnifiedCollaboratorParser
from modules.web_search import ResearcherWebSearch
from modules.simple_search import SimpleResearcherSearch
from modules.display_utils import TerminalDisplay, Icons

# For basic geocoding (keeping original functionality)
import requests


# --- Configuration ---
BASE_DIR = Path(__file__).parent.parent.resolve()  # Parent of src/
INPUT_DIR = BASE_DIR / "input_data"
OUTPUT_DIR = BASE_DIR / "output_data"

# Cache files
CACHE_DIR = BASE_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)
GEOCODE_CACHE_FILE = CACHE_DIR / ".geocode_cache.json"
LLM_CACHE_FILE = CACHE_DIR / ".llm_cache.json"
SEARCH_CACHE_FILE = CACHE_DIR / ".search_cache.json"


class EnhancedCollaboratorExtractor:
    """Enhanced collaborator extractor with LLM and web search capabilities."""
    
    def __init__(self, config_file: Optional[Path] = None, verbose: int = 2, use_cache: bool = True):
        """Initialize the enhanced extractor."""
        self.verbose = verbose
        self.display = TerminalDisplay(verbose=verbose)
        self.use_cache = use_cache
        
        # Load configuration
        self.config = ConfigManager(config_file, verbose=verbose)
        
        # Initialize LLM provider
        self.llm_manager = LLMProviderManager(self.config)
        
        # Initialize unified parser with automatic fallback
        self.gemini_parser = None
        try:
            # Use UnifiedCollaboratorParser for automatic LLM fallback
            self.gemini_parser = UnifiedCollaboratorParser(
                self.config,
                verbose=verbose
            )
            if verbose >= 2:
                self.display.info("LLM Parser", "Unified parser with automatic fallback", Icons.SPARKLES)
        except RuntimeError as e:
            if verbose >= 1:
                self.display.warning(f"Failed to initialize LLM parser: {e}")
            # Fall back to Gemini-only parser if available
            google_api_key = self.config.get_api_key('google')
            if google_api_key:
                gemini_model = self.config.get('llm_config.primary_model', 'gemini-2.5-flash')
                self.gemini_parser = GeminiCollaboratorParser(
                    google_api_key,
                    model=gemini_model,
                    verbose=verbose
                )
        
        # Initialize web search
        self.web_search = None
        self.simple_search = None
        google_api_key = self.config.get_api_key('google')
        if google_api_key:
            cx = self.config.get('api_keys.google_cx')
            if cx and cx.strip():
                # Use Google Custom Search if CX is configured
                gemini_model = self.config.get('llm_config.primary_model', 'gemini-2.5-flash')
                self.web_search = ResearcherWebSearch(
                    google_api_key,
                    cx=cx,
                    verbose=verbose,
                    use_llm_extraction=True,
                    model=gemini_model,
                    config=self.config
                )
                if verbose >= 2:
                    self.display.info("Search engine", "Google Custom Search API", Icons.SEARCH)
            else:
                # Fall back to simple search
                self.simple_search = SimpleResearcherSearch(verbose=verbose)
                if verbose >= 1:
                    self.display.warning("Google Custom Search Engine ID not configured. Using simple search.")
                    self.display.info("To enable full search", "Set 'google_cx' in config.json", Icons.INFO)
        else:
            # Use simple search as fallback
            self.simple_search = SimpleResearcherSearch(verbose=verbose)
            if verbose >= 2:
                self.display.info("Search engine", "Simple direct search (limited)", Icons.SEARCH)
        
        # Load caches
        self.geocode_cache = self._load_cache(GEOCODE_CACHE_FILE)
        self.llm_cache = self._load_cache(LLM_CACHE_FILE)
        self.search_cache = self._load_cache(SEARCH_CACHE_FILE)
    
    def _load_cache(self, cache_file: Path) -> Dict:
        """Load cache from file."""
        if not self.use_cache:
            return {}
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError, OSError):
                return {}
        return {}
    
    def _save_cache(self, cache_file: Path, cache_data: Dict):
        """Save cache to file."""
        if not self.use_cache:
            return
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
    
    def _save_caches(self):
        """Save all caches to their respective files."""
        self._save_cache(GEOCODE_CACHE_FILE, self.geocode_cache)
        self._save_cache(LLM_CACHE_FILE, self.llm_cache)
        self._save_cache(SEARCH_CACHE_FILE, self.search_cache)
    
    def clean_latex_string(self, text: str) -> str:
        """Clean LaTeX commands from text."""
        # First, handle textsc which should be done before other replacements
        text = re.sub(r'\\textsc\{([^}]*)\}', r'\1', text)
        
        # Handle accented characters
        replacements = {
            "\\v{s}": "š", "\\v{S}": "Š", "\\v{c}": "č", "\\v{C}": "Č",
            "\\v{z}": "ž", "\\v{Z}": "Ž", "\\c{c}": "ç", "\\c{C}": "Ç",
            "\\'e": "é", "\\`e": "è", "\\^e": "ê", '\\"e': "ë",
            "\\'a": "á", "\\`a": "à", "\\^a": "â", '\\"a': "ä",
            "\\'i": "í", "\\`i": "ì", "\\^i": "î", '\\"i': "ï",
            "\\'o": "ó", "\\`o": "ò", "\\^o": "ô", '\\"o': "ö",
            "\\'u": "ú", "\\`u": "ù", "\\^u": "û", '\\"u': "ü",
            "\\o": "ø", "\\O": "Ø",
            "\\&": "&", "--": "-", "~": " ",
        }
        
        for latex, unicode_char in replacements.items():
            text = text.replace(latex, unicode_char)
            text = text.replace("{" + latex + "}", unicode_char)
        
        # Clean up any remaining backslashes that aren't part of a command
        # This handles cases like \Ng, \Hinton, etc.
        text = re.sub(r'\\([A-Za-z]+)(?![{a-zA-Z])', r'\1', text)
        
        # Clean up braces
        text = text.replace("{", "").replace("}", "")
        
        return text
    
    def parse_latex_entry_basic(self, entry_match) -> Dict[str, Any]:
        """Basic parsing without LLM (fallback)."""
        full_name_raw, rest = entry_match.groups()
        
        # Clean the full name
        full_name_clean = re.sub(r'\\textsc\{(.*?)\}', r'\1', full_name_raw)
        full_name_clean = self.clean_latex_string(full_name_clean).strip()
        
        # Split name
        name_parts = full_name_clean.split()
        first, last = "", ""
        if len(name_parts) > 1:
            first = " ".join(name_parts[:-1])
            last = name_parts[-1]
        else:
            last = full_name_clean
        
        # Clean rest
        rest = self.clean_latex_string(rest)
        rest = re.sub(r"\(formerly at\) ?", "", rest, flags=re.IGNORECASE).strip()
        
        # Extract basic info
        country, city, university, affiliation = "", "", "", ""
        
        # Known countries list
        known_countries = [
            "Algeria", "Australia", "Austria", "Belgium", "Brazil", "Cameroon", 
            "Canada", "Chile", "China", "Cyprus", "Czech Republic", "Denmark", 
            "Estonia", "Finland", "France", "Germany", "Greece", "Hong Kong", 
            "India", "Indonesia", "Iran", "Ireland", "Italy", "Japan", "Kazakhstan", 
            "Lebanon", "Malaysia", "Morocco", "Netherlands", "New Zealand", "Norway", 
            "Pakistan", "Poland", "Portugal", "Qatar", "Romania", "Russia", 
            "Saudi Arabia", "Singapore", "Slovak Republic", "South Korea", "Spain", 
            "Sweden", "Switzerland", "Taiwan", "Thailand", "Tunisia", "Turkey", 
            "UAE", "UK", "Ukraine", "United Arab Emirates", "United Kingdom", 
            "USA", "Vietnam"
        ]
        
        # Country normalization
        country_map = {
            "UAE": "United Arab Emirates",
            "UK": "United Kingdom",
            "USA": "United States of America"
        }
        
        # Extract country
        for c in sorted(known_countries, key=len, reverse=True):
            if re.search(r'\b' + re.escape(c) + r'\b', rest, re.IGNORECASE):
                country = c
                # Only remove the last occurrence of country to preserve university names like "University of Norway"
                # First check if country appears at the end (possibly with trailing punctuation)
                country_at_end_pattern = re.compile(r',?\s*\b' + re.escape(c) + r'\b\.?\s*$', re.IGNORECASE)
                if country_at_end_pattern.search(rest):
                    rest = country_at_end_pattern.sub('', rest).strip()
                else:
                    # If not at end, might be embedded, so be more careful
                    # Look for pattern like ", Country." or ", Country" at the end
                    parts = rest.rsplit(',', 1)
                    if len(parts) == 2 and c.lower() in parts[1].lower():
                        rest = parts[0].strip()
                break
        
        country = country_map.get(country, country)
        
        # Extract city and affiliation
        parts = [p.strip() for p in rest.split(',') if p.strip()]
        
        # Special handling for Malaysian university format
        # "Universiti Malaysia Terengganu, Terengganu, Malaysia" 
        # where "Terengganu" is both in university name and is the state/city
        if country == "Malaysia" and parts:
            # Look for Malaysian state names that might be cities
            malaysian_states = ['terengganu', 'kelantan', 'pahang', 'selangor', 'perak', 
                               'kedah', 'perlis', 'penang', 'johor', 'melaka', 'sabah', 'sarawak']
            for i, part in enumerate(parts):
                if part.lower() in malaysian_states and i > 0:  # Not the first part (which is affiliation)
                    city = part
                    parts.pop(i)
                    break
        
        if not city and parts:
            # Look for city - it's often the second-to-last or last part before country
            # Check if last part is a region/state indicator
            if len(parts) >= 2:
                last_part = parts[-1].lower()
                second_last = parts[-2]
                
                # If last part contains "region", "state", "province", it's likely not the city
                if any(keyword in last_part for keyword in ['region', 'state', 'province', 'oblast', 'prefecture']):
                    # Use second-to-last as city if it's not a university/institute
                    if ("university" not in second_last.lower() and 
                        "institute" not in second_last.lower() and
                        len(second_last.split()) < 4):
                        city = second_last
                        parts.pop(-2)  # Remove the city, keep the region
                else:
                    # Use last part as city
                    city_candidate = parts[-1]
                    if ("university" not in city_candidate.lower() and 
                        "institute" not in city_candidate.lower() and
                        len(city_candidate.split()) < 4):
                        city = city_candidate
                        parts.pop(-1)
            else:
                # Only one part left, check if it could be a city
                city_candidate = parts[-1] if parts else ""
                if (city_candidate and
                    "university" not in city_candidate.lower() and 
                    "institute" not in city_candidate.lower() and
                    len(city_candidate.split()) < 4):
                    city = city_candidate
                    parts.pop(-1)
        
        # Affiliation is usually first part
        if parts:
            affiliation = parts[0]
        
        # Find university
        for p in parts:
            if any(keyword in p.lower() for keyword in 
                   ["university", "universit", "polytech", "école", "institute", "institut"]):
                university = p
                break
        
        if not university and affiliation:
            university = affiliation
        
        # Capitalize city name properly
        if city:
            # Handle multi-word cities and special cases
            city_words = city.split()
            capitalized_words = []
            for word in city_words:
                # Keep acronyms and special patterns
                if word.isupper() and len(word) > 1:
                    capitalized_words.append(word)
                # Handle hyphenated words (e.g., Saint-Martin-d'Hères)
                elif '-' in word:
                    parts = word.split('-')
                    capitalized_parts = [p.capitalize() if p and not (p.startswith("d'") or p.startswith("l'")) else p for p in parts]
                    capitalized_words.append('-'.join(capitalized_parts))
                # Handle apostrophe words (e.g., d'Hères)
                elif "'" in word and len(word) > 2:
                    if word.startswith("d'") or word.startswith("l'"):
                        capitalized_words.append(word)
                    else:
                        capitalized_words.append(word.capitalize())
                else:
                    capitalized_words.append(word.capitalize())
            city = ' '.join(capitalized_words)
        
        return {
            'firstName': first,
            'lastName': last,
            'fullName': full_name_clean,
            'affiliation': affiliation,
            'university': university,
            'city': city,
            'country': country,
            'llm_enhanced': False
        }
    
    def fix_university_field(self, collaborator: Dict[str, Any]) -> None:
        """
        Fix cases where university field contains department name instead of university.
        
        Args:
            collaborator: Collaborator data dictionary (modified in place)
        """
        university = collaborator.get('university', '')
        affiliation = collaborator.get('affiliation', '')
        
        # Check if university field contains department/faculty/school
        if university and any(word in university.lower() for word in ['department', 'faculty', 'school of', 'institute of']):
            # Try to find real university name in the full affiliation
            full_text = f"{affiliation} {university}".strip()
            
            # Look for university patterns
            university_patterns = [
                r'(Universit[yéi]\s+[^,]+?)(?=,|\s+Faculty|\s+Department|\s+School|$)',  # University/Université/Universiti
                r'(Institut[eo]?\s+[^,]+?)(?=,|\s+Faculty|\s+Department|$)',  # Institute/Instituto
                r'(College\s+[^,]+?)(?=,|\s+Faculty|\s+Department|$)',
                r'(Academia\s+[^,]+?)(?=,|\s+Faculty|\s+Department|$)',
                r'(École\s+[^,]+?)(?=,|\s+Faculty|\s+Department|$)',
                r'(Hochschule\s+[^,]+?)(?=,|\s+Faculty|\s+Department|$)',  # German
                r'(Universidad\s+[^,]+?)(?=,|\s+Faculty|\s+Department|$)',  # Spanish
                r'(Rechenzentrum\s+[^,]+?)(?=,|\s+Faculty|\s+Department|$)',  # German Computing Center
            ]
            
            found_university = None
            for pattern in university_patterns:
                match = re.search(pattern, full_text, re.IGNORECASE)
                if match:
                    potential_university = match.group(1).strip()
                    # Clean up any trailing words that shouldn't be part of university name
                    potential_university = re.sub(r'\s+(Civil|Mechanical|Electrical|Chemical|Computer)\s*$', '', potential_university, flags=re.IGNORECASE)
                    # Make sure it's not just "University" or "Institute"
                    if len(potential_university.split()) > 1:
                        found_university = potential_university
                        break
            
            if found_university:
                # Move current "university" to department if not already there
                if not collaborator.get('department'):
                    collaborator['department'] = university
                # Set the real university
                collaborator['university'] = found_university
                
                if self.verbose >= 3:
                    print(f"         🔧 Fixed university field: '{university}' → '{found_university}'")
    
    def flatten_gemini_result(self, gemini_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert Gemini's nested structure to flat structure expected by the rest of the code.
        
        Gemini returns nested structure like:
        {
            'nameInformation': {'firstName': 'John', 'lastName': 'Doe'},
            'affiliationDetails': {'affiliation': 'MIT', ...},
            ...
        }
        
        This function flattens it to:
        {
            'firstName': 'John',
            'lastName': 'Doe',
            'affiliation': 'MIT',
            ...
        }
        """
        if not gemini_result:
            return gemini_result
        
        flattened = {}
        
        # Extract from nameInformation
        if 'nameInformation' in gemini_result:
            name_info = gemini_result['nameInformation']
            flattened['firstName'] = name_info.get('firstName', '')
            flattened['lastName'] = name_info.get('lastName', '')
            flattened['nameVariants'] = name_info.get('nameVariants', [])
        
        # Extract from affiliationDetails
        if 'affiliationDetails' in gemini_result:
            aff_info = gemini_result['affiliationDetails']
            flattened['affiliation'] = aff_info.get('affiliation', '')
            flattened['university'] = aff_info.get('university', '')
            flattened['department'] = aff_info.get('department', '')
            flattened['institutionType'] = aff_info.get('institutionType', '')
            flattened['researchGroup'] = aff_info.get('researchGroup', '')
        
        # Extract from locationInformation
        if 'locationInformation' in gemini_result:
            loc_info = gemini_result['locationInformation']
            # Only update if the value is not empty
            if loc_info.get('city'):
                flattened['city'] = loc_info.get('city')
            if loc_info.get('country'):
                flattened['country'] = loc_info.get('country')
            if loc_info.get('alternateLocations'):
                flattened['alternateLocations'] = loc_info.get('alternateLocations')
        
        # Extract from enhancedLocationData
        if 'enhancedLocationData' in gemini_result:
            loc_data = gemini_result['enhancedLocationData']
            flattened['improvedLocationString'] = loc_data.get('improvedLocationString', '')
            flattened['locationConfidence'] = loc_data.get('confidenceScore', 0)
        
        # Extract from additionalContext
        if 'additionalContext' in gemini_result:
            context = gemini_result['additionalContext']
            flattened['isFormerAffiliation'] = context.get('isFormerAffiliation', False)
            flattened['academicTitle'] = context.get('academicTitle', '')
            flattened['specialNotes'] = context.get('specialNotes', '')
        
        # Keep any top-level fields that aren't nested
        for key, value in gemini_result.items():
            if key not in ['nameInformation', 'affiliationDetails', 'locationInformation', 
                          'enhancedLocationData', 'additionalContext'] and key not in flattened:
                flattened[key] = value
        
        # Ensure llm_enhanced flag is set
        flattened['llm_enhanced'] = True
        
        return flattened
    
    def parse_latex_entry_with_llm(self, latex_entry: str) -> Optional[Dict[str, Any]]:
        """Parse entry using LLM for better extraction."""
        if not self.gemini_parser:
            return None
        
        # Check cache
        entry_hash = hash(latex_entry)
        if entry_hash in self.llm_cache:
            if self.verbose >= 3:
                print("[LLM] Cache hit for entry")
            return self.llm_cache[entry_hash]
        
        # Parse with Gemini
        try:
            parsed = self.gemini_parser.parse_collaborator_entry(latex_entry)
            if parsed:
                # Flatten the nested structure first
                flattened = self.flatten_gemini_result(parsed)
                
                # Enhance location (using original nested structure)
                location_enhanced = self.gemini_parser.enhance_location(parsed)
                if location_enhanced.get('enhanced_location'):
                    flattened['enhanced_location'] = location_enhanced['enhanced_location']
                    flattened['location_confidence'] = location_enhanced.get('confidence', 0.5)
                
                # Get research profile estimate
                if flattened.get('firstName') and flattened.get('lastName'):
                    name = f"{flattened['firstName']} {flattened['lastName']}"
                    profile = self.gemini_parser.extract_research_profile(
                        name, 
                        flattened.get('affiliation', '')
                    )
                    if profile:
                        flattened['estimated_research_profile'] = profile
                
                flattened['llm_enhanced'] = True
                
                # Cache the flattened result
                self.llm_cache[entry_hash] = flattened
                
                return flattened
                
        except Exception as e:
            if self.verbose >= 2:
                print(f"[LLM] Error parsing entry: {str(e)}")
        
        return None
    
    def search_university_location(self, university_name: str, country: str = None) -> Optional[Dict[str, str]]:
        """
        Search for precise university location using web search and LLM extraction.
        
        Args:
            university_name: Name of the university
            country: Country name (optional, helps narrow search)
            
        Returns:
            Dict with city, state/region, and full address if found
        """
        if not university_name or not self.web_search:
            return None
        
        # Known problematic universities with hardcoded correct locations
        known_locations = {
            'université grenoble alpes': {
                'city': 'Saint-Martin-d\'Hères',
                'state': 'Auvergne-Rhône-Alpes',
                'country': 'France'
            },
            'universiti malaysia terengganu': {
                'city': 'Kuala Nerus',
                'state': 'Terengganu',
                'country': 'Malaysia'
            },
            'kazakh national university': {
                'city': 'Almaty',
                'state': None,
                'country': 'Kazakhstan'
            }
        }
        
        # Check if this is a known university
        uni_lower = university_name.lower()
        for known_uni, location in known_locations.items():
            if known_uni in uni_lower:
                if self.verbose >= 3:
                    print(f"[Location] Using known location for {university_name}")
                return location
        
        # Check cache first
        cache_key = f"univ_location:{university_name}:{country or ''}"
        if cache_key in self.search_cache:
            if self.verbose >= 3:
                print(f"[Location] Using cached location for {university_name}")
            return self.search_cache[cache_key]
        
        # Create search query
        query = f'"{university_name}" location address city'
        if country:
            query += f' {country}'
        
        try:
            if self.verbose >= 3:
                print(f"[Location] Searching for university location: {university_name}")
            
            # Perform search
            search_results = self.web_search._perform_search(query)
            
            if not search_results:
                return None
            
            # Use LLM extraction if available
            if self.gemini_parser and self.config.get('search_settings.use_llm_extraction', True):
                # Combine search results for LLM analysis
                combined_text = ""
                for i, result in enumerate(search_results[:3], 1):
                    combined_text += f"Result {i}:\n"
                    combined_text += f"Title: {result.get('title', '')}\n"
                    combined_text += f"Snippet: {result.get('snippet', '')}\n\n"
                
                # Create LLM prompt for location extraction
                prompt = f"""Extract the precise location information for {university_name} from these search results.

{combined_text}

Please extract:
1. City name (the actual city where the university is located, not just "University City" or similar)
2. State or region (if applicable)
3. Full street address (if available)
4. Country (confirm or correct if needed)

Important:
- For "Université Grenoble Alpes", the actual city is "Saint-Martin-d'Hères", not "Grenoble"
- For "Universiti Malaysia Terengganu", extract the actual city name (e.g., "Kuala Nerus" or "Kuala Terengganu")
- Avoid extracting generic descriptions like "it is the third largest university"
- Focus on actual place names and addresses

Return ONLY a JSON object with these fields:
{{
    "city": "actual city name",
    "state": "state or region if applicable",
    "address": "full street address if found",
    "country": "country name",
    "confidence": 0.0-1.0
}}"""

                try:
                    # Import genai from gemini_client if available
                    from modules.gemini_client import genai, GEMINI_AVAILABLE
                    
                    if not GEMINI_AVAILABLE:
                        raise ImportError("Google Generative AI not available")
                    
                    generation_config = genai.GenerationConfig(
                        temperature=0.1,
                        max_output_tokens=512,
                    )
                    
                    response = self.gemini_parser.model.generate_content(
                        prompt,
                        generation_config=generation_config,
                        safety_settings=self.gemini_parser.safety_settings
                    )
                    
                    if response.candidates and response.candidates[0].content.parts:
                        # Extract JSON from response
                        json_text = response.text.strip()
                        json_text = re.sub(r'^```json\s*\n?', '', json_text)
                        json_text = re.sub(r'\n?```\s*$', '', json_text)
                        
                        location_data = json.loads(json_text)
                        
                        # Validate the extracted data
                        if location_data.get('city') and location_data.get('confidence', 0) >= 0.5:
                            # Filter out obviously wrong extractions
                            city = location_data['city']
                            # Add more validation patterns
                            bad_patterns = [
                                'it is', 'the university', 'located in', 'based in', 'campus',
                                'due to', 'because', 'since', 'n/a', 'none', 'unknown',
                                'department', 'faculty', 'school of'
                            ]
                            
                            # Check for reasonable city name
                            if (len(city) > 5 and  # Reasonable city name length
                                len(city) < 50 and  # Not too long
                                not any(phrase in city.lower() for phrase in bad_patterns) and
                                not city.lower().endswith(' to') and
                                not city.lower().endswith(' of') and
                                city.count(' ') < 5):  # Not a full sentence
                                
                                # Additional validation for address field
                                address = location_data.get('address', '')
                                if address and len(address) < 8:
                                    # Address too short, likely garbage data
                                    if self.verbose >= 3:
                                        print(f"[Location] Skipping cache due to suspicious address: '{address}'")
                                else:
                                    # Cache and return the result
                                    self.search_cache[cache_key] = location_data
                                    self._save_caches()
                                
                                if self.verbose >= 2:
                                    print(f"[Location] LLM extracted location: {location_data['city']}")
                                    if self.verbose >= 3:
                                        if location_data.get('state'):
                                            print(f"           State/Region: {location_data['state']}")
                                        if location_data.get('address'):
                                            print(f"           Full address: {location_data['address']}")
                                        print(f"           Confidence: {location_data.get('confidence', 'N/A')}")
                                
                                return location_data
                        else:
                            if self.verbose >= 3:
                                print(f"[Location] LLM extraction low confidence or invalid: {location_data}")
                                
                except Exception as e:
                    if self.verbose >= 2:
                        print(f"[Location] LLM extraction error: {e}")
                    # Fall back to regex-based extraction
            
            # Fall back to regex-based extraction if LLM not available or failed
            location_data = {
                'city': None,
                'state': None,
                'address': None,
                'country': country
            }
            
            # Look through first few results
            for result in search_results[:3]:
                snippet = result.get('snippet', '').lower()
                title = result.get('title', '').lower()
                
                # Look for address patterns
                # Try to find full address
                address_patterns = [
                    r'(?:address|located at|campus at)[:\s]+([^.]+)',
                    r'(\d+[^,]+,\s*\d{5}\s+[^,]+,\s*[^.]+)',  # Street number, postal code, city
                    r'(\d+[^,]+,\s*[^,]+,\s*[^.]+)',  # Street, city, country
                ]
                
                for pattern in address_patterns:
                    match = re.search(pattern, snippet, re.IGNORECASE)
                    if match:
                        location_data['address'] = match.group(1).strip()
                        
                        # Extract city from address
                        city_match = re.search(r'\d{5}\s+([^,]+)', location_data['address'])
                        if city_match:
                            location_data['city'] = city_match.group(1).strip()
                        else:
                            parts = location_data['address'].split(',')
                            if len(parts) >= 2:
                                location_data['city'] = parts[-2].strip()
                        break
                
                # If no address found, look for city mentions
                if not location_data['city']:
                    # Create more specific patterns based on the university name
                    university_words = university_name.split()
                    
                    # Special handling for Malaysian universities
                    if country and country.lower() == 'malaysia':
                        # Look for Malaysian postal codes (5 digits) followed by city name
                        malaysia_patterns = [
                            r'(\d{5})\s+([A-Za-z\s]+?)(?:,\s*(?:malaysia|' + university_name.lower() + '))',
                            r'(?:' + re.escape(university_name) + r')[^.]*?(?:in|at)\s+([A-Za-z\s]+?)(?:,\s*malaysia)',
                        ]
                        for pattern in malaysia_patterns:
                            match = re.search(pattern, snippet, re.IGNORECASE)
                            if match:
                                city_candidate = match.group(2 if '\\d{5}' in pattern else 1).strip()
                                # Validate it's not part of the university name
                                if not any(word.lower() in city_candidate.lower() for word in university_words):
                                    location_data['city'] = city_candidate
                                    break
                    
                    # General patterns - more restrictive
                    if not location_data['city']:
                        # Escape university name for use in regex
                        escaped_uni_name = re.escape(university_name)
                        
                        city_patterns = [
                            # Look for university name followed by location preposition and city
                            rf'{escaped_uni_name}[^.]*?(?:is located in|is in|campus in)\s+([A-Za-z\s\-\']+?)(?:[,.]|\s+(?:and|with|which))',
                            # Look for address patterns with city
                            r'(?:address|location):\s*[^,]+,\s*([A-Za-z\s\-\']+?)(?:,|\s+\d{5})',
                            # Look for "city of X" pattern
                            r'(?:city of|town of)\s+([A-Za-z\s\-\']+?)(?:[,.]|$)',
                        ]
                        
                        for pattern in city_patterns:
                            matches = re.finditer(pattern, snippet + ' ' + title, re.IGNORECASE)
                            for match in matches:
                                potential_city = match.group(1).strip()
                                
                                # Validate the extracted city
                                # Must be between 2 and 40 characters
                                if 2 < len(potential_city) < 40:
                                    # Must not be a country name
                                    country_names = ['france', 'malaysia', 'usa', 'uk', 'germany', 'india', 
                                                   'spain', 'italy', 'china', 'japan', 'australia']
                                    # Must not contain institution words
                                    institution_words = ['laboratory', 'observatory', 'institute', 'university',
                                                       'college', 'school', 'academy', 'research', 'center']
                                    # Must not be mostly numbers or special characters
                                    if (potential_city.lower() not in country_names and
                                        not any(word in potential_city.lower() for word in institution_words) and
                                        sum(c.isalpha() or c.isspace() or c in "-'" for c in potential_city) / len(potential_city) > 0.8):
                                        location_data['city'] = potential_city
                                        break
                            
                            if location_data['city']:
                                break
            
            # Cache the result if found
            if location_data['city']:
                # Capitalize the city name properly
                city = location_data['city']
                city_words = city.split()
                capitalized_words = []
                for word in city_words:
                    if word.isupper() and len(word) > 1:
                        capitalized_words.append(word)
                    elif '-' in word:
                        parts = word.split('-')
                        capitalized_parts = [p.capitalize() if p and not (p.startswith("d'") or p.startswith("l'")) else p for p in parts]
                        capitalized_words.append('-'.join(capitalized_parts))
                    elif "'" in word and len(word) > 2:
                        if word.startswith("d'") or word.startswith("l'"):
                            capitalized_words.append(word)
                        else:
                            capitalized_words.append(word.capitalize())
                    else:
                        capitalized_words.append(word.capitalize())
                location_data['city'] = ' '.join(capitalized_words)
                
                self.search_cache[cache_key] = location_data
                self._save_caches()
                
                if self.verbose >= 2:
                    print(f"[Location] Regex found location: {location_data['city']}")
                    if self.verbose >= 3 and location_data.get('address'):
                        print(f"           Full address: {location_data['address']}")
                
                return location_data
            
        except Exception as e:
            if self.verbose >= 2:
                print(f"[Location] Error searching university location: {e}")
        
        return None
    
    def geocode_location(self, location: str, skip: bool = False) -> Optional[Dict[str, float]]:
        """Geocode a location string."""
        if skip or not location:
            return None
        
        cache_key = location.lower()
        if cache_key in self.geocode_cache:
            return self.geocode_cache[cache_key]
        
        # Use OpenStreetMap Nominatim
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            "q": location,
            "format": "json",
            "limit": 1
        }
        
        try:
            response = requests.get(
                url, 
                params=params, 
                headers={"User-Agent": "collab2json-enhanced"}
            )
            response.raise_for_status()
            data = response.json()
            
            if data:
                coords = {
                    "lat": float(data[0]["lat"]), 
                    "lng": float(data[0]["lon"])
                }
                self.geocode_cache[cache_key] = coords
                return coords
                
        except Exception as e:
            if self.verbose >= 2:
                print(f"[Geocode] Error: {e}")
        
        self.geocode_cache[cache_key] = None
        return None
    
    async def search_researcher_info(self, name: str, affiliation: str, 
                                   country: str = None) -> Dict[str, Any]:
        """Search for researcher information online."""
        # Use appropriate search method
        if self.web_search:
            search_method = self.web_search
        elif self.simple_search:
            search_method = self.simple_search
        else:
            if self.verbose >= 3:
                self.display.warning(f"No search method available for {name}")
            return {}
        
        # Check cache
        cache_key = f"{name}|{affiliation}|{country or ''}"
        if cache_key in self.search_cache:
            if self.verbose >= 3:
                print(f"[Search] Cache hit for {name}")
            return self.search_cache[cache_key]
        
        # Perform search
        try:
            # Check if quality mode is enabled
            quality_mode = self.config.get('search_settings.quality_mode', True)
            
            # Use quality mode for web search if available
            if hasattr(search_method, 'search_researcher'):
                # Check if the method accepts quality_mode parameter
                import inspect
                sig = inspect.signature(search_method.search_researcher)
                if 'quality_mode' in sig.parameters:
                    results = search_method.search_researcher(name, affiliation, country, quality_mode=quality_mode)
                else:
                    results = search_method.search_researcher(name, affiliation, country)
            else:
                results = search_method.search_researcher(name, affiliation, country)
                
            self.search_cache[cache_key] = results
            return results
        except Exception as e:
            if self.verbose >= 2:
                print(f"[Search] Error searching for {name}: {str(e)}")
            return {}
    
    def normalize_name_for_matching(self, name: str) -> str:
        """
        Normalize a name for comparison by removing accents and converting to lowercase.
        
        Args:
            name: Name to normalize
            
        Returns:
            Normalized name
        """
        import unicodedata
        
        # Remove accents
        normalized = unicodedata.normalize('NFD', name)
        normalized = ''.join(char for char in normalized if unicodedata.category(char) != 'Mn')
        
        # Convert to lowercase
        normalized = normalized.lower()
        
        return normalized
    
    def verify_orcid(self, orcid_id: str, researcher_name: str) -> bool:
        """
        Verify ORCID ID belongs to the researcher by fetching data from ORCID API.
        
        Args:
            orcid_id: ORCID identifier (e.g., "0000-0000-0000-0000")
            researcher_name: Full name of the researcher
            
        Returns:
            True if ORCID belongs to researcher, False otherwise
        """
        # Check cache first
        cache_key = f"orcid_verify:{orcid_id}"
        if cache_key in self.search_cache:
            cached_result = self.search_cache[cache_key]
            if self.verbose >= 3:
                print(f"           Using cached ORCID verification: {cached_result}")
            return cached_result
        
        try:
            # ORCID public API URL
            url = f"https://pub.orcid.org/v3.0/{orcid_id}/person"
            headers = {
                'Accept': 'application/json',
                'User-Agent': 'collab2json-verifier'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract name from ORCID
                orcid_name = ""
                name_data = data.get('name', {})
                
                # Try to get full name
                given_names = name_data.get('given-names', {}).get('value', '')
                family_name = name_data.get('family-name', {}).get('value', '')
                
                if given_names and family_name:
                    orcid_name = f"{given_names} {family_name}"
                elif name_data.get('credit-name', {}).get('value'):
                    orcid_name = name_data['credit-name']['value']
                
                if self.verbose >= 3:
                    print(f"           ORCID name: '{orcid_name}'")
                
                # Compare names
                if orcid_name:
                    # Normalize both names for comparison
                    orcid_normalized = self.normalize_name_for_matching(orcid_name)
                    researcher_normalized = self.normalize_name_for_matching(researcher_name)
                    
                    # Split on spaces and hyphens
                    orcid_parts = set(re.split(r'[\s\-]+', orcid_normalized))
                    researcher_parts = set(re.split(r'[\s\-]+', researcher_normalized))
                    
                    # Remove common particles
                    particles = {'de', 'van', 'von', 'der', 'den', 'del', 'della', 'di', 'da', 'al', 'el'}
                    orcid_parts = {p for p in orcid_parts if p not in particles and len(p) > 1}
                    researcher_parts = {p for p in researcher_parts if p not in particles and len(p) > 1}
                    
                    # Check overlap - at least 2 parts should match, or all parts if less than 2
                    common_parts = orcid_parts & researcher_parts
                    min_matches = min(2, min(len(orcid_parts), len(researcher_parts)))
                    
                    is_valid = len(common_parts) >= min_matches
                    
                    # Cache result
                    self.search_cache[cache_key] = is_valid
                    self._save_caches()
                    
                    if self.verbose >= 3:
                        print(f"           Name match: {len(common_parts)}/{min_matches} parts")
                    
                    return is_valid
                else:
                    # No name in ORCID profile - can't verify
                    if self.verbose >= 3:
                        print("           No name found in ORCID profile")
                    return False
                    
            elif response.status_code == 404:
                if self.verbose >= 3:
                    print(f"           ORCID {orcid_id} not found")
                # Cache negative result
                self.search_cache[cache_key] = False
                self._save_caches()
                return False
            else:
                if self.verbose >= 3:
                    print(f"           ORCID API returned status {response.status_code}")
                return True  # Don't remove on API errors
                
        except Exception as e:
            if self.verbose >= 2:
                print(f"         ⚠️  ORCID verification error: {e}")
            # Don't remove on errors - give benefit of doubt
            return True
    
    def verify_researcher_profiles(self, collaborator: Dict[str, Any]) -> Dict[str, Any]:
        """
        Re-verify existing researcher profiles using strict validation rules.
        
        Args:
            collaborator: Collaborator data with existing profiles
            
        Returns:
            Updated collaborator data with verified profiles
        """
        if self.verbose >= 3:
            name = f"{collaborator.get('firstName', '')} {collaborator.get('lastName', '')}"
            self.display.info("Profile Verification", f"Verifying profiles for {name}", Icons.SEARCH)
        
        # Extract researcher name
        full_name = collaborator.get('fullName', '')
        if not full_name:
            first = collaborator.get('firstName', '')
            last = collaborator.get('lastName', '')
            full_name = f"{first} {last}".strip()
        
        if not full_name:
            return collaborator
        
        # Track what was removed
        removed_items = []
        
        # 1. Verify email belongs to researcher
        if collaborator.get('email'):
            email = collaborator['email']
            # Use the same validation logic from web_search module
            email_username = email.split('@')[0].lower()
            email_username_no_numbers = re.sub(r'\d+', '', email_username)
            
            # Parse researcher name - handle hyphenated names
            normalized_name = self.normalize_name_for_matching(full_name)
            # Split on both spaces and hyphens to handle names like "Emma-Imen"
            name_parts = re.split(r'[\s\-]+', normalized_name)
            name_parts = [p for p in name_parts if p not in ['bin', 'binti', 'ibn', 'bint', 'al', 'el', 'de', 'van', 'von'] and len(p) > 1]
            
            # Check if email matches researcher name
            valid_email = False
            if name_parts:
                first_name = name_parts[0] if name_parts else ""
                last_name = name_parts[-1] if len(name_parts) > 1 else ""
                
                # Check various patterns
                valid_patterns = []
                if first_name and last_name:
                    valid_patterns.extend([
                        first_name, last_name,
                        f"{first_name}.{last_name}",
                        f"{first_name}_{last_name}",
                        f"{first_name[0]}{last_name}",
                        f"{last_name}.{first_name}"
                    ])
                
                for pattern in valid_patterns:
                    if pattern and (pattern == email_username or pattern in email_username or 
                                   pattern == email_username_no_numbers or pattern in email_username_no_numbers):
                        valid_email = True
                        break
            
            if not valid_email:
                removed_items.append(f"email: {email}")
                collaborator['email'] = None
                if self.verbose >= 2:
                    print(f"         ❌ Removed email '{email}' - doesn't match researcher name")
        
        # 2. Verify profiles
        profiles = collaborator.get('profiles', {})
        if profiles:
            verified_profiles = {}
            
            # Parse name for profile validation - handle hyphenated names and accents
            normalized_name = self.normalize_name_for_matching(full_name)
            # Split on both spaces and hyphens
            name_parts = re.split(r'[\s\-]+', normalized_name)
            name_parts = [p for p in name_parts if p not in ['bin', 'binti', 'ibn', 'bint', 'al', 'el', 'de', 'van', 'von'] and len(p) > 1]
            
            # Keep all parts as significant (don't filter by length)
            significant_parts = name_parts
            
            for profile_type, profile_id in profiles.items():
                if not profile_id or profile_id == 'found':
                    # Keep 'found' values as they indicate presence
                    verified_profiles[profile_type] = profile_id
                    continue
                
                # Verify ResearchGate profiles
                if profile_type == 'researchgate':
                    # Normalize profile ID for comparison
                    profile_id_normalized = self.normalize_name_for_matching(profile_id)
                    # Remove trailing numbers
                    profile_id_clean = re.sub(r'[-_]\d+$', '', profile_id_normalized)
                    
                    # Count matching name parts
                    matching_parts = sum(1 for part in significant_parts if part in profile_id_clean)
                    
                    # For ResearchGate, we need at least 1 name part to match
                    # For single-part names, that one part must match
                    # For multi-part names, at least one significant part must match
                    min_matches = 1
                    
                    if matching_parts >= min_matches:
                        verified_profiles[profile_type] = profile_id
                        if self.verbose >= 3:
                            print(f"         ✓ Kept ResearchGate '{profile_id}' - {matching_parts} name parts match")
                    else:
                        removed_items.append(f"ResearchGate: {profile_id}")
                        if self.verbose >= 2:
                            print(f"         ❌ Removed ResearchGate '{profile_id}' - no name parts match")
                
                # Verify LinkedIn profiles
                elif profile_type == 'linkedin':
                    # Normalize for comparison
                    profile_id_normalized = self.normalize_name_for_matching(profile_id)
                    
                    # Check if any name part appears in the LinkedIn ID
                    # Also check for partial matches (like "myl" for "mylene")
                    matching_parts = 0
                    for part in significant_parts:
                        if part in profile_id_normalized:
                            matching_parts += 1
                        elif len(part) >= 3:
                            # Check if the profile ID contains the first 3+ letters of the name part
                            if profile_id_normalized.startswith(part[:3]) or part[:3] in profile_id_normalized:
                                matching_parts += 0.5  # Partial match
                    
                    if matching_parts >= 0.5:  # Accept partial matches
                        verified_profiles[profile_type] = profile_id
                        if self.verbose >= 3:
                            print(f"         ✓ Kept LinkedIn '{profile_id}' - name match found")
                    else:
                        # Only remove if it's clearly wrong (like "adama-compaore" for "Emma-Imen Turki")
                        removed_items.append(f"LinkedIn: {profile_id}")
                        if self.verbose >= 2:
                            print(f"         ❌ Removed LinkedIn '{profile_id}' - no name match")
                
                # Verify ORCID by fetching data from ORCID API
                elif profile_type == 'orcid':
                    if self.verbose >= 3:
                        print(f"         🔍 Verifying ORCID {profile_id}")
                    
                    # Fetch ORCID data
                    orcid_valid = self.verify_orcid(profile_id, full_name)
                    
                    if orcid_valid:
                        verified_profiles[profile_type] = profile_id
                    else:
                        removed_items.append(f"ORCID: {profile_id}")
                        if self.verbose >= 2:
                            print(f"         ❌ Removed ORCID '{profile_id}' - name mismatch")
                
                # Keep other IDs (Google Scholar, etc.)
                else:
                    verified_profiles[profile_type] = profile_id
            
            collaborator['profiles'] = verified_profiles
        
        # 3. Verify homepage is not a profile site
        if collaborator.get('homepage'):
            homepage = collaborator['homepage'].lower()
            profile_sites = [
                'researchgate.net', 'orcid.org', 'scholar.google',
                'linkedin.com', 'github.com', 'twitter.com', 'x.com',
                'loop.frontiersin.org', 'scopus.com', 'academia.edu'
            ]
            
            is_profile_site = any(site in homepage for site in profile_sites)
            if is_profile_site:
                # Check if we have the corresponding profile
                for site, profile_key in [
                    ('researchgate.net', 'researchgate'),
                    ('orcid.org', 'orcid'),
                    ('scholar.google', 'google_scholar'),
                    ('linkedin.com', 'linkedin')
                ]:
                    if site in homepage:
                        # If we don't have this profile or it was removed, remove homepage too
                        if not collaborator.get('profiles', {}).get(profile_key):
                            removed_items.append(f"homepage: {collaborator['homepage']}")
                            collaborator['homepage'] = None
                            if self.verbose >= 2:
                                print(f"         ❌ Removed homepage from {site} - profile was invalid")
                        break
        
        # Add verification metadata
        if removed_items:
            collaborator['verification_removed'] = removed_items
            collaborator['verified_date'] = time.strftime('%Y-%m-%d')
        
        return collaborator

    async def process_collaborators(self, tex_content: str, 
                                  skip_geocoding: bool = False,
                                  skip_search: bool = False,
                                  verify_only: bool = False,
                                  existing_data: Optional[List[Dict]] = None) -> Tuple[List[Dict], List[Dict]]:
        """Process all collaborators from LaTeX content."""
        
        # If verify_only mode, work with existing data
        if verify_only and existing_data:
            self.display.section("Verifying Existing Profiles", Icons.SEARCH)
            self.display.info("Total profiles to verify", len(existing_data), Icons.INFO)
            
            verified_results = []
            removed_count = 0
            
            for idx, collaborator in enumerate(existing_data, 1):
                name = f"{collaborator.get('firstName', '')} {collaborator.get('lastName', '')}"
                
                if self.verbose >= 2:
                    print(f"\n[Verify] Processing #{idx}/{len(existing_data)}: {name}")
                
                # Create a copy to avoid modifying original
                verified_collab = collaborator.copy()
                
                # Verify profiles
                verified_collab = self.verify_researcher_profiles(verified_collab)
                
                # Count removed items
                if verified_collab.get('verification_removed'):
                    removed_count += len(verified_collab['verification_removed'])
                
                verified_results.append(verified_collab)
                
                # Progress display
                if self.verbose >= 1 and self.verbose < 2:
                    self.display.progress(idx, len(existing_data), "Verifying")
            
            # Summary
            self.display.section("Verification Summary", Icons.CHECK)
            self.display.info("Profiles verified", len(verified_results), Icons.PERSON)
            if removed_count > 0:
                self.display.warning(f"Items removed during verification: {removed_count}")
            
            return verified_results, []
        
        # Normal processing mode
        self.display.section("Extracting Collaborators", Icons.PERSON)
        
        # Find all entries
        pattern = re.compile(r"^\s*\\item\[(.*?):\]\s*(.*)$", re.MULTILINE)
        matches = list(pattern.finditer(tex_content))
        
        self.display.info("Total collaborators found", len(matches), Icons.SPARKLES)
        
        results = []
        failed_geocode = []
        
        # Skip batch parsing due to safety filter issues - go directly to individual processing
        # Batch processing often triggers Gemini safety filters with multiple entries
        batch_results = None
        
        # Process individually for better reliability
        if True:  # Always use individual processing
            self.display.subsection("Individual Processing with LLM")
            for idx, match in enumerate(matches, 1):
                # Show progress at start of each entry
                if self.verbose >= 2:
                    print(f"\n[Gemini] Processing entry #{idx}/{len(matches)}...")
                
                # Extract basic info first for display
                basic_result = self.parse_latex_entry_basic(match)
                
                # Try LLM parsing first
                llm_result = None
                if self.gemini_parser:
                    if self.verbose >= 3:
                        self.display.llm_processing_detail(
                            f"Parsing entry #{idx} with Gemini",
                            basic_result.get('fullName', 'Unknown')
                        )
                    llm_result = self.parse_latex_entry_with_llm(match.group(0))
                    
                    # Show immediate feedback on success/failure
                    if llm_result and self.verbose >= 2:
                        extracted_name = f"{llm_result.get('firstName', '')} {llm_result.get('lastName', '')}"
                        print(f"         ✓ Extracted: {extracted_name.strip()}")
                        if self.verbose >= 3:
                            print(f"         Affiliation: {llm_result.get('affiliation', 'N/A')}")
                            location = f"{llm_result.get('city', '')}, {llm_result.get('country', '')}".strip(', ')
                            if location:
                                print(f"         Location: {location}")
                    elif self.verbose >= 2:
                        print("         ✗ Failed to parse with LLM, using basic parsing")
                
                # Merge results
                if llm_result:
                    # LLM result might have empty fields, so merge carefully
                    collaborator = {
                        'id': idx,
                        **basic_result,  # Start with basic result
                    }
                    # Update with non-empty LLM fields
                    for key, value in llm_result.items():
                        if value and (key not in collaborator or not collaborator[key]):
                            collaborator[key] = value
                else:
                    collaborator = {
                        'id': idx,
                        **basic_result
                    }
                
                # Fix university field if it contains department instead of university
                self.fix_university_field(collaborator)
                
                # Enhance location with university search if needed
                # Only search if city is truly missing or invalid
                city = collaborator.get('city') or ''
                city = city.strip() if city else ''
                needs_university_search = False
                
                if collaborator.get('university'):
                    # Check if we need to search for university location
                    if not city or city.lower() in ['none', '', 'n/a']:
                        needs_university_search = True
                    # Check for special cases where city needs verification
                    elif 'grenoble' in city.lower() and 'grenoble' in collaborator.get('university', '').lower():
                        needs_university_search = True
                    # Check if city is actually a university name or contains "universit"
                    elif 'universit' in city.lower() and len(city.split()) > 1:
                        needs_university_search = True
                    
                    if needs_university_search:
                        if self.verbose >= 3:
                            print(f"         🔍 Searching for precise location of {collaborator['university']}")
                        
                        university_location = self.search_university_location(
                            collaborator['university'],
                            collaborator.get('country')
                        )
                        
                        if university_location and university_location.get('city'):
                            # Validate the extracted city
                            extracted_city = university_location['city']
                            
                            # Skip if extracted city contains phrases that indicate bad extraction
                            bad_phrases = ['it is', 'the university', 'located in', 'based in', 
                                         'campus', 'observatory', 'laboratory', 'founded in',
                                         'about', 'students', 'largest', 'third', 'with about',
                                         'due to', 'because', 'since', 'n/a', 'none', 'unknown']
                            
                            # More comprehensive validation
                            is_valid_city = (
                                len(extracted_city) > 2 and
                                len(extracted_city) < 50 and  # City names shouldn't be too long
                                not any(phrase in extracted_city.lower() for phrase in bad_phrases) and
                                not extracted_city.lower().endswith(' to') and  # Catch "Due To" pattern
                                not extracted_city.lower().endswith(' of') and
                                not any(char.isdigit() for char in extracted_city[:5]) and  # Cities shouldn't start with numbers
                                extracted_city.count(' ') < 4  # Not a sentence
                            )
                            
                            if is_valid_city:
                                # Check if we already have a good city name
                                current_city = collaborator.get('city', '').strip()
                                
                                # Don't replace a good city with the search result if current looks better
                                if (current_city and 
                                    len(current_city) > 3 and 
                                    len(current_city) < 30 and
                                    not any(bad in current_city.lower() for bad in ['n/a', 'none', 'unknown']) and
                                    not current_city.lower().startswith('universit')):
                                    # Current city looks good, only replace if extracted is clearly better
                                    if len(extracted_city.split()) > len(current_city.split()) + 1:
                                        # Extracted city has significantly more detail, skip replacement
                                        if self.verbose >= 3:
                                            print(f"         ℹ️  Keeping existing city '{current_city}' (looks better than '{extracted_city}')")
                                        continue
                                
                                # Save original city if we're replacing it
                                if city and city.lower() not in ['none', '', 'n/a']:
                                    collaborator['original_city'] = city
                                    
                                # Update collaborator location
                                collaborator['city'] = extracted_city
                                if university_location.get('state'):
                                    collaborator['state'] = university_location['state']
                                if university_location.get('address'):
                                    collaborator['university_address'] = university_location['address']
                                
                                if self.verbose >= 2:
                                    if 'original_city' in collaborator:
                                        print(f"         📍 Corrected location: {collaborator['original_city']} → {extracted_city}")
                                    else:
                                        print(f"         📍 Enhanced location: {extracted_city}")
                            else:
                                if self.verbose >= 3:
                                    print(f"         ⚠️  Invalid city extraction: '{extracted_city}' - keeping original")
                
                # Display based on verbosity
                if self.verbose >= 3:
                    # Use detailed extraction display
                    self.display.detailed_extraction_info(
                        idx, len(matches), match.group(0), collaborator
                    )
                elif self.verbose >= 2:
                    # Medium verbosity - show name and basic info
                    name = collaborator.get('fullName', 'Unknown')
                    location = f"{collaborator.get('city', '')}, {collaborator.get('country', '')}".strip(', ')
                    self.display.timed_info(
                        f"#{idx:03d} {name} - {collaborator.get('affiliation', 'N/A')} ({location})",
                        Icons.PERSON
                    )
                elif self.verbose >= 1:
                    # Low verbosity - show card
                    self.display.collaborator_card(collaborator)
                
                results.append(collaborator)
                
                # Progress bar for lower verbosity
                if self.verbose < 2:
                    self.display.progress(idx, len(matches), "Processing")
        
        # Geocoding phase
        if not skip_geocoding:
            self.display.section("Geocoding Locations", Icons.GLOBE)
            geocoded_count = 0
            
            for idx, collaborator in enumerate(results, 1):
                # Determine best location string
                location_str = ""
                
                # Try enhanced location first if available
                if collaborator.get('enhanced_location'):
                    location_str = collaborator['enhanced_location']
                elif collaborator.get('city') and collaborator.get('country'):
                    location_str = f"{collaborator['city']}, {collaborator['country']}"
                elif collaborator.get('university') and collaborator.get('country'):
                    location_str = f"{collaborator['university']}, {collaborator['country']}"
                elif collaborator.get('country'):
                    location_str = collaborator['country']
                
                # Show what we're geocoding
                if location_str and self.verbose >= 3:
                    self.display.timed_info(
                        f"Geocoding #{idx}: {location_str[:50]}{'...' if len(location_str) > 50 else ''}",
                        Icons.PIN
                    )
                
                # Geocode
                coords = self.geocode_location(location_str, skip=skip_geocoding)
                collaborator['locationString'] = location_str
                collaborator['coordinates'] = coords or {"lat": None, "lng": None}
                
                if coords:
                    geocoded_count += 1
                    if self.verbose >= 3:
                        self.display.extraction_field(
                            "Coordinates",
                            f"lat: {coords['lat']:.4f}, lng: {coords['lng']:.4f}",
                            indent=6
                        )
                elif location_str:
                    failed_geocode.append({
                        'id': collaborator['id'],
                        'firstName': collaborator.get('firstName', ''),
                        'lastName': collaborator.get('lastName', ''),
                        'locationString': location_str,
                        'reason': 'Geocoding failed'
                    })
                    if self.verbose >= 3:
                        self.display.extraction_field("Geocoding", "Failed", indent=6)
                
                # Progress for lower verbosity
                if self.verbose >= 1 and self.verbose < 3:
                    self.display.progress(idx, len(results), "Geocoding")
            
            if self.verbose >= 1:
                self.display.info("Successfully geocoded", f"{geocoded_count}/{len(results)} locations", Icons.PIN)
        
        # Web search phase
        if not skip_search and (self.web_search or self.simple_search):
            self.display.section("Searching for Researcher Information", Icons.SEARCH)
            
            # Prepare search tasks
            search_tasks = []
            for collaborator in results:
                if collaborator.get('firstName') and collaborator.get('lastName'):
                    name = f"{collaborator['firstName']} {collaborator['lastName']}"
                    search_tasks.append({
                        'name': name,
                        'affiliation': collaborator.get('affiliation', ''),
                        'country': collaborator.get('country', ''),
                        'collaborator_id': collaborator['id']
                    })
            
            # Search in batches
            if search_tasks:
                self.display.info("Researchers to search", len(search_tasks), Icons.GLOBE)
                
                # Create event loop if not exists
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                # Search with rate limiting
                search_count = 0
                for i in range(0, len(search_tasks), 3):  # Smaller batch of 3 to be more conservative
                    batch = search_tasks[i:i+3]
                    batch_results = []
                    
                    for task in batch:
                        search_count += 1
                        
                        # Show detailed search info
                        if self.verbose >= 3:
                            self.display.search_attempt_detail(
                                task['name'],
                                task['affiliation'],
                                task['country'],
                                search_count,
                                len(search_tasks)
                            )
                        elif self.verbose >= 2:
                            self.display.spinner(f"Searching: {task['name']} ({search_count}/{len(search_tasks)})")
                        
                        result = await self.search_researcher_info(
                            task['name'],
                            task['affiliation'],
                            task['country']
                        )
                        batch_results.append((task['collaborator_id'], result))
                        
                        # Show result based on verbosity
                        if self.verbose >= 3:
                            # Very verbose - show all details
                            found_items = []
                            if result.get('homepage'):
                                found_items.append(f"Homepage: {result['homepage'][:40]}...")
                            if result.get('email'):
                                found_items.append(f"Email: {result['email']}")
                            profiles_found = [(k, v) for k, v in result.get('profiles', {}).items() if v]
                            for platform, profile_id in profiles_found:
                                found_items.append(f"{platform}: {profile_id}")
                            
                            if found_items:
                                print(f"\n      ✅ Found {len(found_items)} item(s):")
                                for item in found_items:
                                    print(f"         - {item}")
                            else:
                                print("      ❌ No profiles found")
                            
                            # Show metrics if available
                            metrics = result.get('academic_metrics', {})
                            if any(metrics.values()):
                                print("      📊 Metrics:")
                                for metric, value in metrics.items():
                                    if value:
                                        print(f"         - {metric}: {value}")
                        elif self.verbose >= 2 and any(result.get(k) for k in ['homepage', 'email', 'profiles']):
                            # Medium verbose - show card
                            self.display.clear_line()
                            self.display.search_result_card(task['name'], result)
                        
                        time.sleep(1.5)  # Increased rate limit to respect API quotas
                    
                    # Merge search results
                    for collab_id, search_result in batch_results:
                        for collaborator in results:
                            if collaborator['id'] == collab_id:
                                collaborator.update({
                                    'homepage': search_result.get('homepage'),
                                    'email': search_result.get('email'),
                                    'academicTitle': search_result.get('title'),
                                    'profiles': search_result.get('profiles', {}),
                                    'academicMetrics': search_result.get('academic_metrics', {}),
                                    'researchInterests': search_result.get('research_interests', []),
                                    'searchConfidence': search_result.get('search_confidence', 0.0)
                                })
                                break
                
                self.display.clear_line()
                
                # Summary of search results
                found_homepage = sum(1 for r in results if r.get('homepage'))
                found_profiles = sum(1 for r in results if any(r.get('profiles', {}).values()))
                self.display.success(f"Found {found_homepage} homepages and {found_profiles} researcher profiles")
        
        # Save caches
        self._save_cache(GEOCODE_CACHE_FILE, self.geocode_cache)
        self._save_cache(LLM_CACHE_FILE, self.llm_cache)
        self._save_cache(SEARCH_CACHE_FILE, self.search_cache)
        
        return results, failed_geocode


async def main():
    """Main function to run the enhanced collaborator extraction."""
    parser = argparse.ArgumentParser(
        description="Extract collaborator data from LaTeX with LLM enhancement and web search",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Input .tex file in 'input_data' directory"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Configuration file (default: config.json)"
    )
    parser.add_argument(
        "--skip-geocoding",
        action="store_true",
        help="Skip geocoding process"
    )
    parser.add_argument(
        "--skip-search",
        action="store_true",
        help="Skip web search for researcher info"
    )
    parser.add_argument(
        "--verbose", "-v",
        type=int,
        default=2,
        choices=[0, 1, 2, 3],
        help="Verbosity level (0-3, default: 2)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON filename (default: based on input)"
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output"
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear all cache files before running"
    )
    parser.add_argument(
        "--clear-search-cache",
        action="store_true",
        help="Clear only search cache before running"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable cache (don't read or write cache files)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Re-verify existing researcher profiles (email, homepage, ResearchGate, etc.) using strict validation rules"
    )
    
    args = parser.parse_args()
    
    # Initialize display
    display = TerminalDisplay(use_colors=not args.no_color, verbose=args.verbose)
    
    # Handle cache clearing
    if args.clear_cache:
        display.info("Cache management", "Clearing all cache files...", Icons.INFO)
        cache_files = [
            GEOCODE_CACHE_FILE, LLM_CACHE_FILE, SEARCH_CACHE_FILE,
            CACHE_DIR / ".api_cache.json", CACHE_DIR / ".keyword_cache.json",
            CACHE_DIR / ".llm_gemini_cache.json", CACHE_DIR / ".llm_parse_cache.json",
            CACHE_DIR / ".llm_parser_cache.json", CACHE_DIR / ".llm_enricher_cache.json"
        ]
        for cache_file in cache_files:
            if cache_file.exists():
                cache_file.unlink()
                if args.verbose >= 2:
                    display.success(f"Deleted {cache_file.name}")
        display.success("All cache files cleared")
    elif args.clear_search_cache:
        display.info("Cache management", "Clearing search cache...", Icons.INFO)
        if SEARCH_CACHE_FILE.exists():
            SEARCH_CACHE_FILE.unlink()
            display.success("Search cache cleared")
    
    # Setup paths
    tex_file_path = INPUT_DIR / args.input_file
    if args.output:
        output_json_path = OUTPUT_DIR / args.output
    else:
        output_json_path = OUTPUT_DIR / f"{tex_file_path.stem}.json"
    
    config_path = Path(args.config) if args.config else None
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Check input file
    if not tex_file_path.exists():
        display.error(f"Input file not found: {tex_file_path}")
        return
    
    # Show header
    display.header("Enhanced Collaborator Extraction System", Icons.ROCKET)
    
    # Show configuration
    display.section("Configuration", Icons.INFO)
    display.info("Input file", tex_file_path.name, Icons.BOOK)
    display.info("Output file", output_json_path.name, Icons.CHART)
    display.info("Config file", config_path.name if config_path else "Using defaults", Icons.INFO)
    display.info("Verbosity", f"Level {args.verbose}", Icons.SPARKLES)
    
    start_time = time.time()
    
    # Initialize extractor
    display.timed_info("Initializing extraction system...", Icons.HOURGLASS)
    extractor = EnhancedCollaboratorExtractor(
        config_file=config_path,
        verbose=args.verbose,
        use_cache=not args.no_cache
    )
    
    # Check available features
    display.section("System Capabilities", Icons.STAR)
    features = []
    
    if extractor.llm_manager.is_available():
        provider = extractor.llm_manager.get_active_provider_name()
        features.append((True, "LLM Enhancement", provider))
    else:
        features.append((False, "LLM Enhancement", "Not configured"))
    
    if extractor.web_search:
        features.append((True, "Web Search", "Google Custom Search API"))
    elif extractor.simple_search:
        features.append((True, "Web Search", "Simple Search (Limited)"))
    else:
        features.append((False, "Web Search", "Not configured"))
    
    features.append((not args.skip_geocoding, "Geocoding", "OpenStreetMap Nominatim"))
    features.append((True, "Caching", "Enabled for all operations"))
    
    display.feature_status(features)
    
    # Read LaTeX content
    display.timed_info("Reading LaTeX file...", Icons.BOOK)
    tex_content = tex_file_path.read_text(encoding="utf-8")
    display.success(f"Loaded {len(tex_content)} characters")
    
    # Handle verify mode
    if args.verify:
        # Check if output file exists
        if not output_json_path.exists():
            display.error(f"Output file not found for verification: {output_json_path}")
            display.info("Run without --verify flag first to generate the data", "", Icons.INFO)
            return
        
        # Load existing data
        display.timed_info("Loading existing data for verification...", Icons.BOOK)
        with open(output_json_path, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
        display.success(f"Loaded {len(existing_data)} collaborators")
        
        # Process in verify mode
        collaborators, failed_geocode = await extractor.process_collaborators(
            tex_content="",  # Not needed for verify mode
            skip_geocoding=True,
            skip_search=True,
            verify_only=True,
            existing_data=existing_data
        )
    else:
        # Normal processing
        collaborators, failed_geocode = await extractor.process_collaborators(
            tex_content,
            skip_geocoding=args.skip_geocoding,
            skip_search=args.skip_search
        )
    
    # Write output
    display.section("Saving Results", Icons.CHART)
    display.timed_info(f"Writing {len(collaborators)} collaborators to JSON...", Icons.SPARKLES)
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(collaborators, f, ensure_ascii=False, indent=2)
    
    display.success(f"Output saved to: {output_json_path}")
    
    # Summary
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    display.section("Processing Summary", Icons.STAR)
    
    # Calculate statistics
    stats = {
        "Total collaborators": len(collaborators),
        "LLM enhanced": sum(1 for c in collaborators if c.get('llm_enhanced', False)),
        "With homepage": sum(1 for c in collaborators if c.get('homepage')),
        "With email": sum(1 for c in collaborators if c.get('email')),
        "With ORCID": sum(1 for c in collaborators if c.get('profiles', {}).get('orcid')),
        "With Google Scholar": sum(1 for c in collaborators if c.get('profiles', {}).get('google_scholar')),
        "With any profile": sum(1 for c in collaborators if any(c.get('profiles', {}).values())),
        "Successfully geocoded": len(collaborators) - len(failed_geocode),
        "Failed geocoding": len(failed_geocode),
        "Execution time (seconds)": elapsed_time,
    }
    
    # Add verification stats if in verify mode
    if args.verify:
        total_removed = sum(len(c.get('verification_removed', [])) for c in collaborators)
        stats["Items removed in verification"] = total_removed
        stats["Profiles with removals"] = sum(1 for c in collaborators if c.get('verification_removed'))
    
    display.summary_table(stats)
    
    # Show failures if any
    if failed_geocode and args.verbose >= 1:
        display.section("Geocoding Failures", Icons.WARNING)
        for fail in failed_geocode[:5]:  # Show first 5
            display.warning(f"#{fail['id']} {fail['firstName']} {fail['lastName']} - {fail['locationString']}")
        if len(failed_geocode) > 5:
            display.info("", f"... and {len(failed_geocode) - 5} more", Icons.INFO)
    
    # Final message
    if args.verify:
        display.header(f"Verification Complete! {Icons.CHECK}", Icons.SPARKLES)
    else:
        display.header(f"Extraction Complete! {Icons.CHECK}", Icons.SPARKLES)
    display.timed_info(f"Total execution time: {elapsed_time:.2f} seconds", Icons.STOPWATCH)


if __name__ == "__main__":
    # Run async main
    asyncio.run(main())