# -*- coding: utf-8 -*-
"""
Google Gemini client for advanced collaborator information extraction.

This module provides specialized functions for using Gemini to parse
LaTeX collaborator entries and extract enhanced information.

Author: Dr. Denys Dutykh (Khalifa University of Science and Technology, Abu Dhabi, UAE)
Date: 2025-01-04
"""

import json
import re
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class GeminiCollaboratorParser:
    """Specialized Gemini client for parsing collaborator information."""
    
    # System prompt for collaborator parsing
    COLLABORATOR_PROMPT = """You are an expert at parsing LaTeX bibliography entries containing collaborator information. 
Your task is to extract and enhance collaborator data with high accuracy.

For each collaborator entry, extract and enhance the following information:

1. **Name Information**:
   - firstName: Extract first name(s), middle names, initials
   - lastName: Extract last name (handle LaTeX commands like \\textsc{})
   - nameVariants: Common variations of the name (e.g., with/without middle initial)

2. **Affiliation Details**:
   - affiliation: The primary affiliation (department, lab, or institute)
   - university: The university name (if applicable) - IMPORTANT: Extract the actual university name, NOT the department
     * Look for words like "University", "Universiti", "Université", "Universidad", "Universität"
     * Example: "Civil Engineering Department, Universiti Kebangsaan Malaysia" → university: "Universiti Kebangsaan Malaysia"
     * NOT: university: "Civil Engineering Department" (this is a department, not a university)
   - institutionType: university, research_institute, company, government, other
   - department: Specific department name if mentioned (e.g., "Civil Engineering Department", "Department of Physics")
   - researchGroup: Research group or lab name if mentioned

3. **Location Information**:
   - city: City name (handle special characters)
   - country: Full country name (expand abbreviations like UAE, UK, USA)
   - alternateLocations: If "(formerly at)" is mentioned, extract that too

4. **Enhanced Location Data**:
   - Suggest improved location strings for better geocoding
   - Identify ambiguous locations that need clarification
   - Provide confidence score for location extraction (0-1)

5. **Additional Context**:
   - isFormerAffiliation: true if "(formerly at)" is present
   - academicTitle: If mentioned (Professor, Dr., etc.)
   - specialNotes: Any special information about the person

Handle these LaTeX patterns:
- \\textsc{Name} - Small caps formatting
- Accented characters: \\'e, \\`a, \\^o, \\v{c}, etc.
- Special characters: ~, --, ---, \\&
- Complex affiliations with multiple commas

Return a JSON object with all extracted information. Be precise and handle edge cases."""

    LOCATION_ENHANCEMENT_PROMPT = """You are an expert at improving location data for geocoding.
Given collaborator information, suggest the best location string for accurate geocoding.

Consider:
1. If only country is given, suggest the capital city or main academic hub
2. If institution is well-known, you might know its location
3. Disambiguate common city names with state/region
4. Fix common misspellings or variations

Return JSON with:
- original_location: The original location string
- enhanced_location: Your improved suggestion
- confidence: 0-1 score
- reasoning: Brief explanation of changes"""

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash", verbose: int = 2):
        """Initialize the Gemini collaborator parser."""
        self.api_key = api_key
        self.model_name = model
        self.verbose = verbose
        self.model = None
        
        if GEMINI_AVAILABLE and api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model)
            
            # Configure safety settings for academic content
            # Set all thresholds to BLOCK_NONE since we're processing academic collaborator names
            self.safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
            ]
            
            # Note: HARM_CATEGORY_CIVIC_INTEGRITY is not available in all models
            # It would be added as: {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "BLOCK_NONE"}
    
    def _generate_with_timeout(self, prompt: str, generation_config, safety_settings, timeout_seconds: int = 30):
        """
        Generate content with timeout to prevent hanging.
        
        Args:
            prompt: The prompt to generate from
            generation_config: Gemini generation configuration
            safety_settings: Safety settings
            timeout_seconds: Timeout in seconds (default 30)
            
        Returns:
            Generated response or raises TimeoutError
        """
        def _generate():
            return self.model.generate_content(
                prompt,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
        
        # Execute with timeout
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_generate)
            try:
                return future.result(timeout=timeout_seconds)
            except FutureTimeoutError:
                raise RuntimeError(f"Gemini generation timed out after {timeout_seconds} seconds")
    
    def _preprocess_latex(self, latex_text: str) -> str:
        """
        Preprocess LaTeX text to avoid triggering safety filters.
        
        Args:
            latex_text: Raw LaTeX text
            
        Returns:
            Preprocessed text that's less likely to trigger filters
        """
        # Replace LaTeX special characters with Unicode equivalents
        replacements = {
            r"\'e": "é",
            r"\'E": "É", 
            r"\`e": "è",
            r"\`E": "È",
            r"\^e": "ê",
            r"\^E": "Ê",
            r"\"e": "ë",
            r"\"E": "Ë",
            r"\'a": "á",
            r"\'A": "Á",
            r"\`a": "à",
            r"\`A": "À",
            r"\^a": "â",
            r"\^A": "Â",
            r"\"a": "ä",
            r"\"A": "Ä",
            r"\'o": "ó",
            r"\'O": "Ó",
            r"\`o": "ò",
            r"\`O": "Ò",
            r"\^o": "ô",
            r"\^O": "Ô",
            r"\"o": "ö",
            r"\"O": "Ö",
            r"\'u": "ú",
            r"\'U": "Ú",
            r"\`u": "ù",
            r"\`U": "Ù",
            r"\^u": "û",
            r"\^U": "Û",
            r"\"u": "ü",
            r"\"U": "Ü",
            r"\'i": "í",
            r"\'I": "Í",
            r"\`i": "ì",
            r"\`I": "Ì",
            r"\^i": "î",
            r"\^I": "Î",
            r"\"i": "ï",
            r"\"I": "Ï",
            r"\~n": "ñ",
            r"\~N": "Ñ",
            r"\c{c}": "ç",
            r"\c{C}": "Ç",
            r"--": "–",  # en dash
            r"---": "—", # em dash
        }
        
        # Apply replacements
        processed = latex_text
        for latex_char, unicode_char in replacements.items():
            processed = processed.replace(latex_char, unicode_char)
        
        # Remove or simplify other LaTeX commands that might cause issues
        # but keep important structural elements
        processed = re.sub(r'\\textsc\{([^}]+)\}', r'\1', processed)  # Remove \textsc
        processed = re.sub(r'\\emph\{([^}]+)\}', r'\1', processed)    # Remove \emph
        processed = re.sub(r'\\textbf\{([^}]+)\}', r'\1', processed)   # Remove \textbf
        processed = re.sub(r'\\textit\{([^}]+)\}', r'\1', processed)   # Remove \textit
        
        return processed
    
    def parse_collaborator_entry(self, latex_entry: str) -> Optional[Dict[str, Any]]:
        """
        Parse a single LaTeX collaborator entry using Gemini.
        
        Args:
            latex_entry: Raw LaTeX entry like "\\item[Name:] Affiliation"
            
        Returns:
            Parsed collaborator data or None if parsing fails
        """
        if not self.model:
            return None
        
        # Preprocess the LaTeX entry to avoid safety filter issues
        preprocessed_entry = self._preprocess_latex(latex_entry)
        
        prompt = f"""Parse this LaTeX collaborator entry and extract all information:

{preprocessed_entry}

{self.COLLABORATOR_PROMPT}

Return ONLY a valid JSON object."""

        try:
            generation_config = genai.GenerationConfig(
                temperature=0.1,  # Low temperature for consistent parsing
                max_output_tokens=2048,
            )
            
            # First attempt with preprocessed content
            response = self._generate_with_timeout(
                prompt,
                generation_config,
                self.safety_settings
            )
            
            # Check if response was blocked
            if not response.candidates or not response.candidates[0].content.parts:
                finish_reason = response.candidates[0].finish_reason if response.candidates else "No candidates"
                if self.verbose >= 2:
                    print(f"[Gemini] Response blocked or empty. Finish reason: {finish_reason}")
                    if finish_reason == 2:
                        print("[Gemini] Content was blocked by safety filters, attempting with further sanitization...")
                        # Show which safety ratings triggered the block
                        if response.candidates and response.candidates[0].safety_ratings:
                            for rating in response.candidates[0].safety_ratings:
                                if rating.probability != 1:  # 1 is NEGLIGIBLE
                                    print(f"         - {rating.category}: {rating.probability}")
                
                # Retry with more aggressive sanitization
                # Remove all backslashes and LaTeX commands
                sanitized_entry = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', preprocessed_entry)
                sanitized_entry = re.sub(r'\\[a-zA-Z]+', '', sanitized_entry)
                sanitized_entry = sanitized_entry.replace('\\', '')
                
                # Create a simpler prompt
                retry_prompt = f"""Extract information from this text about a research collaborator:

{sanitized_entry}

Extract: firstName, lastName, affiliation, university, city, country, department

Return ONLY a valid JSON object."""
                
                if self.verbose >= 3:
                    print(f"[Gemini] Retrying with sanitized content: {sanitized_entry[:100]}...")
                
                response = self._generate_with_timeout(
                    retry_prompt,
                    generation_config,
                    self.safety_settings
                )
                
                # If still blocked, give up
                if not response.candidates or not response.candidates[0].content.parts:
                    if self.verbose >= 2:
                        print("[Gemini] Still blocked after sanitization. Skipping entry.")
                    return None
            
            # Extract JSON from response
            json_text = response.text.strip()
            # Remove markdown code blocks if present
            json_text = re.sub(r'^```json\s*\n?', '', json_text)
            json_text = re.sub(r'\n?```\s*$', '', json_text)
            
            parsed_data = json.loads(json_text)
            
            # Ensure we have at least basic structure even if partially filled
            if isinstance(parsed_data, dict):
                # Ensure location info exists
                if 'locationInformation' not in parsed_data:
                    parsed_data['locationInformation'] = {}
                
                # For simpler responses, map flat fields to nested structure
                if 'city' in parsed_data and 'locationInformation' in parsed_data:
                    if not parsed_data['locationInformation'].get('city'):
                        parsed_data['locationInformation']['city'] = parsed_data.get('city')
                if 'country' in parsed_data and 'locationInformation' in parsed_data:
                    if not parsed_data['locationInformation'].get('country'):
                        parsed_data['locationInformation']['country'] = parsed_data.get('country')
                
                # Ensure name info exists
                if 'nameInformation' not in parsed_data:
                    parsed_data['nameInformation'] = {}
                if 'firstName' in parsed_data and not parsed_data['nameInformation'].get('firstName'):
                    parsed_data['nameInformation']['firstName'] = parsed_data.get('firstName')
                if 'lastName' in parsed_data and not parsed_data['nameInformation'].get('lastName'):
                    parsed_data['nameInformation']['lastName'] = parsed_data.get('lastName')
            
            return parsed_data
            
        except Exception as e:
            if self.verbose >= 2:
                print(f"[Gemini] Error parsing entry: {str(e)}")
                # If it's an attribute error about response.text, provide more context
                if "response.text" in str(e):
                    print("[Gemini] This typically means the content was blocked by safety filters")
            return None
    
    def enhance_location(self, collaborator_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance location information for better geocoding.
        
        Args:
            collaborator_data: Parsed collaborator data
            
        Returns:
            Enhanced location data
        """
        if not self.model:
            return {"enhanced": False}
        
        context = {
            "name": f"{collaborator_data.get('firstName', '')} {collaborator_data.get('lastName', '')}",
            "affiliation": collaborator_data.get('affiliation', ''),
            "university": collaborator_data.get('university', ''),
            "city": collaborator_data.get('city', ''),
            "country": collaborator_data.get('country', '')
        }
        
        prompt = f"""Enhance this location data for geocoding:

{json.dumps(context, indent=2)}

{self.LOCATION_ENHANCEMENT_PROMPT}"""

        try:
            generation_config = genai.GenerationConfig(
                temperature=0.3,
                max_output_tokens=1024,
            )
            
            response = self._generate_with_timeout(
                prompt,
                generation_config,
                self.safety_settings
            )
            
            # Extract JSON from response
            json_text = response.text.strip()
            json_text = re.sub(r'^```json\s*\n?', '', json_text)
            json_text = re.sub(r'\n?```\s*$', '', json_text)
            
            return json.loads(json_text)
            
        except Exception as e:
            if self.verbose >= 2:
                print(f"[Gemini] Error enhancing location: {str(e)}")
            return {"enhanced": False}
    
    def extract_research_profile(self, name: str, affiliation: str) -> Optional[Dict[str, Any]]:
        """
        Extract research profile information for a collaborator.
        
        Args:
            name: Full name of the collaborator
            affiliation: Their affiliation/institution
            
        Returns:
            Research profile data or None
        """
        if not self.model:
            return None
        
        prompt = f"""Based on the researcher name and affiliation, provide likely research profile information:

Name: {name}
Affiliation: {affiliation}

Provide your best estimate of:
1. Likely research field/discipline (based on institution and department)
2. Academic level (if determinable from name format or institution type)
3. Common research keywords for their likely field
4. Institution ranking/prestige level (world-class, national, regional)

Return JSON with:
- likely_field: Most probable research field
- likely_keywords: Array of 3-5 relevant research keywords
- institution_type: Type of institution
- institution_prestige: high, medium, or standard
- confidence: Your confidence level (0-1)

Note: This is an estimate based on patterns, not specific knowledge about the individual."""

        try:
            generation_config = genai.GenerationConfig(
                temperature=0.5,
                max_output_tokens=1024,
            )
            
            response = self._generate_with_timeout(
                prompt,
                generation_config,
                self.safety_settings
            )
            
            json_text = response.text.strip()
            json_text = re.sub(r'^```json\s*\n?', '', json_text)
            json_text = re.sub(r'\n?```\s*$', '', json_text)
            
            return json.loads(json_text)
            
        except Exception as e:
            if self.verbose >= 2:
                print(f"[Gemini] Error extracting research profile: {str(e)}")
            return None
    
    def batch_parse_collaborators(self, latex_entries: List[str]) -> List[Dict[str, Any]]:
        """
        Parse multiple collaborator entries in batch for efficiency.
        
        Args:
            latex_entries: List of LaTeX entries
            
        Returns:
            List of parsed collaborator data
        """
        if not self.model or not latex_entries:
            return []
        
        # Process in batches of 10 for efficiency
        batch_size = 10
        all_results = []
        
        for i in range(0, len(latex_entries), batch_size):
            batch = latex_entries[i:i + batch_size]
            
            # Show progress
            if self.verbose >= 2:
                batch_num = i // batch_size + 1
                total_batches = (len(latex_entries) + batch_size - 1) // batch_size
                print(f"[Gemini] Processing batch {batch_num}/{total_batches} (entries {i+1}-{min(i+batch_size, len(latex_entries))})")
            
            # Show individual entries if very verbose
            if self.verbose >= 3:
                print(f"[Gemini] Batch contains {len(batch)} entries:")
                for idx, entry in enumerate(batch, start=i+1):
                    entry_preview = entry.strip()[:80] + "..." if len(entry.strip()) > 80 else entry.strip()
                    print(f"         Entry #{idx}: {entry_preview}")
            
            # Preprocess all entries in the batch
            preprocessed_batch = [self._preprocess_latex(entry) for entry in batch]
            
            prompt = f"""Parse these LaTeX collaborator entries and extract all information for each:

{chr(10).join(f"{idx+1}. {entry}" for idx, entry in enumerate(preprocessed_batch))}

{self.COLLABORATOR_PROMPT}

Return a JSON array with one object per entry, maintaining the same order."""

            try:
                if self.verbose >= 3:
                    print("[Gemini] Sending batch to LLM for parsing...")
                
                generation_config = genai.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=4096,
                )
                
                response = self._generate_with_timeout(
                    prompt,
                    generation_config,
                    self.safety_settings
                )
                
                # Check if response was blocked
                if not response.candidates or not response.candidates[0].content.parts:
                    finish_reason = response.candidates[0].finish_reason if response.candidates else "No candidates"
                    if self.verbose >= 2:
                        print(f"[Gemini] Batch response blocked or empty. Finish reason: {finish_reason}")
                        if finish_reason == 2:
                            print("[Gemini] Content was blocked by safety filters")
                    raise Exception(f"Response blocked with finish_reason: {finish_reason}")
                
                json_text = response.text.strip()
                json_text = re.sub(r'^```json\s*\n?', '', json_text)
                json_text = re.sub(r'\n?```\s*$', '', json_text)
                
                batch_results = json.loads(json_text)
                if isinstance(batch_results, list):
                    if self.verbose >= 3:
                        print(f"[Gemini] Successfully parsed {len(batch_results)} entries from batch")
                        # Show extracted data for each entry
                        for idx, result in enumerate(batch_results):
                            entry_num = i + idx + 1
                            print(f"\n[Gemini] Entry #{entry_num} extracted data:")
                            
                            # Handle both old flat structure and new nested structure
                            if 'nameInformation' in result:
                                # New nested structure
                                name_info = result.get('nameInformation', {})
                                print(f"         Name: {name_info.get('firstName', '')} {name_info.get('lastName', '')}")
                                
                                aff_info = result.get('affiliationDetails', {})
                                if aff_info.get('affiliation'):
                                    print(f"         Affiliation: {aff_info.get('affiliation')}")
                                if aff_info.get('university'):
                                    print(f"         University: {aff_info.get('university')}")
                                
                                loc_info = result.get('locationInformation', {})
                                if loc_info.get('city') or loc_info.get('country'):
                                    print(f"         Location: {loc_info.get('city', '')}, {loc_info.get('country', '')}".strip(', '))
                                
                                if aff_info.get('department'):
                                    print(f"         Department: {aff_info.get('department')}")
                                
                                add_context = result.get('additionalContext', {})
                                if add_context.get('isFormerAffiliation'):
                                    print("         Note: Former affiliation")
                            else:
                                # Old flat structure
                                print(f"         Name: {result.get('firstName', '')} {result.get('lastName', '')}")
                                if result.get('affiliation'):
                                    print(f"         Affiliation: {result.get('affiliation')}")
                                if result.get('university'):
                                    print(f"         University: {result.get('university')}")
                                if result.get('city') or result.get('country'):
                                    print(f"         Location: {result.get('city', '')}, {result.get('country', '')}".strip(', '))
                                if result.get('department'):
                                    print(f"         Department: {result.get('department')}")
                                if result.get('isFormerAffiliation'):
                                    print("         Note: Former affiliation")
                    
                    all_results.extend(batch_results)
                
            except Exception as e:
                if self.verbose >= 1:
                    print(f"[Gemini] Batch parsing error: {str(e)}")
                # Fall back to individual parsing
                if self.verbose >= 3:
                    print("[Gemini] Falling back to individual parsing for this batch")
                
                for idx, entry in enumerate(batch):
                    if self.verbose >= 3:
                        print(f"\n[Gemini] Parsing entry #{i+idx+1} individually...")
                    result = self.parse_collaborator_entry(entry)
                    if result:
                        all_results.append(result)
                        if self.verbose >= 3:
                            # Handle both old flat structure and new nested structure
                            if 'nameInformation' in result:
                                name_info = result.get('nameInformation', {})
                                name = f"{name_info.get('firstName', '')} {name_info.get('lastName', '')}"
                            else:
                                name = f"{result.get('firstName', '')} {result.get('lastName', '')}"
                            print(f"         Success: {name.strip()}")
                    else:
                        if self.verbose >= 3:
                            print("         Failed to parse entry")
        
        return all_results


class UnifiedCollaboratorParser:
    """
    Unified collaborator parser that uses LLMProviderManager for systematic fallback.
    
    This class provides the same interface as GeminiCollaboratorParser but automatically
    falls back to the configured fallback LLM (e.g., Claude) when the primary LLM fails
    for any reason including safety filters, JSON parsing errors, or other exceptions.
    
    All LLM requests are cached to save time and API costs.
    """
    
    # Copy the system prompts from GeminiCollaboratorParser
    COLLABORATOR_PROMPT = GeminiCollaboratorParser.COLLABORATOR_PROMPT
    LOCATION_ENHANCEMENT_PROMPT = GeminiCollaboratorParser.LOCATION_ENHANCEMENT_PROMPT
    
    def __init__(self, config, verbose: int = 2, cache_dir: Optional[str] = None):
        """
        Initialize the unified parser with LLM provider manager.
        
        Args:
            config: Configuration manager instance
            verbose: Verbosity level (0-3)
            cache_dir: Directory for cache files (default: current directory)
        """
        self.config = config
        self.verbose = verbose
        
        # Import necessary modules
        from pathlib import Path
        import hashlib
        
        # Set up cache files
        cache_base = Path(cache_dir) if cache_dir else Path.cwd()
        cache_folder = cache_base / "cache"
        cache_folder.mkdir(exist_ok=True)
        self.parse_cache_file = cache_folder / ".llm_unified_parse_cache.json"
        self.location_cache_file = cache_folder / ".llm_unified_location_cache.json"
        self.profile_cache_file = cache_folder / ".llm_unified_profile_cache.json"
        
        # Load caches
        self.parse_cache = self._load_cache(self.parse_cache_file)
        self.location_cache = self._load_cache(self.location_cache_file)
        self.profile_cache = self._load_cache(self.profile_cache_file)
        
        # Import LLMProviderManager
        from .llm_provider import LLMProviderManager
        self.llm_manager = LLMProviderManager(config)
        
        # Check if we have at least one provider
        if not self.llm_manager.is_available():
            raise RuntimeError("No LLM providers available. Please check your API keys in config.json")
        
        # Store the model name for compatibility
        self.model = self.llm_manager.get_active_provider_name()
        
        # Safety settings for compatibility (though they're Gemini-specific)
        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
        ]
    
    def _load_cache(self, cache_file) -> Dict[str, Any]:
        """Load cache from file."""
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                if self.verbose >= 2:
                    print(f"[Cache] Error loading {cache_file}: {e}")
        return {}
    
    def _save_cache(self, cache_data: Dict[str, Any], cache_file):
        """Save cache to file."""
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            if self.verbose >= 2:
                print(f"[Cache] Error saving {cache_file}: {e}")
    
    def _get_cache_key(self, prompt: str, prefix: str = "") -> str:
        """Generate a cache key from prompt."""
        import hashlib
        # Create a hash of the prompt for consistent caching
        prompt_hash = hashlib.md5(prompt.encode('utf-8')).hexdigest()
        return f"{prefix}:{prompt_hash}" if prefix else prompt_hash
    
    def _preprocess_latex(self, latex_text: str) -> str:
        """
        Preprocess LaTeX text to avoid triggering safety filters.
        (Same as GeminiCollaboratorParser)
        """
        # Use the same preprocessing as GeminiCollaboratorParser
        replacements = {
            r"\'e": "é", r"\'E": "É", r"\`e": "è", r"\`E": "È",
            r"\^e": "ê", r"\^E": "Ê", r"\"e": "ë", r"\"E": "Ë",
            r"\'a": "á", r"\'A": "Á", r"\`a": "à", r"\`A": "À",
            r"\^a": "â", r"\^A": "Â", r"\"a": "ä", r"\"A": "Ä",
            r"\'o": "ó", r"\'O": "Ó", r"\`o": "ò", r"\`O": "Ò",
            r"\^o": "ô", r"\^O": "Ô", r"\"o": "ö", r"\"O": "Ö",
            r"\'u": "ú", r"\'U": "Ú", r"\`u": "ù", r"\`U": "Ù",
            r"\^u": "û", r"\^U": "Û", r"\"u": "ü", r"\"U": "Ü",
            r"\'i": "í", r"\'I": "Í", r"\`i": "ì", r"\`I": "Ì",
            r"\^i": "î", r"\^I": "Î", r"\"i": "ï", r"\"I": "Ï",
            r"\~n": "ñ", r"\~N": "Ñ", r"\c{c}": "ç", r"\c{C}": "Ç",
            r"--": "–", r"---": "—"
        }
        
        processed = latex_text
        for latex_char, unicode_char in replacements.items():
            processed = processed.replace(latex_char, unicode_char)
        
        # Remove LaTeX commands
        processed = re.sub(r'\\textsc\{([^}]+)\}', r'\1', processed)
        processed = re.sub(r'\\emph\{([^}]+)\}', r'\1', processed)
        processed = re.sub(r'\\textbf\{([^}]+)\}', r'\1', processed)
        processed = re.sub(r'\\textit\{([^}]+)\}', r'\1', processed)
        
        return processed
    
    def _generate_with_fallback(self, prompt: str, system_prompt: str = None,
                               temperature: float = 0.1, max_tokens: int = 2048) -> Optional[str]:
        """
        Generate response with automatic fallback to secondary LLM.
        
        Returns:
            Generated text or None if all providers fail
        """
        # Try primary provider first
        try:
            response = self.llm_manager.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                use_fallback=False  # Don't use automatic fallback, we handle it explicitly
            )
            if response:
                if self.verbose >= 3:
                    print(f"[LLM] Successfully generated with primary provider")
                return response
        except Exception as e:
            if self.verbose >= 2:
                print(f"[LLM] Primary provider failed: {str(e)}")
                if "finish_reason" in str(e) and "2" in str(e):
                    print("[LLM] Content was blocked by safety filters")
        
        # Try fallback provider
        if self.verbose >= 2:
            print("[LLM] Attempting with fallback provider...")
        
        try:
            response = self.llm_manager.fallback_provider.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            ) if self.llm_manager.fallback_provider else None
            
            if response:
                if self.verbose >= 2:
                    print(f"[LLM] Successfully generated with fallback provider")
                return response
        except Exception as e:
            if self.verbose >= 2:
                print(f"[LLM] Fallback provider also failed: {str(e)}")
        
        return None
    
    def parse_collaborator_entry(self, latex_entry: str) -> Optional[Dict[str, Any]]:
        """
        Parse a single LaTeX collaborator entry using LLM with automatic fallback.
        Results are cached to save API calls.
        
        Args:
            latex_entry: Raw LaTeX entry like "\\item[Name:] Affiliation"
            
        Returns:
            Parsed collaborator data or None if parsing fails
        """
        # Check cache first
        cache_key = self._get_cache_key(latex_entry, "parse")
        if cache_key in self.parse_cache:
            if self.verbose >= 3:
                print("[Cache] Hit for collaborator parsing")
            return self.parse_cache[cache_key]
        
        # Preprocess the LaTeX entry
        preprocessed_entry = self._preprocess_latex(latex_entry)
        
        prompt = f"""Parse this LaTeX collaborator entry and extract all information:

{preprocessed_entry}

{self.COLLABORATOR_PROMPT}

Return ONLY a valid JSON object."""

        # Generate with fallback
        response_text = self._generate_with_fallback(
            prompt=prompt,
            temperature=0.1,
            max_tokens=2048
        )
        
        if not response_text:
            # Cache the failure to avoid repeated attempts
            self.parse_cache[cache_key] = None
            self._save_cache(self.parse_cache, self.parse_cache_file)
            return None
        
        try:
            # Extract JSON from response
            json_text = response_text.strip()
            # Remove markdown code blocks if present
            json_text = re.sub(r'^```json\s*\n?', '', json_text)
            json_text = re.sub(r'\n?```\s*$', '', json_text)
            
            parsed_data = json.loads(json_text)
            
            # Ensure we have at least basic structure (same as GeminiCollaboratorParser)
            if isinstance(parsed_data, dict):
                # Ensure location info exists
                if 'locationInformation' not in parsed_data:
                    parsed_data['locationInformation'] = {}
                
                # For simpler responses, map flat fields to nested structure
                if 'city' in parsed_data and 'locationInformation' in parsed_data:
                    if not parsed_data['locationInformation'].get('city'):
                        parsed_data['locationInformation']['city'] = parsed_data.get('city')
                if 'country' in parsed_data and 'locationInformation' in parsed_data:
                    if not parsed_data['locationInformation'].get('country'):
                        parsed_data['locationInformation']['country'] = parsed_data.get('country')
                
                # Ensure name info exists
                if 'nameInformation' not in parsed_data:
                    parsed_data['nameInformation'] = {}
                if 'firstName' in parsed_data and not parsed_data['nameInformation'].get('firstName'):
                    parsed_data['nameInformation']['firstName'] = parsed_data.get('firstName')
                if 'lastName' in parsed_data and not parsed_data['nameInformation'].get('lastName'):
                    parsed_data['nameInformation']['lastName'] = parsed_data.get('lastName')
            
            # Cache the successful result
            self.parse_cache[cache_key] = parsed_data
            self._save_cache(self.parse_cache, self.parse_cache_file)
            
            return parsed_data
            
        except Exception as e:
            if self.verbose >= 2:
                print(f"[LLM] Error parsing JSON response: {str(e)}")
            # Cache the failure
            self.parse_cache[cache_key] = None
            self._save_cache(self.parse_cache, self.parse_cache_file)
            return None
    
    def enhance_location(self, collaborator_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance location information for better geocoding with automatic fallback.
        Results are cached to save API calls.
        
        Args:
            collaborator_data: Parsed collaborator data
            
        Returns:
            Enhanced location data
        """
        context = {
            "name": f"{collaborator_data.get('firstName', '')} {collaborator_data.get('lastName', '')}",
            "affiliation": collaborator_data.get('affiliation', ''),
            "university": collaborator_data.get('university', ''),
            "city": collaborator_data.get('city', ''),
            "country": collaborator_data.get('country', '')
        }
        
        # Generate cache key from context
        context_str = json.dumps(context, sort_keys=True)
        cache_key = self._get_cache_key(context_str, "location")
        
        # Check cache first
        if cache_key in self.location_cache:
            if self.verbose >= 3:
                print("[Cache] Hit for location enhancement")
            return self.location_cache[cache_key]
        
        prompt = f"""Enhance this location data for geocoding:

{json.dumps(context, indent=2)}

{self.LOCATION_ENHANCEMENT_PROMPT}"""

        # Generate with fallback
        response_text = self._generate_with_fallback(
            prompt=prompt,
            temperature=0.3,
            max_tokens=1024
        )
        
        if not response_text:
            result = {"enhanced": False}
            self.location_cache[cache_key] = result
            self._save_cache(self.location_cache, self.location_cache_file)
            return result
        
        try:
            # Extract JSON from response
            json_text = response_text.strip()
            json_text = re.sub(r'^```json\s*\n?', '', json_text)
            json_text = re.sub(r'\n?```\s*$', '', json_text)
            
            result = json.loads(json_text)
            
            # Cache the successful result
            self.location_cache[cache_key] = result
            self._save_cache(self.location_cache, self.location_cache_file)
            
            return result
            
        except Exception as e:
            if self.verbose >= 2:
                print(f"[LLM] Error parsing location enhancement: {str(e)}")
            result = {"enhanced": False}
            self.location_cache[cache_key] = result
            self._save_cache(self.location_cache, self.location_cache_file)
            return result
    
    def extract_research_profile(self, name: str, affiliation: str) -> Optional[Dict[str, Any]]:
        """
        Extract research profile information with automatic fallback.
        Results are cached to save API calls.
        
        Args:
            name: Full name of the collaborator
            affiliation: Their affiliation/institution
            
        Returns:
            Research profile data or None
        """
        # Generate cache key
        cache_input = f"{name}|{affiliation}"
        cache_key = self._get_cache_key(cache_input, "profile")
        
        # Check cache first
        if cache_key in self.profile_cache:
            if self.verbose >= 3:
                print("[Cache] Hit for research profile")
            return self.profile_cache[cache_key]
        
        prompt = f"""Based on the researcher name and affiliation, provide likely research profile information:

Name: {name}
Affiliation: {affiliation}

Provide your best estimate of:
1. Likely research field/discipline (based on institution and department)
2. Academic level (if determinable from name format or institution type)
3. Common research keywords for their likely field
4. Institution ranking/prestige level (world-class, national, regional)

Return JSON with:
- likely_field: Most probable research field
- likely_keywords: Array of 3-5 relevant research keywords
- institution_type: Type of institution
- institution_prestige: high, medium, or standard
- confidence: Your confidence level (0-1)"""

        # Generate with fallback
        response_text = self._generate_with_fallback(
            prompt=prompt,
            temperature=0.5,
            max_tokens=1024
        )
        
        if not response_text:
            # Cache the failure
            self.profile_cache[cache_key] = None
            self._save_cache(self.profile_cache, self.profile_cache_file)
            return None
        
        try:
            # Extract JSON from response
            json_text = response_text.strip()
            json_text = re.sub(r'^```json\s*\n?', '', json_text)
            json_text = re.sub(r'\n?```\s*$', '', json_text)
            
            result = json.loads(json_text)
            
            # Cache the successful result
            self.profile_cache[cache_key] = result
            self._save_cache(self.profile_cache, self.profile_cache_file)
            
            return result
            
        except Exception as e:
            if self.verbose >= 2:
                print(f"[LLM] Error parsing research profile: {str(e)}")
            # Cache the failure
            self.profile_cache[cache_key] = None
            self._save_cache(self.profile_cache, self.profile_cache_file)
            return None
    
    def batch_parse_collaborators(self, latex_entries: List[str]) -> List[Dict[str, Any]]:
        """
        Parse multiple collaborator entries in batch.
        Note: This implementation processes entries individually with fallback support.
        """
        if not latex_entries:
            return []
        
        results = []
        for i, entry in enumerate(latex_entries):
            if self.verbose >= 2:
                print(f"[LLM] Processing entry {i+1}/{len(latex_entries)}")
            
            result = self.parse_collaborator_entry(entry)
            if result:
                results.append(result)
            else:
                results.append({})  # Empty dict for failed entries
        
        return results