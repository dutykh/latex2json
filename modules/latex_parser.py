# -*- coding: utf-8 -*-
"""
LaTeX parser module for extracting book chapter information from LaTeX bibliography files.

This module handles parsing of etaremune environments and extracting structured data
from LaTeX bibliography entries.

Author: Dr. Denys Dutykh (Khalifa University of Science and Technology, Abu Dhabi, UAE)
Date: 2025-07-01
"""

import re
from typing import List, Dict, Optional
from pylatexenc.latex2text import LatexNodes2Text

try:
    from .llm_parser import LLMBibliographyParser
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False


class LatexBibliographyParser:
    """Parser for LaTeX bibliography entries in etaremune environment."""
    
    def __init__(self, verbose: int = 2, config=None, use_llm: bool = None):
        """
        Initialize the parser.
        
        Args:
            verbose: Verbosity level (0-3)
            config: Configuration manager instance (optional)
            use_llm: Force enable/disable LLM parsing (None=auto from config)
        """
        self.verbose = verbose
        self.latex_converter = LatexNodes2Text()
        self.config = config
        self.llm_parser = None
        
        # Initialize LLM parser if available and enabled
        if LLM_AVAILABLE and config:
            if use_llm is None:
                use_llm = config.get('preferences.use_llm_parser', True)
            
            if use_llm:
                try:
                    self.llm_parser = LLMBibliographyParser(config, verbose=verbose)
                    if self.llm_parser.enabled:
                        if self.verbose >= 2:
                            print("[PARSE] LLM parser enabled for enhanced parsing")
                    else:
                        self.llm_parser = None
                except Exception as e:
                    if self.verbose >= 1:
                        print(f"[PARSE] Failed to initialize LLM parser: {e}")
                    self.llm_parser = None
        
    def parse_file(self, content: str) -> List[Dict]:
        """
        Parse the entire LaTeX file content.
        
        Args:
            content: The LaTeX file content
            
        Returns:
            List of parsed bibliography entries
        """
        entries = []
        current_year = None
        
        # Find the etaremune environment
        etaremune_match = re.search(r'\\begin\{etaremune\}(.*?)\\end\{etaremune\}', 
                                    content, re.DOTALL)
        
        if not etaremune_match:
            if self.verbose >= 1:
                print("[PARSE] No etaremune environment found")
            return entries
        
        etaremune_content = etaremune_match.group(1)
        
        # Split by \listpart or \item to process entries
        parts = re.split(r'(\\listpart\{.*?\}|\\item\s)', etaremune_content)
        
        for i, part in enumerate(parts):
            # Extract year from \listpart
            year_match = re.match(r'\\listpart\{.*?(\d{4}).*?\}', part)
            if year_match:
                current_year = int(year_match.group(1))
                if self.verbose >= 2:
                    print(f"[PARSE] Found year section: {current_year}")
                continue
            
            # Process \item entries
            if part.strip() == '\\item':
                # Get the content after \item until next \item or \listpart
                if i + 1 < len(parts):
                    entry_content = parts[i + 1]
                    entry = self._parse_entry(entry_content, current_year)
                    if entry:
                        entries.append(entry)
        
        return entries
    
    def _parse_entry(self, content: str, year: Optional[int]) -> Optional[Dict]:
        """
        Parse a single bibliography entry.
        
        Args:
            content: The entry content
            year: The year from the current section
            
        Returns:
            Parsed entry dictionary or None if parsing fails
        """
        if not content.strip():
            return None
            
        if self.verbose >= 3:
            print(f"[PARSE] Processing entry: {content[:100]}...")
        
        # Try LLM parsing first if available
        if self.llm_parser:
            try:
                llm_result = self.llm_parser.parse_entry(content)
                if llm_result:
                    # Ensure year is set from section if not in LLM result
                    if year and not llm_result.get('year'):
                        llm_result['year'] = year
                    
                    # Ensure all required fields exist
                    required_fields = ['authors', 'title', 'pages', 'publisher', 'editors', 
                                     'book_title', 'series', 'volume', 'url', 'doi', 
                                     'chapter_number', 'abstract', 'keywords']
                    for field in required_fields:
                        if field not in llm_result:
                            if field in ['authors', 'editors', 'keywords']:
                                llm_result[field] = []
                            elif field in ['pages', 'publisher']:
                                llm_result[field] = {}
                            else:
                                llm_result[field] = '' if field != 'volume' else None
                    
                    return llm_result
            except Exception as e:
                if self.verbose >= 1:
                    print(f"[PARSE] LLM parsing failed: {e}, falling back to regex")
        
        # Fall back to regex parsing
        entry = {
            "authors": [],
            "title": "",
            "year": year,
            "pages": {},
            "publisher": {},
            "editors": [],
            "book_title": "",
            "series": "",
            "volume": None,
            "url": "",
            "doi": "",
            "chapter_number": None,
            "abstract": "",
            "keywords": []
        }
        
        # Extract URL if present
        url_match = re.search(r'\\url\{(.*?)\}', content)
        if url_match:
            entry["url"] = url_match.group(1)
            content = content.replace(url_match.group(0), '')
            if self.verbose >= 2:
                print(f"[PARSE] Found URL: {entry['url']}")
        
        # Split at "In:" or "in" to separate authors/title from book info
        # Handle variations like ". In:", ", In:", ", in"
        in_match = re.search(r'[.,]\s+[Ii]n:?\s+', content)
        if in_match:
            author_title_part = content[:in_match.start() + 1]  # Include the comma
            book_info_part = content[in_match.end():]
            if self.verbose >= 3:
                print(f"[PARSE] Found book info separator at position {in_match.start()}")
        else:
            # No "In:" found, try to parse what we have
            author_title_part = content
            book_info_part = ""
            if self.verbose >= 3:
                print("[PARSE] No 'In:' separator found")
        
        # Parse authors and title
        self._parse_authors_and_title(author_title_part, entry)
        
        # Parse book information
        if book_info_part:
            self._parse_book_info(book_info_part, entry)
        
        # Parse pages and year from the end of content
        self._parse_pages_and_year(content, entry)
        
        if self.verbose >= 2:
            print("[PARSE] Extracted data:")
            print(f"  - Authors: {len(entry['authors'])} found")
            if entry['authors'] and self.verbose >= 3:
                for auth in entry['authors']:
                    print(f"    • {auth['firstName']} {auth['lastName']}")
            print(f"  - Title: {entry['title'][:60]}..." if len(entry['title']) > 60 else f"  - Title: {entry['title']}")
            if entry['editors']:
                print(f"  - Editors: {len(entry['editors'])} found")
            if entry['book_title']:
                print(f"  - Book title: {entry['book_title'][:50]}..." if len(entry['book_title']) > 50 else f"  - Book title: {entry['book_title']}")
            if entry['publisher'].get('name'):
                print(f"  - Publisher: {entry['publisher']['name']}, {entry['publisher'].get('location', 'Unknown location')}")
            if entry['pages']:
                print(f"  - Pages: {entry['pages'].get('start', '?')}-{entry['pages'].get('end', '?')}")
            if entry['year']:
                print(f"  - Year: {entry['year']}")
        
        return entry
    
    def _parse_authors_and_title(self, content: str, entry: Dict) -> None:
        """Parse authors and title from the first part of the entry."""
        # Remove LaTeX commands for text conversion
        content = re.sub(r'\\textbf\{(.*?)\}', r'\1', content)  # Remove bold
        
        # Find title in \textit{}
        title_match = re.search(r'\\textit\{(.*?)\}', content)
        if title_match:
            entry["title"] = self._clean_latex_text(title_match.group(1))
            # Get everything before the title as authors
            authors_text = content[:title_match.start()].strip()
            entry["authors"] = self._parse_authors(authors_text)
    
    def _parse_authors(self, authors_text: str) -> List[Dict[str, str]]:
        """Parse author names into structured format."""
        authors = []
        
        # Clean up the text
        authors_text = authors_text.replace('~', ' ').strip()
        authors_text = re.sub(r'\s*\.\s*$', '', authors_text)  # Remove trailing period
        
        # Split by "and" first, then by commas
        if ' and ' in authors_text:
            parts = authors_text.split(' and ')
            # Process all but the last part for commas
            for part in parts[:-1]:
                authors.extend(self._split_author_names(part))
            # Add the last part
            authors.extend(self._split_author_names(parts[-1]))
        else:
            authors.extend(self._split_author_names(authors_text))
        
        return authors
    
    def _split_author_names(self, text: str) -> List[Dict[str, str]]:
        """Split a text containing multiple authors separated by commas."""
        authors = []
        names = [n.strip() for n in text.split(',') if n.strip()]
        
        for name in names:
            author = self._parse_single_author(name)
            if author:
                authors.append(author)
        
        return authors
    
    def _parse_single_author(self, name: str) -> Optional[Dict[str, str]]:
        """Parse a single author name into first and last name."""
        name = name.strip()
        if not name:
            return None
        
        # First clean LaTeX commands like \textsc{}
        name = re.sub(r'\\textsc\{(.*?)\}', r'\1', name)
        name = self._clean_latex_text(name)
        
        # Handle names with dots and tildes
        name = name.replace('~', ' ')
        name = re.sub(r'\s+', ' ', name)  # Normalize spaces
        
        # Split into parts
        parts = name.split()
        if not parts:
            return None
        
        # Assume last part is surname, rest is first name
        if len(parts) == 1:
            return {"firstName": "", "lastName": parts[0]}
        else:
            return {
                "firstName": ' '.join(parts[:-1]),
                "lastName": parts[-1]
            }
    
    def _parse_book_info(self, content: str, entry: Dict) -> None:
        """Parse book information including editors, title, series, publisher."""
        original_content = content
        
        # Extract editors (format: "Name (Eds.)" or "Name1 and Name2 (Eds.)")
        editors_match = re.search(r'^(.*?)\s*\(Eds?\.\)(?:\s*:)?', content)
        if editors_match:
            editors_text = editors_match.group(1)
            entry["editors"] = self._parse_authors(editors_text)
            content = content[editors_match.end():].strip()
        
        # Look for publisher patterns with location
        # Handle both "Springer, Singapore" and "Springer Cham" formats
        # First try with comma
        publisher_with_location = re.search(r'(?:^|[.,]\s*)(Springer|Elsevier|Wiley|Birkh[äa\\\\"]*user|CRC Press|Taylor \& Francis|Oxford University Press|Cambridge University Press),\s*([A-Z][a-zA-Z\s]+?)(?:,\s*pp[\.\s~]|,\s*\d{4}|$)', original_content)
        
        # If not found, try without comma (e.g., "Springer Cham")
        if not publisher_with_location:
            publisher_with_location = re.search(r'(?:^|[.,]\s*)(Springer|Elsevier|Wiley)\s+(Cham|Berlin|Heidelberg|New York|London|Amsterdam|Paris|Tokyo|Singapore|Dordrecht)(?:,\s*pp[\.\s~]|,\s*\d{4}|$)', original_content)
        
        # If still not found, try to match any publisher name before year
        if not publisher_with_location:
            # Match pattern: ", Publisher, 2018" or known publishers without location
            publisher_simple = re.search(r',\s*(Springer|Elsevier|Wiley|Birkh[äa\\\\"]*user|CRC Press|Taylor \& Francis|Oxford University Press|Cambridge University Press|[A-Z][a-zA-Z\s&]+?),\s*\d{4}', original_content)
            if publisher_simple:
                entry["publisher"]["name"] = publisher_simple.group(1).strip()
                # No location specified
                entry["publisher"]["location"] = ""
        
        if publisher_with_location:
            entry["publisher"]["name"] = publisher_with_location.group(1).strip()
            entry["publisher"]["location"] = publisher_with_location.group(2).strip()
            
            # Extract book title - everything between editors and publisher
            if editors_match:
                book_title_start = editors_match.end()
            else:
                book_title_start = 0
            book_title_end = original_content.find(publisher_with_location.group(0))
            book_title_content = original_content[book_title_start:book_title_end].strip()
            
            if self.verbose >= 3:
                print(f"[PARSE] Found publisher: {entry['publisher']['name']} in {entry['publisher']['location']}")
        else:
            # Fallback: extract what we can
            book_title_content = content
        
        # Now parse the book title content for title, series, and volume
        # First check for series with parenthetical volume: "Book Title, Series Name (number)"
        series_paren_pattern = re.search(r'^(.+?),\s*([^,]+?)\s*\((?:\\textbf\{)?(\d+)(?:\})?\)', book_title_content)
        
        if series_paren_pattern:
            # Extract book title, series, and volume from parentheses
            entry["book_title"] = self._clean_latex_text(series_paren_pattern.group(1).strip())
            entry["series"] = self._clean_latex_text(series_paren_pattern.group(2).strip())
            entry["volume"] = int(series_paren_pattern.group(3))
            
            if self.verbose >= 3:
                print(f"[PARSE] Series with parentheses: book='{entry['book_title']}', series='{entry['series']}', vol={entry['volume']}")
        # Then check if there's a comma-separated pattern with Vol.
        # Pattern: "Book Title, Series Name, Vol. X" (may have more content after)
        elif (vol_pattern := re.search(r'^(.+?),\s*(.+?),\s*Vol\.?[~\s]+(?:\\textbf\{)?(\d+)(?:\})?', book_title_content)):
            # Comma-separated format with series before volume
            entry["book_title"] = self._clean_latex_text(vol_pattern.group(1).strip())
            entry["series"] = self._clean_latex_text(vol_pattern.group(2).strip())
            entry["volume"] = int(vol_pattern.group(3))
            
            if self.verbose >= 3:
                print(f"[PARSE] Comma pattern match: book='{entry['book_title']}', series='{entry['series']}', vol={entry['volume']}")
        else:
            # Handle pattern: "Book Title Volume X. Conference Name. Series Name"
            # Split by periods to identify components
            parts = [p.strip() for p in book_title_content.split('.') if p.strip()]
            
            if self.verbose >= 3:
                print(f"[PARSE] Book title content: {book_title_content}")
                print(f"[PARSE] Split into parts: {parts}")
            
            if len(parts) >= 2:
                # First part should contain book title and possibly Volume X
                first_part = parts[0]
            
                # Check if the first part ends with Volume X or Vol. X
                # Handle LaTeX commands like \textbf{6}
                volume_match = re.search(r'(.+)\s+(?:Volume|Vol\.?)[~\s]+(?:\\textbf\{)?(\d+)(?:\})?\.?\s*$', first_part)
                if volume_match:
                    if self.verbose >= 3:
                        print(f"[PARSE] Volume match found: '{volume_match.group(1)}' Volume {volume_match.group(2)}")
                    entry["book_title"] = self._clean_latex_text(volume_match.group(1).strip())
                    entry["volume"] = int(volume_match.group(2))
                else:
                    entry["book_title"] = self._clean_latex_text(first_part)
                
                # Process remaining parts for series
                # Look for known series patterns
                remaining_parts = parts[1:]
                series_candidates = []
                
                for part in remaining_parts:
                    if self.verbose >= 3:
                        print(f"[PARSE] Checking part for series: '{part}'")
                    
                    # Skip if it's just a repetition of part of the book title
                    title_words = [w for w in entry["book_title"].split() if len(w) > 4]
                    if title_words and any(title_part in part for title_part in title_words):
                        if self.verbose >= 3:
                            print(f"[PARSE] Skipping '{part}' - contains book title words")
                        continue
                    
                    # Check if it's exactly the publisher name (not a series containing publisher)
                    if entry["publisher"].get("name") and part == entry["publisher"]["name"]:
                        if self.verbose >= 3:
                            print(f"[PARSE] Skipping '{part}' - exactly matches publisher name")
                        continue
                    
                    # Skip if it looks like page numbers (starts with digits and dash)
                    if re.match(r'^\d+[-–—]', part):
                        if self.verbose >= 3:
                            print(f"[PARSE] Skipping '{part}' - looks like page numbers")
                        continue
                    
                    # Otherwise add as series candidate
                    if self.verbose >= 3:
                        print(f"[PARSE] Adding as series candidate: '{part}'")
                    series_candidates.append(part)
            
                # Choose the most likely series (prefer "Springer Water" type patterns)
                if self.verbose >= 3:
                    print(f"[PARSE] Series candidates: {series_candidates}")
                
                for candidate in series_candidates:
                    if any(pub in candidate for pub in ['Springer', 'Elsevier', 'Wiley']) and ' ' in candidate:
                        # This looks like a publisher series (e.g., "Springer Water")
                        entry["series"] = self._clean_latex_text(candidate)
                        if self.verbose >= 3:
                            print(f"[PARSE] Selected series: {candidate}")
                        break
                else:
                    # No publisher series found, use the first non-empty candidate
                    if series_candidates:
                        entry["series"] = self._clean_latex_text(series_candidates[0])
            else:
                # Single part, extract volume if present
                # Handle both "Volume X" and "Vol. X" formats with LaTeX commands
                volume_match = re.search(r'(.+?)\s+(?:Volume|Vol\.?)[~\s]+(?:\\textbf\{)?(\d+)(?:\})?', book_title_content)
                if volume_match:
                    entry["book_title"] = self._clean_latex_text(volume_match.group(1).strip())
                    entry["volume"] = int(volume_match.group(2))
                    # Check for series after volume
                    remaining = book_title_content[volume_match.end():].strip()
                    if remaining:
                        # Remove leading period if present
                        if remaining.startswith('.'):
                            remaining = remaining[1:].strip()
                        if remaining:
                            entry["series"] = self._clean_latex_text(remaining)
                else:
                    # No volume found
                    entry["book_title"] = self._clean_latex_text(book_title_content)
        
        # Clean up book title - remove any remaining artifacts
        book_title = entry["book_title"]
        if self.verbose >= 3:
            print(f"[PARSE] Book title before cleanup: '{book_title}'")
        
        # Remove page numbers pattern (pp. X-Y or pp.X-Y or pp.~X-Y)
        book_title = re.sub(r',?\s*(?:pp\.[\s~]*)\d+[-–—]+\d+', '', book_title)
        # Remove trailing ", pp" without page numbers
        book_title = re.sub(r',\s*pp\s*$', '', book_title)
        # Remove trailing punctuation
        book_title = re.sub(r'\s*[,\.]\s*$', '', book_title)
        # Remove year at the end ONLY if it's preceded by a comma (publication year)
        # Don't remove years that are part of the title (like "SimHydro 2023")
        book_title = re.sub(r',\s*\d{4}\s*$', '', book_title)
        # Remove publisher info if it's still in the book title
        if entry["publisher"].get("name"):
            book_title = book_title.replace(f", {entry['publisher']['name']}", "")
            book_title = book_title.replace(entry['publisher']['name'] + ",", "")
        
        # Final cleanup
        book_title = book_title.strip().strip(',').strip()
        entry["book_title"] = book_title
    
    def _parse_pages_and_year(self, content: str, entry: Dict) -> None:
        """Parse page numbers and year from the entry."""
        # Parse pages (format: "pp. 47--64" or "pp. 47-64" or "pp.~47--64")
        pages_match = re.search(r'pp\.[\s~]*(\d+)\s*[-–—]+\s*(\d+)', content)
        if pages_match:
            entry["pages"] = {
                "start": int(pages_match.group(1)),
                "end": int(pages_match.group(2))
            }
        else:
            # Try to match pages without "pp." prefix (e.g., "197--210")
            # Look for pattern after comma or colon: ", 197--210"
            pages_match = re.search(r'[,:]\s*(\d+)\s*[-–—]+\s*(\d+)(?:\s*,|\s*$)', content)
            if pages_match:
                entry["pages"] = {
                    "start": int(pages_match.group(1)),
                    "end": int(pages_match.group(2))
                }
        
        # Parse year if not already set (format: ", 2024")
        if not entry["year"]:
            year_match = re.search(r',\s*(\d{4})\s*(?:\\\\|$)', content)
            if year_match:
                entry["year"] = int(year_match.group(1))
    
    def _clean_latex_text(self, text: str) -> str:
        """Clean LaTeX text by removing commands and converting special characters."""
        # Remove common LaTeX commands
        text = re.sub(r'\\textsc\{(.*?)\}', r'\1', text)
        text = re.sub(r'\\textit\{(.*?)\}', r'\1', text)
        text = re.sub(r'\\textbf\{(.*?)\}', r'\1', text)
        text = re.sub(r'\\emph\{(.*?)\}', r'\1', text)
        
        # Convert special characters
        replacements = {
            '---': '—',
            '--': '–',
            '~': ' ',
            '\\&': '&',
            '\\"a': 'ä',
            '\\"o': 'ö',
            '\\"u': 'ü',
            "\\'e": 'é',
            "\\`e": 'è',
            "\\^e": 'ê',
            "\\^o": 'ô',
            "\\c{c}": 'ç'
        }
        
        for latex, unicode_char in replacements.items():
            text = text.replace(latex, unicode_char)
        
        # Try using pylatexenc for remaining conversions
        try:
            text = self.latex_converter.latex_to_text(text)
        except Exception:
            # Fallback to basic cleaning
            text = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', text)
            text = re.sub(r'\\[a-zA-Z]+', '', text)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text