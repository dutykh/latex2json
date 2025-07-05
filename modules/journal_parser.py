# -*- coding: utf-8 -*-
"""
LaTeX parser module for extracting journal paper information from LaTeX bibliography files.

This module handles parsing of etaremune environments and extracting structured data
from LaTeX journal paper entries, including preprint URLs from comments.

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


class LatexJournalParser:
    """Parser for LaTeX journal paper entries in etaremune environment."""
    
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
                    self.llm_parser = LLMBibliographyParser(config, verbose=verbose, entry_type='journal')
                    if self.llm_parser.enabled:
                        if self.verbose >= 2:
                            print("[PARSE] LLM parser enabled for enhanced journal parsing")
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
            List of parsed journal entries
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
                    if self.verbose >= 3:
                        entry_num = len(entries) + 1
                        print(f"\n[PARSE] Processing entry #{entry_num}")
                        latex_preview = entry_content[:100].replace('\n', ' ').strip()
                        if latex_preview:
                            print(f"[PARSE] LaTeX preview: {latex_preview}...")
                    entry = self._parse_entry(entry_content, current_year)
                    if entry:
                        entries.append(entry)
                        if self.verbose >= 2:
                            title = entry.get('title', 'Unknown')[:50]
                            authors_count = len(entry.get('authors', []))
                            print(f"[PARSE] ✓ Parsed: {title}... ({authors_count} authors)")
        
        return entries
    
    def _parse_entry(self, content: str, year: Optional[int]) -> Optional[Dict]:
        """
        Parse a single journal entry.
        
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
        
        # Extract preprint URLs from comments first
        arxiv_url, hal_url, rg_url = self._extract_preprint_urls(content)
        
        # Try LLM parsing first if available
        if self.llm_parser:
            try:
                llm_result = self.llm_parser.parse_entry(content)
                if llm_result:
                    # Ensure year is set from section if not in LLM result
                    if year and not llm_result.get('year'):
                        llm_result['year'] = year
                    
                    # Add preprint URLs
                    if arxiv_url:
                        llm_result['arxiv_url'] = arxiv_url
                    if hal_url:
                        llm_result['hal_url'] = hal_url
                    
                    # Ensure all required fields exist
                    required_fields = ['authors', 'title', 'journal', 'volume', 'issue', 
                                     'pages', 'year', 'url', 'doi', 'abstract', 'keywords',
                                     'arxiv_url', 'hal_url', 'open_access']
                    for field in required_fields:
                        if field not in llm_result:
                            if field in ['authors', 'keywords']:
                                llm_result[field] = []
                            elif field == 'pages':
                                llm_result[field] = {}
                            elif field == 'open_access':
                                llm_result[field] = bool(arxiv_url or hal_url)
                            else:
                                llm_result[field] = ''
                    
                    return llm_result
            except Exception as e:
                if self.verbose >= 1:
                    print(f"[PARSE] LLM parsing failed: {e}, falling back to regex")
        
        # Fall back to regex parsing
        entry = {
            "authors": [],
            "title": "",
            "journal": "",
            "volume": "",
            "issue": "",
            "pages": {},
            "year": year,
            "doi": "",
            "url": "",
            "arxiv_url": arxiv_url or "",
            "hal_url": hal_url or "",
            "abstract": "",
            "keywords": [],
            "keyword_source": "",
            "open_access": bool(arxiv_url or hal_url)
        }
        
        # Remove comments to avoid interference with parsing
        clean_content = re.sub(r'%.*$', '', content, flags=re.MULTILINE)
        
        # Extract URL if present
        url_match = re.search(r'\\url\{(.*?)\}', clean_content)
        if url_match:
            entry["url"] = url_match.group(1)
            clean_content = clean_content.replace(url_match.group(0), '')
            if self.verbose >= 2:
                print(f"[PARSE] Found URL: {entry['url']}")
        
        # Parse authors and title
        self._parse_authors_and_title(clean_content, entry)
        
        # Parse journal information
        self._parse_journal_info(clean_content, entry)
        
        # Parse pages and year
        self._parse_pages_and_year(clean_content, entry)
        
        if self.verbose >= 2:
            print("[PARSE] Extracted data:")
            print(f"  - Authors: {len(entry['authors'])} found")
            if entry['authors'] and self.verbose >= 3:
                for auth in entry['authors']:
                    print(f"    • {auth['firstName']} {auth['lastName']}")
            print(f"  - Title: {entry['title'][:60]}..." if len(entry['title']) > 60 else f"  - Title: {entry['title']}")
            print(f"  - Journal: {entry['journal']}")
            if entry['volume']:
                vol_info = f"Vol. {entry['volume']}"
                if entry['issue']:
                    vol_info += f"({entry['issue']})"
                print(f"  - Volume/Issue: {vol_info}")
            if entry['pages']:
                print(f"  - Pages: {entry['pages'].get('start', '?')}-{entry['pages'].get('end', '?')}")
            if entry['year']:
                print(f"  - Year: {entry['year']}")
            if entry['arxiv_url']:
                print(f"  - ArXiv: {entry['arxiv_url']}")
            if entry['hal_url']:
                print(f"  - HAL: {entry['hal_url']}")
        
        return entry
    
    def _extract_preprint_urls(self, content: str) -> tuple:
        """
        Extract preprint URLs from LaTeX comments.
        
        Args:
            content: The full entry content including comments
            
        Returns:
            Tuple of (arxiv_url, hal_url, researchgate_url)
        """
        arxiv_url = None
        hal_url = None
        rg_url = None
        
        # Look for comments containing preprint info
        comment_match = re.search(r'%\s*(.*?)$', content, re.MULTILINE)
        if comment_match:
            comment = comment_match.group(1)
            if self.verbose >= 3:
                print(f"[PARSE] Found comment: {comment}")
            
            # Check for Yes/No indicators for HAL, ArXiv, RG
            if 'Yes' in comment:
                # Extract what's marked as Yes
                yes_items = []
                if re.search(r'Yes\s*[-–]\s*HAL', comment, re.IGNORECASE):
                    yes_items.append('HAL')
                if re.search(r'Yes\s*[-–]\s*Arxiv', comment, re.IGNORECASE):
                    yes_items.append('ArXiv')
                if re.search(r'Yes\s*[-–]\s*RG', comment, re.IGNORECASE):
                    yes_items.append('RG')
                
                # Also handle format like "Yes - (Arxiv & RG)"
                paren_match = re.search(r'Yes\s*[-–]\s*\((.*?)\)', comment, re.IGNORECASE)
                if paren_match:
                    items = paren_match.group(1)
                    if 'HAL' in items:
                        yes_items.append('HAL')
                    if 'Arxiv' in items or 'ArXiv' in items:
                        yes_items.append('ArXiv')
                    if 'RG' in items:
                        yes_items.append('RG')
                
                if self.verbose >= 3 and yes_items:
                    print(f"[PARSE] Preprint available on: {', '.join(yes_items)}")
        
        # Extract actual URLs from the content
        # Look for ArXiv URLs
        arxiv_match = re.search(r'https?://arxiv\.org/abs/(\d+\.\d+)/?', content)
        if arxiv_match:
            arxiv_url = arxiv_match.group(0).rstrip('/')
            if self.verbose >= 2:
                print(f"[PARSE] Found ArXiv URL: {arxiv_url}")
        
        # Look for HAL URLs
        hal_match = re.search(r'https?://hal\.[a-z\-]+\.fr/[a-z\-]+\d+/?', content)
        if hal_match:
            hal_url = hal_match.group(0).rstrip('/')
            if self.verbose >= 2:
                print(f"[PARSE] Found HAL URL: {hal_url}")
        
        return arxiv_url, hal_url, rg_url
    
    def _parse_authors_and_title(self, content: str, entry: Dict) -> None:
        """Parse authors and title from the entry."""
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
    
    def _parse_journal_info(self, content: str, entry: Dict) -> None:
        """Parse journal information including name, volume, issue."""
        # Remove title and authors to focus on journal info
        title_match = re.search(r'\\textit\{.*?\}', content)
        if title_match:
            journal_part = content[title_match.end():].strip()
        else:
            journal_part = content
        
        # Remove trailing URL if present
        journal_part = re.sub(r'\\\\?\s*\\url\{.*?\}\s*$', '', journal_part)
        
        # Common journal patterns:
        # 1. "Journal Name, Vol(Issue), pages, year"
        # 2. "Journal Name, Volume, pages, year"
        # 3. "Journal Name, Volume(Issue), ID, year"
        
        # Try to match journal with volume(issue) pattern
        vol_issue_pattern = re.search(
            r'^\s*[.,]?\s*([^,]+?),\s*(?:\\textbf\{)?(\d+)(?:\})?(?:\((\d+)\))?,?\s*',
            journal_part
        )
        
        if vol_issue_pattern:
            entry["journal"] = self._clean_latex_text(vol_issue_pattern.group(1).strip())
            entry["volume"] = vol_issue_pattern.group(2)
            if vol_issue_pattern.group(3):
                entry["issue"] = vol_issue_pattern.group(3)
            
            if self.verbose >= 3:
                print(f"[PARSE] Journal: {entry['journal']}, Volume: {entry['volume']}, Issue: {entry['issue']}")
        else:
            # Try simpler pattern without issue
            simple_pattern = re.search(r'^\s*[.,]?\s*([^,]+?),\s*(?:\\textbf\{)?(\d+)(?:\})?,?\s*', journal_part)
            if simple_pattern:
                entry["journal"] = self._clean_latex_text(simple_pattern.group(1).strip())
                entry["volume"] = simple_pattern.group(2)
                
                if self.verbose >= 3:
                    print(f"[PARSE] Journal: {entry['journal']}, Volume: {entry['volume']}")
            else:
                # Try to at least get journal name before first comma
                journal_name_match = re.search(r'^\s*[.,]?\s*([^,]+?)\s*,', journal_part)
                if journal_name_match:
                    entry["journal"] = self._clean_latex_text(journal_name_match.group(1).strip())
                    
                    # Look for volume elsewhere
                    vol_match = re.search(r',\s*(?:\\textbf\{)?(\d+)(?:\})?\s*[,:(]', journal_part)
                    if vol_match:
                        entry["volume"] = vol_match.group(1)
        
        # Clean up journal name
        entry["journal"] = entry["journal"].strip('.').strip(',').strip()
    
    def _parse_pages_and_year(self, content: str, entry: Dict) -> None:
        """Parse page numbers and year from the entry."""
        # Parse pages - various formats
        # Format 1: "123--456" or "123-456" or "123—456"
        pages_match = re.search(r'(\d+)\s*[-–—]+\s*(\d+)', content)
        if pages_match:
            entry["pages"] = {
                "start": int(pages_match.group(1)),
                "end": int(pages_match.group(2))
            }
        else:
            # Format 2: Article number (e.g., "106518")
            # Look for pattern: ", number, year"
            article_match = re.search(r',\s*(\d{4,}),\s*\d{4}', content)
            if article_match:
                article_num = article_match.group(1)
                entry["pages"] = {
                    "article_number": article_num
                }
                if self.verbose >= 3:
                    print(f"[PARSE] Found article number: {article_num}")
        
        # Parse year if not already set (format: ", 2024")
        if not entry["year"]:
            year_match = re.search(r',\s*(\d{4})\s*(?:%|\\\\|$)', content)
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