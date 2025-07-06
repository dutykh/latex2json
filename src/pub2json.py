#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pub2json.py - LaTeX journal publications to JSON converter with metadata enrichment.

This script parses LaTeX bibliography entries from etaremune environments,
enriches them with metadata from publisher APIs, and optionally generates
keywords using Claude API.

Author: Dr. Denys Dutykh (Khalifa University of Science and Technology, Abu Dhabi, UAE)
Date: 2025-07-01

Usage:
    python pub2json.py [options]
    
Examples:
    python pub2json.py                    # Use defaults
    python pub2json.py -v 3               # Maximum verbosity
    python pub2json.py --dry-run          # Parse only, no API calls
    python pub2json.py --no-keywords      # Disable keyword generation
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to path for module imports
sys.path.append(str(Path(__file__).parent.parent))

# Import our modules
from modules.journal_parser import LatexJournalParser
from modules.cache_manager import CacheManager
from modules.config_manager import ConfigManager
from modules.metadata_enricher import MetadataEnricher
from modules.keyword_generator import KeywordGenerator
from modules.publisher_identifier import PublisherIdentifier
from modules.paper_web_search import PaperWebSearch
from modules.paper_content_extractor import PaperContentExtractor
from modules.display_utils import TerminalDisplay, Icons


class JournalProcessor:
    """Main processor for converting LaTeX journal papers to JSON."""
    
    def __init__(self, args: argparse.Namespace):
        """Initialize the processor with command-line arguments."""
        self.args = args
        self.verbose = args.verbose
        
        # Set up paths
        self.base_dir = Path(__file__).parent.parent.resolve()  # Parent of src/
        self.input_file = Path(args.input).resolve()
        self.output_file = Path(args.output).resolve()
        self.config_file = Path(args.config).resolve() if args.config else None
        
        # Cache files (shared with chpts2json.py)
        cache_dir = self.base_dir / "cache"
        cache_dir.mkdir(exist_ok=True)
        self.api_cache_file = cache_dir / ".api_cache.json"
        self.keyword_cache_file = cache_dir / ".keyword_cache.json"
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """Initialize all processing components."""
        # Load configuration
        self.config = ConfigManager(self.config_file, verbose=self.verbose)
        
        # Initialize display
        self.display = TerminalDisplay(verbose=self.verbose)
        
        # Initialize cache manager
        cache_expiry = self.config.get('preferences.cache_expiry_days', 7)
        self.cache = CacheManager(
            self.api_cache_file, 
            self.keyword_cache_file,
            expiry_days=cache_expiry,
            verbose=self.verbose
        )
        
        # Initialize parser with config for LLM support
        self.parser = LatexJournalParser(verbose=self.verbose, config=self.config)
        
        # Initialize new LLM-powered components
        self.publisher_identifier = PublisherIdentifier(self.config, verbose=self.verbose)
        
        # Initialize enricher (only if not dry run)
        if not self.args.dry_run:
            self.enricher = MetadataEnricher(self.config, self.cache, verbose=self.verbose)
            
            # Initialize paper search and content extraction
            self.paper_search = PaperWebSearch(
                self.config, 
                self.publisher_identifier,
                verbose=self.verbose
            )
            self.content_extractor = PaperContentExtractor(
                self.config,
                self.publisher_identifier,
                verbose=self.verbose
            )
        
        # Initialize keyword generator (only if enabled)
        if not self.args.dry_run and not self.args.no_keywords:
            self.keyword_gen = KeywordGenerator(self.config, self.cache, verbose=self.verbose)
    
    def run(self) -> None:
        """Run the main processing pipeline."""
        start_time = time.time()
        
        self._print_header()
        
        try:
            # Check input file
            if not self.input_file.exists():
                print(f"[ERROR] Input file not found: {self.input_file}")
                sys.exit(1)
            
            # Load and parse LaTeX file
            entries = self._parse_latex_file()
            
            if not entries:
                print("[WARNING] No entries found in LaTeX file")
                sys.exit(0)
            
            # Enrich entries (if not dry run)
            if not self.args.dry_run:
                entries = asyncio.run(self._enrich_entries(entries))
            
            # Sort entries if configured
            entries = self._sort_entries(entries)
            
            # Generate output
            self._generate_output(entries)
            
            # Save caches
            if not self.args.dry_run:
                self.cache.save_caches()
            
            # Print summary
            self._print_summary(entries, start_time)
            
        except KeyboardInterrupt:
            print("\n[INFO] Process interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"\n[ERROR] Unexpected error: {e}")
            if self.verbose >= 3:
                import traceback
                traceback.print_exc()
            sys.exit(1)
    
    def _print_header(self) -> None:
        """Print script header."""
        self.display.header("LaTeX to JSON Converter for Journal Papers", Icons.ROCKET)
        
        # Show configuration
        self.display.section("Configuration", Icons.INFO)
        self.display.info("Input file", str(self.input_file), Icons.BOOK)
        self.display.info("Output file", str(self.output_file), Icons.CHART)
        if self.config_file:
            self.display.info("Config file", str(self.config_file), Icons.INFO)
        self.display.info("Verbosity level", f"Level {self.verbose}", Icons.SPARKLES)
        
        if self.args.dry_run:
            self.display.warning("DRY RUN MODE - No API calls will be made")
        if self.args.no_keywords:
            self.display.warning("Keyword generation disabled")
        
        # Show available features
        self.display.section("System Capabilities", Icons.STAR)
        features = []
        
        # Check LLM availability
        if hasattr(self, 'publisher_identifier'):
            features.append((True, "Publisher Identification", "LLM-powered with fallback"))
        
        # Check search availability
        if hasattr(self, 'paper_search') and self.config.get('api_keys.google') and self.config.get('api_keys.google_cx'):
            features.append((True, "Paper Web Search", "Google Custom Search API"))
        else:
            features.append((False, "Paper Web Search", "API keys not configured"))
        
        # Check extraction
        if hasattr(self, 'content_extractor'):
            features.append((True, "Content Extraction", "LLM-powered extraction"))
        
        # Traditional enrichment
        features.append((not self.args.dry_run, "Metadata Enrichment", "CrossRef, Publisher APIs"))
        features.append((not self.args.dry_run and not self.args.no_keywords, "Keyword Generation", "LLM-powered"))
        
        self.display.feature_status(features)
    
    def _parse_latex_file(self) -> List[Dict[str, Any]]:
        """Parse the LaTeX file."""
        print(f"\n[INFO] Reading LaTeX file: {self.input_file.name}")
        print(f"[INFO] File size: {self.input_file.stat().st_size:,} bytes")
        
        content = self.input_file.read_text(encoding='utf-8')
        
        if self.verbose >= 3:
            # Count etaremune environments
            import re
            etaremune_count = len(re.findall(r'\\begin{etaremune}', content))
            item_count = len(re.findall(r'\\item\[', content))
            print(f"[VERBOSE] Found {etaremune_count} etaremune environment(s)")
            print(f"[VERBOSE] Found {item_count} \\item entries in file")
        
        print("[INFO] Parsing LaTeX content...")
        entries = self.parser.parse_file(content)
        
        print(f"[INFO] Successfully parsed {len(entries)} journal paper entries")
        
        if self.verbose >= 1 and entries:
            print("\n[INFO] Parsed entries summary:")
            for i, entry in enumerate(entries, 1):
                title = entry.get('title', 'Unknown title')[:60]
                if len(entry.get('title', '')) > 60:
                    title += "..."
                authors = entry.get('authors', [])
                if authors:
                    first_author = f"{authors[0].get('firstName', '')} {authors[0].get('lastName', '')}".strip()
                    author_info = f"{first_author}" + (" et al." if len(authors) > 1 else "")
                else:
                    author_info = "Unknown authors"
                year = entry.get('year', 'Unknown year')
                journal = entry.get('journal', 'Unknown journal')[:30]
                if len(entry.get('journal', '')) > 30:
                    journal += "..."
                print(f"  {i:3d}. [{year}] {author_info} - {title}")
                print(f"       {journal}")
                
                if self.verbose >= 3:
                    # Show more details
                    print(f"       DOI: {entry.get('doi', 'Not found')}")
                    print(f"       URL: {entry.get('url', 'Not found')[:50]}..." if entry.get('url') else "       URL: Not found")
                    if entry.get('arxiv_url'):
                        print(f"       ArXiv: {entry.get('arxiv_url')}")
                    if entry.get('hal_url'):
                        print(f"       HAL: {entry.get('hal_url')}")
                    
                    # Show compact JSON structure
                    compact_entry = {
                        'authors': f"{len(entry.get('authors', []))} authors",
                        'title': entry.get('title', '')[:40] + '...',
                        'journal': entry.get('journal', 'N/A'),
                        'year': entry.get('year', 'N/A'),
                        'doi': entry.get('doi', 'N/A')[:20] + '...' if entry.get('doi') else 'N/A'
                    }
                    print(f"       JSON preview: {json.dumps(compact_entry, ensure_ascii=False)}")
                    print()
        
        return entries
    
    async def _enrich_entries(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enrich entries with metadata."""
        # Clear expired cache entries first
        if self.config.get('preferences.cache_responses', True):
            removed, remaining = self.cache.clear_expired()
            if self.verbose >= 2 and removed > 0:
                print(f"[CACHE] Cleaned {removed} expired entries, {remaining} entries remain")
        
        # Phase 1: Publisher identification
        if not self.args.no_publisher_id:
            self.display.section("Publisher Identification", Icons.SPARKLES)
            publisher_stats = {'identified': 0, 'cached': 0, 'llm_queries': 0}
            
            for entry in entries:
                journal = entry.get('journal', '')
                doi = entry.get('doi', '')
                
                if journal:
                    publisher, platform, confidence = self.publisher_identifier.identify_publisher(
                        journal, doi
                    )
                    if publisher:
                        entry['publisher_identified'] = publisher
                        entry['publisher_platform'] = platform
                        entry['publisher_confidence'] = confidence
                        publisher_stats['identified'] += 1
                        
                        if self.verbose >= 3:
                            self.display.info(
                                f"Journal: {journal[:50]}...",
                                f"{publisher} ({platform}) - confidence: {confidence:.2f}",
                                Icons.CHECK
                            )
        
            # Show publisher stats
            pub_id_stats = self.publisher_identifier.get_statistics()
            if self.verbose >= 2:
                print(f"[INFO] Publishers identified: {publisher_stats['identified']}/{len(entries)}")
                print(f"       Cache hits: {pub_id_stats['cache_hits']}, "
                      f"DOI matches: {pub_id_stats['doi_hits']}, "
                      f"LLM queries: {pub_id_stats['llm_queries']}")
        
        # Phase 2: Web search for papers
        if not self.args.no_web_search and self.config.get('search_settings.enable_paper_search', True):
            self.display.section("Searching for Papers Online", Icons.SEARCH)
            
            search_tasks = []
            for idx, entry in enumerate(entries):
                if self.verbose >= 2:
                    self.display.progress(idx + 1, len(entries), "Searching")
                
                # Search for paper
                search_result = await self.paper_search.search_paper(
                    entry,
                    use_publisher_strategy=True
                )
                
                # Add search results to entry
                if search_result.get('publisher_url'):
                    entry['publisher_url'] = search_result['publisher_url']
                    entry['search_confidence'] = search_result['search_confidence']
                    
                    if self.verbose >= 3:
                        self.display.info(
                            "Found URL",
                            f"{search_result['publisher_url'][:60]}...",
                            Icons.GLOBE
                        )
                
                # Store alternative URLs
                if search_result.get('alternative_urls'):
                    entry['alternative_urls'] = search_result['alternative_urls']
            
            search_stats = self.paper_search.get_statistics()
            if self.verbose >= 2:
                self.display.success(
                    f"Search complete: {search_stats['urls_found']} URLs found, "
                    f"{search_stats['urls_validated']} validated"
                )
        
        # Phase 3: Extract content from found URLs
        if not self.args.no_content_extraction and self.config.get('extraction.enable_content_extraction', True):
            self.display.section("Extracting Content from URLs", Icons.SPARKLES)
            
            extracted_count = 0
            for idx, entry in enumerate(entries):
                if self.verbose >= 2:
                    self.display.progress(idx + 1, len(entries), "Extracting")
                
                # Try publisher URL first, then any URL found by traditional enricher
                url = entry.get('publisher_url') or entry.get('url')
                
                # If still no URL but we have alternative URLs, try the first one
                if not url and entry.get('alternative_urls'):
                    alt_urls = entry.get('alternative_urls', [])
                    if alt_urls and isinstance(alt_urls[0], dict):
                        url = alt_urls[0].get('url')
                    elif alt_urls and isinstance(alt_urls[0], str):
                        url = alt_urls[0]
                
                if url:
                    if self.verbose >= 3:
                        self.display.info("Extracting from", url[:80] + "..." if len(url) > 80 else url, Icons.GLOBE)
                    
                    extraction = await self.content_extractor.extract_from_url(
                        url,
                        entry,
                        entry.get('publisher_identified')
                    )
                    
                    # Add extracted content
                    if extraction.get('abstract') and not entry.get('abstract'):
                        entry['abstract'] = extraction['abstract']
                        entry['extraction_method'] = extraction['extraction_method']
                        extracted_count += 1
                    
                    if extraction.get('keywords'):
                        # Merge with existing keywords
                        existing = set(entry.get('keywords', []))
                        new_keywords = extraction['keywords']
                        entry['keywords'] = list(existing | set(new_keywords))
                        if not entry.get('keyword_source'):
                            entry['keyword_source'] = 'extracted'
                    
                    if extraction.get('doi') and not entry.get('doi'):
                        entry['doi'] = extraction['doi']
                    
                    # Add additional metadata
                    if extraction.get('additional_metadata'):
                        entry['additional_metadata'] = extraction['additional_metadata']
            
            if self.verbose >= 2:
                extraction_stats = self.content_extractor.get_statistics()
                self.display.success(
                    f"Extraction complete: {extracted_count} abstracts extracted"
                )
        
        # Phase 3.5: Preprint fallback for entries without abstracts
        if not self.args.no_content_extraction and self.config.get('extraction.use_preprint_fallback', True):
            entries_without_abstract = [e for e in entries if not e.get('abstract') and (e.get('arxiv_url') or e.get('hal_url'))]
            
            if entries_without_abstract:
                self.display.section("Preprint Fallback Extraction", Icons.SPARKLES)
                
                if self.verbose >= 2:
                    self.display.info(
                        "Preprint fallback",
                        f"Found {len(entries_without_abstract)} papers without abstracts but with preprint URLs",
                        Icons.INFO
                    )
                
                preprint_extracted = 0
                for idx, entry in enumerate(entries_without_abstract):
                    if self.verbose >= 2:
                        self.display.progress(idx + 1, len(entries_without_abstract), "Extracting from preprints")
                    
                    # Try ArXiv first, then HAL
                    preprint_url = entry.get('arxiv_url') or entry.get('hal_url')
                    
                    if preprint_url:
                        if self.verbose >= 3:
                            self.display.info("Trying preprint", preprint_url, Icons.GLOBE)
                        
                        extraction = await self.content_extractor.extract_from_url(
                            preprint_url,
                            entry,
                            None  # No publisher for preprints
                        )
                        
                        # Add extracted content
                        if extraction.get('abstract'):
                            entry['abstract'] = extraction['abstract']
                            entry['extraction_method'] = extraction['extraction_method']
                            entry['abstract_source'] = 'preprint'
                            preprint_extracted += 1
                            
                            if self.verbose >= 2:
                                self.display.info(
                                    f"Extracted abstract from {'ArXiv' if 'arxiv' in preprint_url else 'HAL'}",
                                    f"{len(extraction['abstract'])} characters",
                                    Icons.CHECK
                                )
                        
                        # Also update keywords if found
                        if extraction.get('keywords') and not entry.get('keywords'):
                            entry['keywords'] = extraction['keywords']
                            entry['keyword_source'] = extraction.get('keyword_source', 'preprint')
                        
                        # Update DOI if found in preprint
                        if extraction.get('doi') and not entry.get('doi'):
                            entry['doi'] = extraction['doi']
                            entry['doi_source'] = 'preprint'
                
                if self.verbose >= 2:
                    self.display.success(
                        f"Preprint extraction complete: {preprint_extracted} additional abstracts extracted"
                    )
        
        # Phase 4: Traditional enrichment (as fallback)
        print("\n[INFO] Starting traditional metadata enrichment...")
        if self.verbose >= 2:
            print(f"[INFO] Will process {len(entries)} entries for enrichment")
            print("[INFO] API sources enabled: CrossRef, Google Scholar, Publisher APIs")
        
        start_time = time.time()
        entries = await self.enricher.enrich_entries(entries)
        enrichment_time = time.time() - start_time
        
        if self.verbose >= 2:
            print(f"[INFO] Metadata enrichment completed in {enrichment_time:.2f} seconds")
        
        # Phase 5: Preprint fallback for entries without abstracts
        if self.config.get('extraction.use_preprint_fallback', True):
            preprint_count = 0
            entries_without_abstract = []
            
            for entry in entries:
                if not entry.get('abstract'):
                    # Check if we have preprint URLs
                    if entry.get('arxiv_url') or entry.get('hal_url'):
                        entries_without_abstract.append(entry)
            
            if entries_without_abstract:
                print(f"\n[INFO] Found {len(entries_without_abstract)} entries without abstracts that have preprint URLs")
                if self.verbose >= 2:
                    print("[INFO] Attempting to extract abstracts from preprint servers...")
                
                for idx, entry in enumerate(entries_without_abstract):
                    if self.verbose >= 2:
                        self.display.progress(idx + 1, len(entries_without_abstract), "Extracting from preprints")
                    
                    # Try ArXiv first
                    if entry.get('arxiv_url') and not entry.get('abstract'):
                        if self.verbose >= 3:
                            self.display.info("Trying ArXiv", entry['arxiv_url'], Icons.GLOBE)
                        
                        extraction = await self.content_extractor.extract_from_url(
                            entry['arxiv_url'],
                            entry,
                            'ArXiv'
                        )
                        
                        if extraction.get('abstract'):
                            entry['abstract'] = extraction['abstract']
                            entry['extraction_method'] = 'arxiv_fallback'
                            if extraction.get('keywords'):
                                existing = set(entry.get('keywords', []))
                                entry['keywords'] = list(existing | set(extraction['keywords']))
                            if extraction.get('doi') and not entry.get('doi'):
                                entry['doi'] = extraction['doi']
                            preprint_count += 1
                    
                    # Try HAL if still no abstract
                    if entry.get('hal_url') and not entry.get('abstract'):
                        if self.verbose >= 3:
                            self.display.info("Trying HAL", entry['hal_url'], Icons.GLOBE)
                        
                        extraction = await self.content_extractor.extract_from_url(
                            entry['hal_url'],
                            entry,
                            'HAL'
                        )
                        
                        if extraction.get('abstract'):
                            entry['abstract'] = extraction['abstract']
                            entry['extraction_method'] = 'hal_fallback'
                            if extraction.get('keywords'):
                                existing = set(entry.get('keywords', []))
                                entry['keywords'] = list(existing | set(extraction['keywords']))
                            if extraction.get('doi') and not entry.get('doi'):
                                entry['doi'] = extraction['doi']
                            preprint_count += 1
                
                if self.verbose >= 2:
                    self.display.success(f"Extracted {preprint_count} abstracts from preprint servers")
        
        # Phase 6: Generate keywords if enabled
        if not self.args.no_keywords:
            print("\n[INFO] Checking for keyword generation...")
            if self.verbose >= 2:
                entries_needing_keywords = sum(1 for e in entries if not e.get('keywords'))
                print(f"[INFO] {entries_needing_keywords} entries need keyword generation")
            
            start_time = time.time()
            await self.keyword_gen.generate_keywords_for_entries(entries)
            keyword_time = time.time() - start_time
            
            if self.verbose >= 2:
                print(f"[INFO] Keyword generation completed in {keyword_time:.2f} seconds")
        
        return entries
    
    def _sort_entries(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort entries based on configuration."""
        sort_by = self.config.get('output.sort_by', 'year')
        
        if sort_by == 'none':
            return entries
        
        if sort_by == 'year':
            # Sort by year (descending), then by first author, then by title
            return sorted(entries, key=lambda e: (
                -(e.get('year') or 0),
                e.get('authors', [{}])[0].get('lastName', ''),
                e.get('title', '')
            ))
        elif sort_by == 'title':
            return sorted(entries, key=lambda e: e.get('title', ''))
        elif sort_by == 'author':
            return sorted(entries, key=lambda e: 
                e.get('authors', [{}])[0].get('lastName', '') if e.get('authors') else ''
            )
        elif sort_by == 'journal':
            return sorted(entries, key=lambda e: e.get('journal', ''))
        
        return entries
    
    def _generate_output(self, entries: List[Dict[str, Any]]) -> None:
        """Generate the output JSON file."""
        print(f"\n[INFO] Writing output to: {self.output_file}")
        
        # Create output directory if needed
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare output data
        output_data = {
            "journal_papers": entries
        }
        
        # Add metadata if configured
        if self.config.get('output.include_metadata', True):
            output_data["metadata"] = self._generate_metadata(entries)
        
        # Write JSON file
        with open(self.output_file, 'w', encoding='utf-8') as f:
            if self.config.get('output.pretty_print', True):
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            else:
                json.dump(output_data, f, ensure_ascii=False)
        
        print("[SUCCESS] Output written successfully")
    
    def _generate_metadata(self, entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate metadata section for output."""
        enriched_count = sum(1 for e in entries 
                           if e.get('doi') or e.get('abstract') or e.get('keywords'))
        
        keyword_stats = {
            'with_keywords': sum(1 for e in entries if e.get('keywords')),
            'from_publisher': sum(1 for e in entries if e.get('keyword_source') == 'publisher'),
            'generated': sum(1 for e in entries if e.get('keyword_source') == 'generated')
        }
        
        preprint_stats = {
            'with_arxiv': sum(1 for e in entries if e.get('arxiv_url')),
            'with_hal': sum(1 for e in entries if e.get('hal_url'))
        }
        
        return {
            'total_papers': len(entries),
            'enriched_count': enriched_count,
            'keyword_stats': keyword_stats,
            'preprint_stats': preprint_stats,
            'generation_date': datetime.now().isoformat(),
            'script_version': '1.0.0',
            'dry_run': self.args.dry_run
        }
    
    def _print_summary(self, entries: List[Dict[str, Any]], start_time: float) -> None:
        """Print processing summary."""
        elapsed = time.time() - start_time
        
        self.display.section("Processing Summary", Icons.STAR)
        
        print(f"Total papers processed: {len(entries)}")
        
        if not self.args.dry_run:
            # Publisher identification statistics
            with_publisher = sum(1 for e in entries if e.get('publisher_identified'))
            print(f"\nPublisher Identification:")
            print(f"  Publishers identified: {with_publisher}/{len(entries)}")
            if hasattr(self, 'publisher_identifier'):
                pub_stats = self.publisher_identifier.get_statistics()
                print(f"  - Cache hits: {pub_stats['cache_hits']}")
                print(f"  - DOI matches: {pub_stats['doi_hits']}")
                print(f"  - Pattern matches: {pub_stats['known_pattern_hits']}")
                print(f"  - LLM queries: {pub_stats['llm_queries']}")
            
            # Web search statistics
            with_url = sum(1 for e in entries if e.get('publisher_url'))
            print(f"\nWeb Search Results:")
            print(f"  Papers with publisher URL: {with_url}")
            if hasattr(self, 'paper_search'):
                search_stats = self.paper_search.get_statistics()
                print(f"  - Searches performed: {search_stats['searches_performed']}")
                print(f"  - URLs found: {search_stats['urls_found']}")
                print(f"  - URLs validated: {search_stats['urls_validated']}")
                print(f"  - Cache hits: {search_stats['cache_hits']}")
            
            # Content extraction statistics
            extracted_abstracts = sum(1 for e in entries if e.get('extraction_method'))
            preprint_abstracts = sum(1 for e in entries if e.get('abstract_source') == 'preprint')
            if hasattr(self, 'content_extractor'):
                extract_stats = self.content_extractor.get_statistics()
                print(f"\nContent Extraction:")
                print(f"  Abstracts extracted: {extracted_abstracts}")
                print(f"  - Structured extractions: {extract_stats['structured_extractions']}")
                print(f"  - LLM extractions: {extract_stats['llm_extractions']}")
                print(f"  - From preprints: {preprint_abstracts}")
                print(f"  - Cache hits: {extract_stats['cache_hits']}")
            
            # Traditional enrichment statistics
            with_doi = sum(1 for e in entries if e.get('doi'))
            with_abstract = sum(1 for e in entries if e.get('abstract'))
            with_keywords = sum(1 for e in entries if e.get('keywords'))
            with_arxiv = sum(1 for e in entries if e.get('arxiv_url'))
            with_hal = sum(1 for e in entries if e.get('hal_url'))
            
            print(f"\nOverall Enrichment Results:")
            print(f"  Papers with DOI: {with_doi}")
            print(f"  Papers with abstract: {with_abstract}")
            print(f"  Papers with keywords: {with_keywords}")
            print(f"  Papers with ArXiv URL: {with_arxiv}")
            print(f"  Papers with HAL URL: {with_hal}")
            
            # Keyword source breakdown
            publisher_keywords = sum(1 for e in entries if e.get('keyword_source') == 'publisher')
            generated_keywords = sum(1 for e in entries if e.get('keyword_source') == 'generated')
            extracted_keywords = sum(1 for e in entries if e.get('keyword_source') == 'extracted')
            
            if with_keywords > 0:
                print(f"\nKeyword Sources:")
                print(f"  - From publisher: {publisher_keywords}")
                print(f"  - Extracted from URL: {extracted_keywords}")
                print(f"  - Generated by LLM: {generated_keywords}")
            
            # Preprint extraction statistics
            arxiv_abstracts = sum(1 for e in entries if e.get('extraction_method') == 'arxiv_fallback')
            hal_abstracts = sum(1 for e in entries if e.get('extraction_method') == 'hal_fallback')
            if arxiv_abstracts > 0 or hal_abstracts > 0:
                print(f"\nPreprint Abstracts:")
                print(f"  - From ArXiv: {arxiv_abstracts}")
                print(f"  - From HAL: {hal_abstracts}")
        
        print(f"\nTotal execution time: {elapsed:.2f} seconds")
        
        # Cache statistics
        if not self.args.dry_run and self.verbose >= 2:
            cache_stats = self.cache.get_cache_stats()
            print("\nCache statistics:")
            print(f"  API cache: {cache_stats['api_cache']['valid_entries']} entries")
            print(f"  Keyword cache: {cache_stats['keyword_cache']['total_entries']} entries")
            if hasattr(self, 'publisher_identifier') and self.publisher_identifier.cache:
                print(f"  Publisher cache: {len(self.publisher_identifier.cache)} entries")
            if hasattr(self, 'paper_search') and self.paper_search.cache:
                print(f"  Search cache: {len(self.paper_search.cache)} entries")
            if hasattr(self, 'content_extractor') and self.content_extractor.cache:
                print(f"  Extraction cache: {len(self.content_extractor.cache)} entries")
        
        self.display.header(f"Processing Complete! {Icons.CHECK}", Icons.SPARKLES)
        
        # Print detailed per-paper summary
        if self.verbose >= 1:
            self._print_detailed_summary(entries)
    
    def _print_detailed_summary(self, entries: List[Dict[str, Any]]) -> None:
        """Print detailed summary for each paper."""
        self.display.section("Detailed Paper Summary", Icons.INFO)
        
        for i, entry in enumerate(entries, 1):
            print(f"\n{Icons.BOOK} Paper {i}: {entry.get('title', 'Unknown')[:60]}...")
            print("=" * 80)
            
            # What was extracted from LaTeX
            print("\n📄 Extracted from LaTeX:")
            print(f"  - Authors: {len(entry.get('authors', []))} authors")
            print(f"  - Journal: {entry.get('journal', 'N/A')}")
            print(f"  - Year: {entry.get('year', 'N/A')}")
            print(f"  - Volume/Issue: {entry.get('volume', 'N/A')}/{entry.get('issue', 'N/A')}")
            if entry.get('arxiv_url'):
                print(f"  - ArXiv URL: {entry['arxiv_url']}")
            if entry.get('hal_url'):
                print(f"  - HAL URL: {entry['hal_url']}")
            
            # What was found on the internet
            print("\n🌐 Found on the Internet:")
            if entry.get('doi'):
                print(f"  - DOI: {entry['doi']} (source: {entry.get('doi_source', 'enricher')})")
            if entry.get('publisher_url'):
                print(f"  - Publisher URL: {entry['publisher_url'][:60]}...")
            if entry.get('publisher_identified'):
                print(f"  - Publisher: {entry['publisher_identified']} ({entry['publisher_platform']})")
            
            # Abstract extraction outcome
            print("\n📝 Abstract:")
            if entry.get('abstract'):
                abstract_source = entry.get('abstract_source', 'unknown')
                extraction_method = entry.get('extraction_method', 'unknown')
                print(f"  - Status: ✓ Extracted ({len(entry['abstract'])} characters)")
                print(f"  - Source: {abstract_source}")
                print(f"  - Method: {extraction_method}")
                print(f"  - Preview: {entry['abstract'][:100]}...")
            else:
                print("  - Status: ✗ Not found")
                print("  - Attempted sources:")
                if entry.get('publisher_url'):
                    print("    • Publisher website (failed - likely 403 error)")
                if not entry.get('arxiv_url') and not entry.get('hal_url'):
                    print("    • No preprint URLs available for fallback")
            
            # Keywords
            print("\n🏷️  Keywords:")
            if entry.get('keywords'):
                print(f"  - Status: ✓ Found ({len(entry['keywords'])} keywords)")
                print(f"  - Source: {entry.get('keyword_source', 'unknown')}")
                print(f"  - Keywords: {', '.join(entry['keywords'][:5])}")
                if len(entry['keywords']) > 5:
                    print(f"              ... and {len(entry['keywords']) - 5} more")
            else:
                print("  - Status: ✗ Not found")
                if not entry.get('abstract'):
                    print("  - Reason: No abstract available (required for keyword generation)")
            
            print("-" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Convert LaTeX journal papers to JSON with metadata enrichment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Use default settings
  %(prog)s -v 3                     # Maximum verbosity
  %(prog)s --dry-run                # Parse only, no API calls
  %(prog)s -c my_config.json        # Use custom config file
  %(prog)s --no-keywords            # Disable keyword generation
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        default='input_data/list_of_papers.tex',
        help='Input LaTeX file (default: input_data/list_of_papers.tex)'
    )
    
    parser.add_argument(
        '-o', '--output',
        default='output_data/papers.json',
        help='Output JSON file (default: output_data/papers.json)'
    )
    
    parser.add_argument(
        '-c', '--config',
        help='Configuration file (default: config.json if exists)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        type=int,
        choices=[0, 1, 2, 3],
        default=2,
        help='Verbosity level (0-3, default: 2)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Parse only, no web requests'
    )
    
    parser.add_argument(
        '--no-keywords',
        action='store_true',
        help='Disable keyword generation'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing output file'
    )
    
    parser.add_argument(
        '--no-web-search',
        action='store_true',
        help='Disable web search for papers'
    )
    
    parser.add_argument(
        '--no-content-extraction',
        action='store_true',
        help='Disable content extraction from URLs'
    )
    
    parser.add_argument(
        '--no-publisher-id',
        action='store_true',
        help='Disable publisher identification'
    )
    
    args = parser.parse_args()
    
    # Check for default config.json if not specified
    if not args.config:
        # Try parent directory first (project root)
        default_config = Path(__file__).parent.parent / 'config.json'
        if default_config.exists():
            args.config = str(default_config)
        else:
            # Try src directory
            default_config = Path(__file__).parent / 'config.json'
            if default_config.exists():
                args.config = str(default_config)
    
    # Check if output exists and not forcing
    output_path = Path(args.output)
    if output_path.exists() and not args.force:
        response = input(f"Output file {output_path} already exists. Overwrite? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(0)
    
    # Run the processor
    processor = JournalProcessor(args)
    processor.run()


if __name__ == "__main__":
    main()