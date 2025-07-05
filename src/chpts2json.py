#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chpts2json.py - LaTeX book chapters to JSON converter with metadata enrichment.

This script parses LaTeX bibliography entries from etaremune environments,
enriches them with metadata from publisher APIs, and optionally generates
keywords using Claude API.

Author: Dr. Denys Dutykh (Khalifa University of Science and Technology, Abu Dhabi, UAE)
Date: 2025-07-01

Usage:
    python chpts2json.py [options]
    
Examples:
    python chpts2json.py                    # Use defaults
    python chpts2json.py -v 3               # Maximum verbosity
    python chpts2json.py --dry-run          # Parse only, no API calls
    python chpts2json.py --no-keywords      # Disable keyword generation
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
from modules.latex_parser import LatexBibliographyParser
from modules.cache_manager import CacheManager
from modules.config_manager import ConfigManager
from modules.metadata_enricher import MetadataEnricher
from modules.keyword_generator import KeywordGenerator


class ChapterProcessor:
    """Main processor for converting LaTeX chapters to JSON."""
    
    def __init__(self, args: argparse.Namespace):
        """Initialize the processor with command-line arguments."""
        self.args = args
        self.verbose = args.verbose
        
        # Set up paths
        self.base_dir = Path(__file__).parent.parent.resolve()  # Parent of src/
        self.input_file = Path(args.input).resolve()
        self.output_file = Path(args.output).resolve()
        self.config_file = Path(args.config).resolve() if args.config else None
        
        # Cache files
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
        
        # Initialize cache manager
        cache_expiry = self.config.get('preferences.cache_expiry_days', 7)
        self.cache = CacheManager(
            self.api_cache_file, 
            self.keyword_cache_file,
            expiry_days=cache_expiry,
            verbose=self.verbose
        )
        
        # Initialize parser with config for LLM support
        self.parser = LatexBibliographyParser(verbose=self.verbose, config=self.config)
        
        # Initialize enricher (only if not dry run)
        if not self.args.dry_run:
            self.enricher = MetadataEnricher(self.config, self.cache, verbose=self.verbose)
        
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
        print("=" * 60)
        print("chpts2json.py - LaTeX to JSON Converter")
        print("=" * 60)
        
        if self.verbose >= 2:
            print(f"[INFO] Input file: {self.input_file}")
            print(f"[INFO] Output file: {self.output_file}")
            if self.config_file:
                print(f"[INFO] Config file: {self.config_file}")
            print(f"[INFO] Verbosity level: {self.verbose}")
            
            if self.args.dry_run:
                print("[INFO] DRY RUN MODE - No API calls will be made")
            if self.args.no_keywords:
                print("[INFO] Keyword generation disabled")
    
    def _parse_latex_file(self) -> List[Dict[str, Any]]:
        """Parse the LaTeX file."""
        print(f"\n[INFO] Reading LaTeX file: {self.input_file.name}")
        
        content = self.input_file.read_text(encoding='utf-8')
        
        print("[INFO] Parsing LaTeX content...")
        entries = self.parser.parse_file(content)
        
        print(f"[INFO] Found {len(entries)} book chapter entries")
        
        if self.verbose >= 1 and entries:
            print("\n[INFO] Parsed entries:")
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
                print(f"  {i:3d}. [{year}] {author_info} - {title}")
        
        return entries
    
    async def _enrich_entries(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enrich entries with metadata."""
        # Clear expired cache entries first
        if self.config.get('preferences.cache_responses', True):
            removed, remaining = self.cache.clear_expired()
            if self.verbose >= 2 and removed > 0:
                print(f"[CACHE] Cleaned {removed} expired entries")
        
        # Enrich with metadata
        print("\n[INFO] Starting metadata enrichment...")
        entries = await self.enricher.enrich_entries(entries)
        
        # Generate keywords if enabled
        if not self.args.no_keywords:
            print("\n[INFO] Checking for keyword generation...")
            await self.keyword_gen.generate_keywords_for_entries(entries)
        
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
        
        return entries
    
    def _generate_output(self, entries: List[Dict[str, Any]]) -> None:
        """Generate the output JSON file."""
        print(f"\n[INFO] Writing output to: {self.output_file}")
        
        # Create output directory if needed
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare output data
        output_data = {
            "book_chapters": entries
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
        
        return {
            'total_chapters': len(entries),
            'enriched_count': enriched_count,
            'keyword_stats': keyword_stats,
            'generation_date': datetime.now().isoformat(),
            'script_version': '1.0.0',
            'dry_run': self.args.dry_run
        }
    
    def _print_summary(self, entries: List[Dict[str, Any]], start_time: float) -> None:
        """Print processing summary."""
        elapsed = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("PROCESSING SUMMARY")
        print("=" * 60)
        
        print(f"Total chapters processed: {len(entries)}")
        
        if not self.args.dry_run:
            # Enrichment statistics
            with_doi = sum(1 for e in entries if e.get('doi'))
            with_abstract = sum(1 for e in entries if e.get('abstract'))
            with_keywords = sum(1 for e in entries if e.get('keywords'))
            
            print(f"Chapters with DOI: {with_doi}")
            print(f"Chapters with abstract: {with_abstract}")
            print(f"Chapters with keywords: {with_keywords}")
            
            # Keyword source breakdown
            publisher_keywords = sum(1 for e in entries if e.get('keyword_source') == 'publisher')
            generated_keywords = sum(1 for e in entries if e.get('keyword_source') == 'generated')
            
            if with_keywords > 0:
                print(f"  - From publisher: {publisher_keywords}")
                print(f"  - Generated: {generated_keywords}")
        
        print(f"\nTotal execution time: {elapsed:.2f} seconds")
        
        # Cache statistics
        if not self.args.dry_run and self.verbose >= 2:
            cache_stats = self.cache.get_cache_stats()
            print("\nCache statistics:")
            print(f"  API cache: {cache_stats['api_cache']['valid_entries']} entries")
            print(f"  Keyword cache: {cache_stats['keyword_cache']['total_entries']} entries")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Convert LaTeX book chapters to JSON with metadata enrichment",
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
        default='input_data/list_of_chpts.tex',
        help='Input LaTeX file (default: input_data/list_of_chpts.tex)'
    )
    
    parser.add_argument(
        '-o', '--output',
        default='output_data/bkchpts.json',
        help='Output JSON file (default: output_data/bkchpts.json)'
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
    
    args = parser.parse_args()
    
    # Check for default config.json if not specified
    if not args.config:
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
    processor = ChapterProcessor(args)
    processor.run()


if __name__ == "__main__":
    main()