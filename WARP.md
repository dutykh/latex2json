# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

**latex2json** is a Python toolkit for parsing academic LaTeX files (collaborators, bibliography entries) and converting them to enriched JSON format with AI-powered metadata enhancement, geocoding, and web search capabilities.

The system uses dual LLM support (Gemini/Claude with automatic fallback), multi-source API enrichment (CrossRef, Google Scholar, ORCID, publisher APIs), and extensive caching to efficiently process academic data.

## Common Development Commands

### Setup and Installation
```bash
# Install dependencies (uses pip, not pnpm - this is a Python project)
make install
# or
pip install -r requirements.txt

# Create config file from example
cp config.json.example config.json
# Edit config.json to add your API keys
```

### Running Parsers
```bash
# Collaborator extraction
python3 src/collab2json.py <input_file.tex>
python3 src/collab2json.py collab_excerpts.tex -v 3  # with high verbosity
python3 src/collab2json.py collab_excerpts.tex --skip-geocoding  # faster

# Book chapter extraction
python3 src/chpts2json.py -v 2
python3 src/chpts2json.py --dry-run -v 3  # test parsing without API calls

# Journal paper extraction
python3 src/pub2json.py -v 2
python3 src/pub2json.py --dry-run -v 3  # test parsing without API calls

# Using Makefile shortcuts
make run FILE=input_file.tex    # Run collaborator extraction
make run-chapters               # Run chapter parser
make run-papers                 # Run paper parser
make test                       # Test with sample data
```

### Code Quality
```bash
# Linting and formatting
make lint        # Run ruff linting
make format      # Auto-format with ruff
ruff check . --fix && ruff format .

# Security checks
make check       # Run safety vulnerability scan
safety check --bare
```

### Cache Management
```bash
# Cache files are auto-generated in cache/ directory
make cache-info              # Show cache statistics
make clean-cache             # Remove all cache files
make clean-search-cache      # Clear only search cache
make clean-publisher-cache   # Clear publisher & paper search caches
make clean-llm-cache         # Clear all LLM caches
make clean-all              # Remove all caches + Python bytecode

# Manual cache clearing for specific operations
rm cache/.llm_parse_cache.json           # Re-parse LaTeX entries
rm cache/.paper_search_cache.json        # Re-search for papers
rm cache/.content_extraction_cache.json  # Re-extract content from URLs
```

## Code Architecture

### Main Scripts (`src/`)
- **`collab2json.py`**: Collaborator parser - LLM-based extraction with web search for researcher profiles, geocoding, and ORCID verification
- **`chpts2json.py`**: Book chapter parser - processes `etaremune` bibliography entries with AI keyword generation
- **`pub2json.py`**: Journal paper parser - handles year-sectioned journals with publisher identification and double-layer abstract extraction

### Core Modules (`modules/`)

#### Configuration & Management
- **`config_manager.py`**: Centralized configuration handler - loads config.json, provides API key access
- **`cache_manager.py`**: TTL-based cache management - handles expiry, statistics, and multiple cache types
- **`display_utils.py`**: Terminal output formatting - colored output, progress indicators, structured displays

#### LLM Integration (Unified Interface Pattern)
- **`llm_provider.py`**: Abstract LLM interface with automatic fallback - supports both Anthropic and Gemini
- **`gemini_client.py`**: Gemini-specific client with timeout protection and safety filter handling
- **`llm_parser.py`**: Alternative LLM parser for LaTeX entries
- **`llm_enricher.py`**: LLM-based metadata extraction from web content

#### Parsing & Extraction
- **`latex_parser.py`**: Core LaTeX parsing utilities - handles `etaremune` environments, cleans LaTeX commands
- **`journal_parser.py`**: Journal-specific parser - extracts DOIs, article numbers (numeric, alphanumeric, PLOS format), handles year sections
- **`publisher_identifier.py`**: LLM-powered publisher identification from journal names
- **`paper_content_extractor.py`**: Extracts abstracts/content from publisher URLs and preprint servers

#### Web Search & Enrichment
- **`web_search.py`**: Advanced researcher profile search with LLM-based extraction and ORCID verification
- **`simple_search.py`**: Fallback search when Google Custom Search is unavailable
- **`paper_web_search.py`**: Paper-specific search with smart single-query strategy and blocked publisher handling
- **`llm_search_engines.py`**: Academic search engine integrations
- **`metadata_enricher.py`**: Multi-source metadata enrichment (CrossRef, Google Scholar, publisher APIs)
- **`keyword_generator.py`**: AI-powered keyword generation (only when abstracts are available)

### Directory Structure
```
latex2json/
├── src/                     # Main entry point scripts
├── modules/                 # Reusable modular components
├── input_data/             # LaTeX input files
├── output_data/            # Generated JSON files
├── cache/                  # Auto-generated cache files (.*.json)
├── config.json.example     # Configuration template with API keys
└── requirements.txt        # Python dependencies
```

## Key Design Patterns

### 1. Dual LLM Fallback System
All LLM operations support automatic fallback (Gemini → Claude) with timeout protection (30s). Safety filters and rate limits trigger automatic provider switching.

### 2. Multi-Layer Caching
- **Geocoding cache**: Location results from Nominatim
- **Search cache**: Web search results and ORCID verification
- **API cache**: CrossRef, publisher API responses
- **LLM caches**: Parse results, enrichments, location extraction, profile analysis
- **Content cache**: Extracted abstracts and paper content

Each cache type has separate TTL management and can be cleared independently.

### 3. Modular Pipeline Architecture
Each parser follows: **LaTeX → LLM Parse → Web Search → API Enrichment → Keyword Generation → JSON Output**

Modules are loosely coupled - each can be disabled via CLI flags (`--dry-run`, `--no-keywords`, `--skip-search`, etc.)

### 4. Verbosity-Driven Debugging
All scripts support `-v 0-3` verbosity levels:
- **0**: Silent (minimal output)
- **1**: Basic progress
- **2**: Detailed progress with entry-by-entry updates
- **3**: Debug mode with full LaTeX content, cache hits/misses, API queries, LLM responses

Use `-v 3` when debugging parsing issues or API failures.

## Configuration (`config.json`)

### Required API Keys
- **`anthropic`**: Claude API key (for LLM fallback and keyword generation)
- **`google`**: Google API key (for Gemini primary LLM)
- **`google_cx`**: Google Custom Search Engine ID (for web search)
- **`crossref_email`**: Email for CrossRef API (polite pool access)

### Key Settings
- **`llm_config.primary_llm`**: Choose `"anthropic"` or `"google"` as primary
- **`search_settings.quality_mode`**: Enable comprehensive targeted searches
- **`paper_search_settings.use_smart_single_search`**: Reduce API calls by 67%
- **`extraction.use_preprint_fallback`**: Fall back to ArXiv/HAL when publisher blocks
- **`caching.api_cache_expiry_days`**: TTL for different cache types

## Development Workflow

### When Adding New Features
1. Add module to `modules/` directory
2. Follow existing pattern: accept `config`, `verbose` parameters in `__init__`
3. Implement caching via `CacheManager` for any external API calls
4. Add timeout protection for all network operations (30s standard)
5. Support verbosity levels 0-3 for debugging
6. Update main script in `src/` to integrate new module

### When Fixing Parsing Issues
1. Run with `--dry-run -v 3` to see raw LaTeX and parsed results
2. Check `cache/.llm_parse_cache.json` for LLM responses
3. Clear specific cache to force re-parsing: `rm cache/.llm_parse_cache.json`
4. Test both LLM parsing and regex fallback paths

### When Debugging API Issues
1. Use `-v 3` to see all API queries and responses
2. Check blocked publishers in `cache/.paper_search_cache.json`
3. Verify API keys in `config.json` are valid and have quota
4. Check for 403 errors (publisher blocking) - system handles gracefully

## Special Considerations

### Profile Verification System
The collaborator parser includes strict validation to ensure data quality:
- Email addresses must contain parts of researcher's name
- ResearchGate profiles validated by name matching in profile ID
- ORCID verification via ORCID API
- Use `--verify` flag to re-verify and clean existing data

### Article Number Extraction
The journal parser handles all formats:
- Short numeric (e.g., "132")
- Long numeric (e.g., "106518")
- Alphanumeric (e.g., "e0305534" for PLOS)

If article numbers are missing, clear parse cache and re-run.

### Publisher Access Limitations
Many publishers (Elsevier, Springer, Taylor & Francis) block automated access. The system handles this via:
- Automatic detection and caching of blocked publishers
- Fallback to CrossRef API for metadata
- Preprint URL extraction from LaTeX comments
- Double-layer extraction (publisher → preprint fallback)

### JSON Repair Logic
LLM responses may return malformed JSON. The system automatically:
- Fixes single quotes, trailing commas, unquoted keys
- Removes JavaScript-style comments
- Extracts valid JSON from mixed text responses
- Falls back to regex parsing when both LLMs fail

## Testing

No formal test suite exists. Testing is done via:
```bash
make test           # Test collaborator parser with sample data
make test-papers    # Test paper parser with sample data
```

For development testing, use `--dry-run` flags to test parsing without making API calls.

## When to Clear Caches

- **Article numbers missing**: `rm cache/.llm_parse_cache.json`
- **Wrong location data**: `make clean-search-cache`
- **Blocked publishers changing**: `make clean-publisher-cache`
- **LLM responses incorrect**: `make clean-llm-cache`
- **Full re-process**: `make clean-cache` (clears all except output files)
