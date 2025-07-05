# LaTeX to JSON Converter Suite

A comprehensive Python toolkit for parsing LaTeX files containing academic data (collaborators, bibliography entries) and converting them to enriched JSON format with optional geocoding and AI-powered metadata enhancement.

## Features

### Core Capabilities
- **Collaborator Parser** (`src/collab2json.py`):
  - LLM-based intelligent parsing using Google Gemini with automatic Claude fallback
  - Web search for researcher profiles (ResearchGate, Google Scholar, ORCID, LinkedIn)
  - Enhanced university location search with LLM extraction for precise geocoding
  - Strict profile validation with ORCID API verification
  - Email and profile ownership validation to ensure correct attribution
  - Profile re-verification mode to clean existing data
  - Geocodes locations using OpenStreetMap Nominatim API
  - Multi-level caching system for all operations

- **Book Chapter Parser** (`src/chpts2json.py`):
  - Parses LaTeX bibliography entries from `etaremune` environments
  - Asynchronous metadata enrichment from multiple sources
  - AI-powered keyword generation using Claude API
  - Advanced caching with TTL (time-to-live) management
  - Multiple fallback mechanisms for robust data extraction

- **Journal Paper Parser** (`src/pub2json.py`):
  - Parses LaTeX journal entries from `etaremune` environments with year sections
  - Extracts preprint URLs from LaTeX comments (ArXiv, HAL, ResearchGate)
  - LLM-powered parsing for complex entries
  - Comprehensive metadata enrichment from multiple sources
  - Detailed verbose output for debugging and monitoring

### Advanced Features
- **LLM Integration**: 
  - Dual LLM support with automatic fallback (Gemini → Claude)
  - Intelligent parsing for complex LaTeX entries
  - AI-powered keyword generation
  - LLM-based location extraction from search results
- **Profile Verification**:
  - ORCID API integration for name verification
  - ResearchGate profile validation
  - Email ownership verification
  - Homepage validation
- **Multi-source Enrichment**: CrossRef, Google Scholar, and publisher-specific APIs
- **Modular Architecture**: Organized modules for caching, configuration, parsing, and enrichment
- **Comprehensive Error Handling**: Graceful failure with detailed logging
- **Performance Optimized**: Asynchronous processing and intelligent caching
- **Verbose Monitoring**: Detailed processing information at multiple verbosity levels

## Prerequisites

- Python 3.8+ (3.6+ for basic functionality)
- Required Python packages (see requirements.txt)

### Installation

Install all dependencies:
```bash
pip install -r requirements.txt
```

For basic collaborator parsing only:
```bash
pip install requests
```

For full functionality including book chapter parsing:
```bash
pip install requests beautifulsoup4 aiohttp pylatexenc scholarly crossref-commons anthropic
```

### Configuration

For advanced features (API enrichment, keyword generation, LLM fallback), copy and configure:
```bash
cp config.json.example config.json
# Edit config.json with your API keys and preferences
```

Key configuration options:
- **API Keys**: Google (for Gemini & Custom Search), Anthropic (for Claude fallback)
- **LLM Settings**: Primary model (Gemini) and fallback model (Claude)
- **Search Settings**: Google Custom Search Engine ID for web search
- **Cache Settings**: TTL for different cache types

## Usage

**Note**: All main scripts have been moved to the `src/` directory.

### Collaborator Parser (src/collab2json.py)

1. Place your input `.tex` file inside the `input_data/` directory.
2. Run the script:

```bash
python3 src/collab2json.py <your_input_file.tex>
```

Example:
```bash
python3 src/collab2json.py collab_excerpts.tex
```

Optional arguments:
- `--skip-geocoding`: Disable geocoding for faster processing
- `--skip-search`: Disable web search for researcher profiles
- `-v, --verbose`: Verbosity level (0-3)
- `--verify`: Re-verify existing profiles using strict validation rules
- `--output`: Specify output JSON filename
- `--no-cache`: Disable cache usage
- `--clear-cache`: Clear all cache files before running
- `--clear-search-cache`: Clear only search cache

Examples:
```bash
# Basic extraction with high verbosity
python3 src/collab2json.py collab_excerpts.tex -v 3

# Skip geocoding for faster processing
python3 src/collab2json.py collab_excerpts.tex --skip-geocoding

# Verify existing data (removes invalid profiles)
python3 src/collab2json.py collab_excerpts.tex --output collaborators.json --verify

# Clear cache and run fresh
python3 src/collab2json.py collab_excerpts.tex --clear-cache

# Custom output file
python3 src/collab2json.py collab_excerpts.tex --output my_collaborators.json
```

### Book Chapter Parser (src/chpts2json.py)

Parse LaTeX bibliography entries with metadata enrichment:

```bash
python3 src/chpts2json.py [options]
```

Options:
- `-i, --input`: Input LaTeX file (default: `input_data/list_of_chpts.tex`)
- `-o, --output`: Output JSON file (default: `output_data/bkchpts.json`)
- `-c, --config`: Configuration file (default: `config.json`)
- `-v, --verbose`: Verbosity level 0-3 (default: 1)
- `--dry-run`: Parse only, no API calls
- `--no-keywords`: Disable keyword generation
- `--force`: Overwrite existing output file

Examples:
```bash
# Basic usage with defaults
python3 chpts2json.py

# Custom input/output with high verbosity
python3 chpts2json.py -i input_data/my_chapters.tex -o output_data/my_output.json -v 2

# Dry run to test parsing without API calls
python3 src/chpts2json.py --dry-run -v 3

# Process without keyword generation
python3 src/chpts2json.py --no-keywords
```

### Journal Paper Parser (src/pub2json.py)

Parse LaTeX journal paper entries with advanced metadata enrichment:

```bash
python3 src/pub2json.py [options]
```

Options:
- `-i, --input`: Input LaTeX file (default: `input_data/list_of_papers.tex`)
- `-o, --output`: Output JSON file (default: `output_data/pubs.json`)
- `-c, --config`: Configuration file (default: `config.json`)
- `-v, --verbose`: Verbosity level 0-3 (default: 2)
- `--dry-run`: Parse only, no API calls
- `--no-keywords`: Disable keyword generation
- `--force`: Overwrite existing output file

Examples:
```bash
# Basic usage with defaults
python3 src/pub2json.py

# Maximum verbosity to see detailed processing steps
python3 src/pub2json.py -v 3

# Dry run with verbose output to test parsing
python3 src/pub2json.py --dry-run -v 3 --force

# Process without keyword generation
python3 src/pub2json.py --no-keywords

# Custom configuration file
python3 src/pub2json.py -c my_config.json
```

#### Verbosity Levels:
- **0**: Silent mode - minimal output
- **1**: Basic progress - shows entry count and summary
- **2**: Detailed progress - shows each entry being processed
- **3**: Debug mode - shows full processing details including:
  - LaTeX content preview
  - Parsing steps (LLM or regex)
  - Extracted metadata fields
  - Cache hits/misses
  - API search queries
  - Enrichment results

### Using Makefile

A Makefile is provided for convenient command execution:

```bash
# Install dependencies
make install

# Run collaborator extraction
make run FILE=collab_excerpts.tex
make run-full          # Run with full dataset

# Run chapter/paper extraction
make run-chapters      # Process book chapters
make run-papers        # Process journal papers

# Testing and development
make test              # Run test with sample data
make test-api          # Test Google API configuration
make lint              # Run code linting
make format            # Format code

# Clean cache files
make clean             # Clean all cache files
make clean-cache       # Clean all cache files (alias)
make clean-search-cache # Clean only search cache file
make clean-all         # Clean everything including Python cache
```

## Project Structure

```
latex2json/
├── src/                  # Main scripts (moved to src/ directory)
│   ├── collab2json.py      # Enhanced collaborator parser
│   ├── chpts2json.py       # Book chapter parser
│   └── pub2json.py         # Journal paper parser
├── modules/              # Core functionality modules
│   ├── cache_manager.py     # Cache management with TTL
│   ├── config_manager.py    # Configuration handling
│   ├── display_utils.py     # Terminal display formatting
│   ├── gemini_client.py     # Gemini LLM client with fallback
│   ├── journal_parser.py    # Journal-specific LaTeX parser
│   ├── keyword_generator.py # AI-powered keyword generation
│   ├── latex_parser.py      # LaTeX parsing utilities
│   ├── llm_enricher.py      # LLM-based metadata extraction
│   ├── llm_parser.py        # Alternative LLM parser
│   ├── llm_provider.py      # Unified LLM provider interface
│   ├── llm_search_engines.py # Academic search engines
│   ├── metadata_enricher.py # Metadata fetching from APIs
│   ├── simple_search.py     # Simple web search implementation
│   └── web_search.py        # Advanced web search with LLM
├── input_data/           # Input LaTeX files
│   ├── collab_excerpts.tex  # Sample collaborator data
│   ├── list_of_chpts.tex    # Sample book chapters
│   ├── list_of_papers.tex   # Sample journal papers
│   └── test_two.tex         # Additional test data
├── output_data/          # Generated JSON files
├── config.json.example   # Configuration template
├── requirements.txt      # Python dependencies
├── CLAUDE.md            # AI assistant instructions
├── PRD.md               # Product requirements document
├── LICENSE              # GNU GPLv3 license
└── README.md            # This file
```

### Cache Files (auto-generated, git-ignored)
- `.geocode_cache.json`: Geocoding results cache
- `.search_cache.json`: Web search results and ORCID verification cache
- `.api_cache.json`: API response cache (CrossRef, publishers)
- `.keyword_cache.json`: Generated keywords cache
- `.llm_cache.json`: Basic LLM parsing cache
- `.llm_unified_parse_cache.json`: Unified parser cache (Gemini/Claude)
- `.llm_unified_location_cache.json`: LLM location extraction cache
- `.llm_unified_profile_cache.json`: LLM profile analysis cache
- `.llm_websearch_cache.json`: LLM web search extraction cache
- `.llm_enricher_cache.json`: LLM enrichment response cache
- `.llm_parse_cache.json`: LLM parsing results cache

## Development

### Code Quality

The project uses modern Python development tools:

```bash
# Run linting and auto-fix
ruff check . --fix
ruff format .

# Security vulnerability scan
safety check -r requirements.txt

# Combined quality check (recommended before commits)
ruff check . && safety check --bare
```

### Testing

```bash
# Test collaborator parsing
python3 src/collab2json.py collab_excerpts.tex --skip-geocoding

# Test book chapter parsing (dry run)
python3 src/chpts2json.py --dry-run -v 2
```

## API Integrations

The project integrates with multiple APIs for enriched metadata:

- **OpenStreetMap Nominatim**: Geocoding for collaborator locations
- **Google Gemini API**: Primary LLM for intelligent parsing and extraction
- **Claude API**: Fallback LLM for parsing and keyword generation
- **Google Custom Search API**: Web search for researcher profiles
- **ORCID API**: Verification of ORCID identifiers
- **CrossRef API**: DOI resolution and metadata
- **Google Scholar**: Academic publication search
- **Publisher APIs**: Direct metadata from Springer, Elsevier, etc.
- **ArXiv API**: Preprint metadata and abstracts
- **HAL API**: French research archive metadata

## Recent Improvements (2025-07-05)

### LLM Robustness Enhancements
- **JSON Repair Logic**: Automatically fixes common JSON formatting issues from LLM responses:
  - Handles single quotes, trailing commas, unquoted keys
  - Removes JavaScript-style comments
  - Extracts valid JSON from mixed text responses
- **Improved Error Handling**: Shows raw LLM responses for debugging when parsing fails
- **Retry Mechanism**: Falls back to simpler prompts when complex extraction fails
- **Timeout Protection**: All LLM calls now have 30-second timeouts to prevent hanging
- **Regex Fallback**: When both LLMs fail, falls back to regex-based extraction

### Location Enhancement Fixes
- **Smart University Detection**: Distinguishes between departments and universities:
  - Recognizes patterns like "Universiti", "Universidad", "Rechenzentrum"
  - Moves department names to appropriate fields
  - Extracts real university names from full affiliation strings
- **Enhanced Validation**: Prevents bad location data from being cached:
  - Detects nonsensical city names (e.g., "Due To", "N/A")
  - Validates city name length and format
  - Protects existing good data from being overwritten
- **Cache Quality Control**: 
  - Removes corrupted entries automatically
  - Validates addresses before caching
  - New `make clean-search-cache` command for targeted cache clearing

### Profile Verification System
- **Strict Validation Rules**: Implemented comprehensive profile verification:
  - Email addresses must contain parts of researcher's name
  - ResearchGate profiles validated by name matching in profile ID
  - ORCID verification via API to confirm ownership
  - LinkedIn profile validation
  - Homepage validation (removes profile site URLs without valid profiles)

- **Verification Mode**: New `--verify` flag to re-verify existing data:
  - Removes incorrectly attributed profiles
  - Validates ORCID IDs against ORCID API
  - Ensures email addresses belong to the researcher
  - Tracks what was removed for transparency

### LLM Integration Improvements
- **Automatic Fallback**: Gemini → Claude fallback when safety filters trigger
- **Unified Parser**: Single interface for multiple LLM providers
- **Enhanced Caching**: Separate caches for different LLM operations
- **Clear JSON Instructions**: LLM prompts now include explicit JSON formatting rules
- **Better Department/University Distinction**: Improved prompts with examples

### Data Quality Improvements
- **Quality Over Quantity**: Prefers missing data over incorrect data
- **Profile Ownership**: Validates that profiles belong to the correct researcher
- **City Extraction**: Fixed issues with Malaysian universities and regions
- **University Recognition**: Now recognizes international patterns:
  - Malaysian: "Universiti"
  - German: "Hochschule", "Rechenzentrum"
  - Spanish: "Universidad"
  - French: "Université", "École"

## Previous Improvements (2025-07-02)

- **Enhanced Verbose Output**: Added comprehensive verbose logging to `pub2json.py` showing:
  - Each LaTeX item being processed with preview
  - Parsing method used (LLM vs regex)
  - Extracted metadata fields in real-time
  - Cache hit/miss information
  - API search progress and results
  - Detailed error messages and fallback attempts
  
- **Improved Journal Parser**: The `journal_parser.py` module now provides:
  - Step-by-step parsing progress
  - LaTeX content preview for each entry
  - Statistics on parsed entries (DOIs, URLs, preprints)
  - Clear indication of successful vs failed parsing

- **Cache Management**: Clear cache files using:
  ```bash
  rm -f .api_cache.json .keyword_cache.json
  ```

- **Publisher Access Handling**: Updated to handle publisher access restrictions:
  - Uses browser-like headers to reduce blocking
  - Gracefully handles HTTP 403 errors from publishers
  - Falls back to alternative sources (CrossRef, Google Scholar, ArXiv)
  - Clear messaging about access restrictions

## Known Limitations

- **Publisher Website Access**: Many academic publishers (e.g., Taylor & Francis, Elsevier, Springer) block automated access to their websites, resulting in HTTP 403 errors. The script handles this gracefully by:
  - Using metadata from CrossRef API (DOI-based)
  - Searching Google Scholar as a fallback
  - Extracting preprint URLs from LaTeX comments (ArXiv, HAL)
  - Using LLM-based enrichment when available
  
  To maximize metadata extraction success:
  - Ensure DOIs are included in your LaTeX entries when possible
  - Include preprint URLs in comments
  - Consider using publisher API keys if available

## Configuration Options

The `config.json` file supports extensive customization:

```json
{
  "api_keys": {
    "google": "your-google-api-key",
    "google_cx": "your-custom-search-engine-id",
    "anthropic": "your-claude-api-key",
    "crossref": "your-email@example.com"
  },
  "llm_config": {
    "primary_model": "gemini-2.5-flash",
    "fallback_model": "claude-3-haiku-20240307",
    "auto_fallback": true
  },
  "search_settings": {
    "quality_mode": true,
    "use_llm_extraction": true,
    "verify_profiles": true
  },
  "caching": {
    "api_cache_expiry_days": 30,
    "keyword_cache_expiry_days": 90,
    "llm_cache_expiry_days": 90
  }
}
```

## Author

- **Dr. Denys Dutykh** - *Initial work* - [Khalifa University of Science and Technology, Abu Dhabi, UAE]

## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for more details.
