# LaTeX to JSON Converter Suite

A comprehensive Python toolkit for parsing LaTeX files containing academic data (collaborators, bibliography entries) and converting them to enriched JSON format with optional geocoding and AI-powered metadata enhancement.

## Features

### Core Capabilities
- **Collaborator Parser** (`latex_to_json.py`):
  - Extracts names and affiliations from LaTeX `description` environments
  - Cleans LaTeX-specific commands and characters (e.g., `\textsc`, `\'{e}`)
  - Geocodes locations using OpenStreetMap Nominatim API
  - Intelligent caching system for geocoding results

- **Book Chapter Parser** (`chpts2json.py`):
  - Parses LaTeX bibliography entries from `etaremune` environments
  - Asynchronous metadata enrichment from multiple sources
  - AI-powered keyword generation using Claude API
  - Advanced caching with TTL (time-to-live) management
  - Multiple fallback mechanisms for robust data extraction

### Advanced Features
- **LLM Integration**: Optional Claude API integration for complex parsing and keyword generation
- **Multi-source Enrichment**: CrossRef, Google Scholar, and publisher-specific APIs
- **Modular Architecture**: Organized modules for caching, configuration, parsing, and enrichment
- **Comprehensive Error Handling**: Graceful failure with detailed logging
- **Performance Optimized**: Asynchronous processing and intelligent caching

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

For advanced features (API enrichment, keyword generation), copy and configure:
```bash
cp config.json.example config.json
# Edit config.json with your API keys and preferences
```

## Usage

### Collaborator Parser (latex_to_json.py)

1. Place your input `.tex` file inside the `input_data/` directory.
2. Run the script:

```bash
python3 latex_to_json.py <your_input_file.tex>
```

Example:
```bash
python3 latex_to_json.py collab_excerpts.tex
```

Optional arguments:
- `--skip-geocoding`: Disable geocoding for faster processing

```bash
python3 latex_to_json.py collab_excerpts.tex --skip-geocoding
```

### Book Chapter Parser (chpts2json.py)

Parse LaTeX bibliography entries with metadata enrichment:

```bash
python3 chpts2json.py [options]
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
python3 chpts2json.py --dry-run -v 3

# Process without keyword generation
python3 chpts2json.py --no-keywords
```

## Project Structure

```
latex2json/
├── latex_to_json.py      # Collaborator parser with geocoding
├── chpts2json.py         # Book chapter parser with API enrichment
├── modules/              # Core functionality modules
│   ├── cache_manager.py     # Cache management with TTL
│   ├── config_manager.py    # Configuration handling
│   ├── keyword_generator.py # AI-powered keyword generation
│   ├── latex_parser.py      # LaTeX parsing utilities
│   ├── llm_enricher.py      # LLM-based metadata extraction
│   ├── llm_parser.py        # Alternative LLM parser
│   ├── llm_search_engines.py # Academic search engines
│   └── metadata_enricher.py # Metadata fetching from APIs
├── input_data/           # Input LaTeX files
│   ├── collab_excerpts.tex  # Sample collaborator data
│   ├── list_of_chpts.tex    # Sample book chapters
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
- `.api_cache.json`: API response cache
- `.keyword_cache.json`: Generated keywords cache
- `.llm_enricher_cache.json`: LLM response cache

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
python3 latex_to_json.py collab_excerpts.tex --skip-geocoding

# Test book chapter parsing (dry run)
python3 chpts2json.py --dry-run -v 2
```

## API Integrations

The project integrates with multiple APIs for enriched metadata:

- **OpenStreetMap Nominatim**: Geocoding for collaborator locations
- **CrossRef API**: DOI resolution and metadata
- **Google Scholar**: Academic publication search
- **Publisher APIs**: Direct metadata from Springer, Elsevier, etc.
- **Claude API**: AI-powered parsing and keyword generation

## Configuration Options

The `config.json` file supports extensive customization:

```json
{
  "api_keys": {
    "anthropic": "your-claude-api-key",
    "crossref": "your-email@example.com"
  },
  "llm_preferences": {
    "use_llm_parser": false,
    "use_llm_enricher": true,
    "model": "claude-3-haiku-20240307"
  },
  "caching": {
    "api_cache_expiry_days": 30,
    "keyword_cache_expiry_days": 90
  }
}
```

## Author

- **Dr. Denys Dutykh** - *Initial work* - [Khalifa University of Science and Technology, Abu Dhabi, UAE]

## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for more details.
