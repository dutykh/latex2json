# Makefile for latex2json project
# Author: Dr. Denys Dutykh
# Date: 2025-01-04

.PHONY: help install clean clean-cache clean-search-cache clean-publisher-cache clean-llm-cache clean-all test test-api test-papers run run-papers run-papers-full lint format check cache-info

# Default target
help:
	@echo "latex2json - LaTeX to JSON converter for academic data"
	@echo "Supports: collaborators, papers, book chapters"
	@echo ""
	@echo "Available targets:"
	@echo "  make install              - Install Python dependencies"
	@echo "  make clean                - Clean all cache files (preserves outputs)"
	@echo "  make clean-cache          - Clean all cache files"
	@echo "  make clean-search-cache   - Clean only search cache"
	@echo "  make clean-publisher-cache - Clean publisher & paper search caches"
	@echo "  make clean-llm-cache      - Clean all LLM-related caches"
	@echo "  make test                 - Run test with sample collaborator data"
	@echo "  make test-papers          - Test paper conversion with sample data"
	@echo "  make test-api             - Test Google API configuration"
	@echo "  make run FILE=...         - Run collaborator extraction"
	@echo "  make run-papers           - Run paper conversion (full)"
	@echo "  make run-papers-full      - Run paper conversion with all features"
	@echo "  make cache-info           - Show cache file information"
	@echo "  make lint                 - Run code linting with ruff"
	@echo "  make format               - Format code with ruff"
	@echo "  make check                - Run security checks with safety"

# Install dependencies
install:
	pip install -r requirements.txt
	@echo "✓ Dependencies installed successfully"

# Clean cache files
clean-cache:
	@echo "Cleaning cache files..."
	@rm -rf cache/
	@mkdir -p cache
	@echo "✓ Cache files cleaned"

# Clean cache files only (preserves output files)
clean: clean-cache
	@echo "✓ Cache files cleaned (output files preserved)"

# Clean only the search cache file
clean-search-cache:
	@echo "Cleaning search cache file..."
	@rm -f cache/.search_cache.json
	@echo "✓ Search cache cleaned"

# Clean only publisher-related cache files
clean-publisher-cache:
	@echo "Cleaning publisher cache files..."
	@rm -f cache/.publisher_cache.json
	@rm -f cache/.paper_search_cache.json
	@rm -f cache/.content_extraction_cache.json
	@echo "✓ Publisher cache files cleaned"

# Clean only LLM cache files
clean-llm-cache:
	@echo "Cleaning LLM cache files..."
	@rm -f cache/.llm_cache.json
	@rm -f cache/.llm_unified_*.json
	@rm -f cache/.llm_websearch_cache.json
	@rm -f cache/.llm_parse_cache.json
	@rm -f cache/.llm_enricher_cache.json
	@echo "✓ LLM cache files cleaned"

# Clean everything including Python cache
clean-all: clean
	@echo "Cleaning Python cache..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "✓ Python cache cleaned"

# Run test with sample data
test:
	@echo "Running test with sample collaborators..."
	python3 src/collab2json.py test_real.tex --output test_output.json -v 2

# Test Google API configuration
test-api:
	@echo "Testing Google API configuration..."
	python3 test_google_api.py

# Run on specific file
run:
ifndef FILE
	@echo "Error: Please specify FILE parameter"
	@echo "Usage: make run FILE=your_file.tex"
	@exit 1
else
	python3 src/collab2json.py $(FILE)
endif

# Run with full dataset
run-full:
	python3 src/collab2json.py collab_excerpts.tex --output collaborators.json -v 2

# Run with cache disabled
run-nocache: clean-cache
	python3 src/collab2json.py collab_excerpts.tex --output collaborators.json -v 2

# Run papers with cache disabled
run-papers-nocache: clean-publisher-cache clean-llm-cache
	python3 src/pub2json.py -v 2

# Run papers in dry-run mode (no API calls)
run-papers-dry:
	python3 src/pub2json.py --dry-run -v 2

# Run chapter conversion
run-chapters:
	python3 src/chpts2json.py -v 2

# Run publication conversion
run-papers:
	python3 src/pub2json.py -v 2

# Run publication conversion with all features enabled
run-papers-full:
	python3 src/pub2json.py -v 2 -i input_data/list_of_papers.tex -o output_data/papers.json

# Test paper conversion with test file
test-papers:
	@echo "Running test with sample papers..."
	python3 src/pub2json.py -i input_data/papers-test.tex -o output_data/papers-test.json -v 2

# Code quality checks
lint:
	@echo "Running linting checks..."
	ruff check .

# Format code
format:
	@echo "Formatting code..."
	ruff format .

# Security check
check:
	@echo "Running security checks..."
	safety check --bare

# Development helpers
watch:
	@echo "Watching for file changes..."
	@while true; do \
		inotifywait -e modify *.py modules/*.py 2>/dev/null && \
		clear && \
		make test; \
	done

# Show cache sizes
cache-info:
	@echo "Cache file sizes:"
	@ls -lh cache/.*.json 2>/dev/null || echo "No cache files found"
	@echo ""
	@echo "Cache entries:"
	@echo "  API cache:         $$(cat cache/.api_cache.json 2>/dev/null | jq 'keys | length' 2>/dev/null || echo '0') entries"
	@echo "  Keyword cache:     $$(cat cache/.keyword_cache.json 2>/dev/null | jq 'keys | length' 2>/dev/null || echo '0') entries"
	@echo "  Geocode cache:     $$(cat cache/.geocode_cache.json 2>/dev/null | jq 'keys | length' 2>/dev/null || echo '0') entries"
	@echo "  Search cache:      $$(cat cache/.search_cache.json 2>/dev/null | jq 'keys | length' 2>/dev/null || echo '0') entries"
	@echo "  Publisher cache:   $$(cat cache/.publisher_cache.json 2>/dev/null | jq 'keys | length' 2>/dev/null || echo '0') entries"
	@echo "  Paper search:      $$(cat cache/.paper_search_cache.json 2>/dev/null | jq 'keys | length' 2>/dev/null || echo '0') entries"
	@echo "  Content extract:   $$(cat cache/.content_extraction_cache.json 2>/dev/null | jq 'keys | length' 2>/dev/null || echo '0') entries"
	@echo "  LLM cache:         $$(cat cache/.llm_cache.json 2>/dev/null | jq 'keys | length' 2>/dev/null || echo '0') entries"
	@echo "  LLM parse:         $$(cat cache/.llm_parse_cache.json 2>/dev/null | jq 'keys | length' 2>/dev/null || echo '0') entries"
	@echo "  LLM enricher:      $$(cat cache/.llm_enricher_cache.json 2>/dev/null | jq 'keys | length' 2>/dev/null || echo '0') entries"
	@echo "  LLM websearch:     $$(cat cache/.llm_websearch_cache.json 2>/dev/null | jq 'keys | length' 2>/dev/null || echo '0') entries"
	@echo ""
	@echo "Total cache size: $$(du -sh cache/ 2>/dev/null | cut -f1 || echo '0')"

# Create sample config from example
config:
	@if [ ! -f config.json ]; then \
		cp config.json.example config.json; \
		echo "✓ Created config.json from example"; \
		echo "⚠️  Please edit config.json and add your API keys"; \
	else \
		echo "config.json already exists"; \
	fi