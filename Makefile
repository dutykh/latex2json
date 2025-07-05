# Makefile for latex2json project
# Author: Dr. Denys Dutykh
# Date: 2025-01-04

.PHONY: help install clean clean-cache clean-search-cache clean-all test test-api run lint format check

# Default target
help:
	@echo "latex2json - LaTeX to JSON converter for collaborator data"
	@echo ""
	@echo "Available targets:"
	@echo "  make install           - Install Python dependencies"
	@echo "  make clean             - Clean cache files only (preserves outputs)"
	@echo "  make clean-cache       - Clean only cache files"
	@echo "  make clean-search-cache - Clean only the search cache file"
	@echo "  make test              - Run test with sample data"
	@echo "  make test-api          - Test Google API configuration"
	@echo "  make run FILE=...      - Run on specific file (e.g., make run FILE=collab_excerpts.tex)"
	@echo "  make lint              - Run code linting with ruff"
	@echo "  make format            - Format code with ruff"
	@echo "  make check             - Run security checks with safety"

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

# Run chapter conversion
run-chapters:
	python3 src/chpts2json.py -v 2

# Run publication conversion
run-papers:
	python3 src/pub2json.py -v 2

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
	@for f in cache/.*.json; do \
		if [ -f "$$f" ]; then \
			echo "$$f: $$(cat $$f | jq 'keys | length' 2>/dev/null || echo "invalid JSON")"; \
		fi \
	done

# Create sample config from example
config:
	@if [ ! -f config.json ]; then \
		cp config.json.example config.json; \
		echo "✓ Created config.json from example"; \
		echo "⚠️  Please edit config.json and add your API keys"; \
	else \
		echo "config.json already exists"; \
	fi