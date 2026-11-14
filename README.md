<!-- Author: Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science and Technology, Abu Dhabi, UAE) -->

# latex2json: publication lists and collaborators from LaTeX to JSON

Three Python scripts turn the LaTeX lists of an academic CV into JSON files that a website can read, and complete each entry with what can be found on the web. The script `src/pub2json.py` reads a list of journal papers and `src/chpts2json.py` a list of book chapters, both written as `etaremune` environments, and they add the DOI, the abstract, the publisher's page and keywords, and for the papers the preprints on arXiv and HAL. The script `src/collab2json.py` reads a list of collaborators and adds their affiliation, the city with its coordinates for a map, and their public profiles (ORCID, Google Scholar, ResearchGate, LinkedIn, homepage, e-mail address). The scripts were written to build the pages of publications and collaborators of the author's academic website, https://www.denys-dutykh.com/, and the repository holds the data of those pages.

The repository is at

https://github.com/dutykh/latex2json

## The data in this repository

The files of `input_data/`, `output_data/` and `cache/` are real data, not samples. The files `input_data/list_of_papers.tex` and `input_data/list_of_chpts.tex` are the author's lists of 178 journal papers and 11 book chapters, and `output_data/papers.json` and `output_data/bkchpts.json` the JSON made from them, with the results of the seven stages of the paper pipeline in `output_data/intermediate/`. The file `input_data/collab_excerpts.tex` lists the author's 204 collaborators with their affiliations, and `output_data/collaborators.json` holds what the collaborator script found about each of them on the web: affiliation, city and coordinates, homepage, profiles and, for most of them, an e-mail address. The folder `cache/` keeps the answers of the web searches and of the language models during those runs. These files are kept in the repository on purpose: they are the working data of the website, and they show in full what each script produces.

## Quick start

```bash
git clone https://github.com/dutykh/latex2json.git
cd latex2json
python3 -m venv .venv && . .venv/bin/activate
python3 -m pip install -r requirements.txt 'bibtexparser<2'
python3 src/pub2json.py --dry-run -v 1 -o papers-dry.json
python3 src/chpts2json.py --dry-run -v 1 -o chapters-dry.json
```

The two dry runs parse the LaTeX lists without any web request and without any key, and write `papers-dry.json` (178 papers) and `chapters-dry.json` (11 chapters). The bound on `bibtexparser` is needed because `scholarly` 1.7.11, its current release, imports a module that version 2 of `bibtexparser` no longer has; without it, the paper and chapter scripts stop at their first line.

The enrichment needs API keys, written into a file `config.json` made from the example (git ignores it):

```bash
cp config.json.example config.json
```

The scripts are run from the root of the repository, since their default paths are relative to it. With these defaults a run writes over the tracked files of `output_data/`: the paper and chapter scripts ask before overwriting an existing output file, unless `--force` is given, and the collaborator script updates its output file in place, reusing the entries whose LaTeX source has not changed.

## The three scripts

**Journal papers.** The script `src/pub2json.py` parses each entry of `input_data/list_of_papers.tex` with Claude, or with regular expressions when no model answers, then completes it in seven stages: the publisher of the journal is identified, the paper is searched on the web, the abstract and the data of the publisher's page are extracted, the preprints on arXiv and HAL are tried, the metadata services (CrossRef, Google Scholar, Semantic Scholar, Unpaywall) are queried, the preprints are tried again for what is still missing, and keywords are generated, only for a paper whose abstract was found. Each stage saves its results in `output_data/intermediate/`, so that a run can resume where it stopped. Publishers that refuse automated access are recognised and remembered.

**Book chapters.** The script `src/chpts2json.py` parses `input_data/list_of_chpts.tex` in the same way, completes each chapter from the metadata services, asynchronously, and generates keywords with Claude.

**Collaborators.** The script `src/collab2json.py` reads a list of collaborators in `input_data/` (each entry an `\item` with the name and the affiliations) and parses it with Gemini, Claude taking over when Gemini refuses or fails. It searches the web for each researcher (Google Custom Search, Google Scholar), keeps a profile only when it passes strict checks (the ORCID identifier confirmed by the ORCID API, the name found in the ResearchGate identifier, an e-mail address that contains a part of the name), finds the city of the university with the help of the language model, and geocodes it with OpenStreetMap Nominatim. The option `--verify` applies the checks again to an existing output file and removes the profiles that fail them.

### Options of the paper and chapter scripts

| Option | Meaning |
| --- | --- |
| `-i FILE`, `--input FILE` | LaTeX list to read (default `input_data/list_of_papers.tex`, or `input_data/list_of_chpts.tex`) |
| `-o FILE`, `--output FILE` | JSON file to write (default `output_data/papers.json`, or `output_data/bkchpts.json`) |
| `-c FILE`, `--config FILE` | configuration file; without it the paper script reads `config.json` at the root, else `src/config.json`, and the chapter script reads `src/config.json`, when they exist |
| `-v N`, `--verbose N` | verbosity from 0 to 3 (default 2) |
| `--dry-run` | parse only, without any web request |
| `--no-keywords` | no keyword generation |
| `--force` | overwrite the output file without asking |
| `--no-web-search` | papers only: no web search |
| `--no-content-extraction` | papers only: no extraction from the pages found |
| `--no-publisher-id` | papers only: no identification of the publisher |
| `--resume-from-stage N` | papers only: resume at stage N (1 to 7) from the intermediate results of the previous run |

### Options of the collaborator script

```text
python3 src/collab2json.py INPUT.tex [options]
```

| Option | Meaning |
| --- | --- |
| `INPUT.tex` | the list to read, a file name inside `input_data/` |
| `--output NAME` | the file to write inside `output_data/` (default: the name of the input with `.json`) |
| `--config FILE` | configuration file (default `config.json` of the current folder) |
| `-v N`, `--verbose N` | verbosity from 0 to 3 (default 2) |
| `--skip-geocoding` | no geocoding |
| `--skip-search` | no web search for the researchers |
| `--verify` | check the profiles of the existing output again, with the strict rules, and remove those that fail |
| `--fix-found-values` | search again the entries where Google Scholar is only marked as found, and record the actual identifiers |
| `--no-cache` | neither read nor write the caches |
| `--clear-cache` | delete seven of the caches (geocoding, searches, metadata, keywords and three of the language models) before running |
| `--clear-search-cache` | delete the search cache only |
| `--no-color` | output without colours |

The file of the website is produced with `python3 src/collab2json.py collab_excerpts.tex --output collaborators.json`. At verbosity 0 the scripts are silent; 1 shows the count of entries and a summary; 2 shows each entry as it is processed; 3 adds the LaTeX source of each entry, the parsing steps, the fields extracted, the hits and misses of the caches, the search queries and the results of the enrichment.

## Make targets

| Target | Command it runs |
| --- | --- |
| `make install` | `pip install -r requirements.txt` |
| `make run FILE=list.tex` | the collaborator script on `input_data/list.tex` |
| `make run-full` | the collaborator script on `collab_excerpts.tex`, writing `output_data/collaborators.json` |
| `make run-nocache` | the same after deleting the folder `cache/` |
| `make run-papers`, `make run-papers-full` | the paper script with its defaults, at verbosity 2 |
| `make run-papers-dry` | the paper script with `--dry-run` |
| `make run-papers-nocache` | the paper script after deleting the caches of publishers, searches and language models |
| `make run-chapters` | the chapter script with its defaults |
| `make clean`, `make clean-cache` | delete the folder `cache/` (the outputs stay) |
| `make clean-search-cache`, `make clean-publisher-cache`, `make clean-llm-cache` | delete some of the caches only |
| `make clean-all` | also delete the Python bytecode caches |
| `make cache-info` | size and number of entries of each cache |
| `make config` | copy `config.json.example` to `config.json` when it does not exist |
| `make lint`, `make format`, `make check` | `ruff check .`, `ruff format .`, `safety check --bare` |

The targets `make test`, `make test-papers` and `make test-api` call files that the repository does not hold (`input_data/test_real.tex`, `input_data/papers-test.tex` and `test_google_api.py`), and so does `make watch`, which runs `make test` after each change of a module.

## Configuration

The file `config.json.example` shows every section of the configuration: `api_keys` (the Google key for Gemini and Custom Search, `google_cx` the identifier of the custom search engine, the Anthropic key for Claude, `crossref_email`, and the keys of the Elsevier, Springer and Wiley services), `llm_config` (the primary and fallback providers and models), `preferences`, `search_settings`, `extraction`, `scraping` and `output`. The web search of the paper script also reads an optional section `paper_search_settings`, with the keys `max_results_per_query`, `use_llm_validation`, `use_smart_single_search`, `blocked_publishers`, `domain_priority` and `trusted_domains`.

## Caches

Every cache lives in the folder `cache/`, and git tracks it, so that a new run reuses the answers of the runs that produced the data of the website. The command `make clean-cache` deletes them all.

| File | Content |
| --- | --- |
| `.api_cache.json` | answers of the metadata services |
| `.keyword_cache.json` | generated keywords |
| `.geocode_cache.json` | geocoding results |
| `.search_cache.json` | web searches about the collaborators and ORCID checks |
| `.llm_cache.json` | language model answers of the collaborator script |
| `.llm_unified_parse_cache.json` | parsing of the collaborator entries by Gemini or Claude |
| `.llm_unified_location_cache.json` | locations extracted by the language model |
| `.llm_unified_profile_cache.json` | profiles analysed by the language model |
| `.llm_websearch_cache.json` | extraction from web search results |
| `.llm_enricher_cache.json` | enrichment of papers and chapters by Claude |
| `.llm_parse_cache.json` | parsing of papers and chapters by Claude |
| `.paper_search_cache.json` | web searches about the papers |
| `.content_extraction_cache.json` | content extracted from the pages of publishers and preprint servers |
| `.publisher_cache.json` | publisher of each journal |

## Repository map

| Path | Content |
| --- | --- |
| `src/` | the three scripts |
| `modules/` | the shared modules: parsers of LaTeX entries (`latex_parser.py`, `journal_parser.py`, `llm_parser.py`), language models (`llm_provider.py`, `gemini_client.py`, `llm_enricher.py`, `keyword_generator.py`), searches (`web_search.py`, `simple_search.py`, `scholar_helper.py`, `paper_web_search.py`, `llm_search_engines.py`), metadata (`metadata_enricher.py`, `publisher_identifier.py`, `paper_content_extractor.py`), and the cache, configuration and display helpers |
| `input_data/` | the LaTeX lists of papers, chapters and collaborators |
| `output_data/` | the JSON files of the website, and `intermediate/` with the seven stages of the paper pipeline |
| `cache/` | the caches described above |
| `templates/` | commented JSON templates for adding a paper, a chapter or a collaborator by hand; `templates/addons/` holds two entries prepared with them |
| `docs/` | three guides for adding a paper, a chapter or a collaborator by hand to the JSON files |
| `config.json.example` | the configuration to copy and fill |
| `requirements.txt`, `Makefile` | the dependencies and the targets above |
| `WARP.md` | notes on the project for the agent of the Warp terminal |

## Known limitations

Many publishers (Elsevier, Springer, Taylor and Francis among them) refuse automated access to their pages with the HTTP status 403. The scripts then fall back on the metadata of CrossRef found through the DOI, on Google Scholar, on the preprint addresses written in the comments of the LaTeX entries (arXiv, HAL), and on the language models when a key is given. The best results come from entries that carry their DOI and, in a comment, the address of their preprint.

## History

On 2 July 2025 the paper script gained its detailed output at every level of verbosity, with a preview of each entry, the parsing method, the fields extracted and the state of the caches, and the publishers' refusals were handled with browser-like headers and fallbacks on CrossRef, Google Scholar and arXiv. On 5 July 2025 came the repair of malformed JSON returned by the language models, a timeout of 30 seconds on every call to a model, the fallback from Gemini to Claude, the distinction between departments and universities in affiliations (Universiti, Universidad, Hochschule, Université and others), the rejection of nonsensical city names before they enter a cache, and the strict checks of profiles with the option `--verify`. On 6 July 2025 the paper script gained the identification of publishers, a single web search per paper, the extraction of content from the publishers' pages, then from the preprints, the extraction of every form of article number (short and long numbers, and identifiers such as `e0305534`), the memory of publishers that refuse access, and the rule that keywords are generated only from an abstract.

## Requirements

Python 3.8 or newer, with the packages of `requirements.txt` (and `bibtexparser` below version 2, as in the Quick start); `ruff` and `safety` are used by `make lint`, `make format` and `make check` only. The enrichment needs network access and the API keys of the configuration.

## How to cite

The file [`CITATION.cff`](CITATION.cff) describes the scripts for citation, and GitHub turns it into a reference, in APA or BibTeX, through the "Cite this repository" button of the repository page.

## Licence

The code is free software under the GNU General Public License, version 3; the text is in [LICENSE](LICENSE).

## Author

Dr. Denys Dutykh, Mathematics Department, Khalifa University of Science and Technology, Abu Dhabi, United Arab Emirates (https://www.denys-dutykh.com/).
