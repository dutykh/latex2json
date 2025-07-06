# -*- coding: utf-8 -*-
"""
Metadata enricher module for fetching additional information from publisher APIs.

This module integrates with CrossRef, publisher-specific APIs, and Google Scholar
to enrich bibliography entries with DOIs, abstracts, keywords, and other metadata.

Author: Dr. Denys Dutykh (Khalifa University of Science and Technology, Abu Dhabi, UAE)
Date: 2025-07-01
"""

import re
import time
import asyncio
import aiohttp
from typing import Dict, Any, Optional, List
from urllib.parse import quote_plus
from bs4 import BeautifulSoup
from scholarly import scholarly

from .cache_manager import CacheManager
from .config_manager import ConfigManager
from .llm_enricher import LLMMetadataEnricher
from .llm_search_engines import AcademicSearchEngines


class MetadataEnricher:
    """Enriches bibliography entries with metadata from various sources."""

    def __init__(self, config: ConfigManager, cache: CacheManager, verbose: int = 2):
        """
        Initialize the metadata enricher.

        Args:
            config: Configuration manager instance
            cache: Cache manager instance
            verbose: Verbosity level (0-3)
        """
        self.config = config
        self.cache = cache
        self.verbose = verbose

        # Initialize session headers
        self.headers = config.get_headers()

        # Rate limiting
        self.last_request_time = {}
        self.delay = config.get("preferences.delay_between_requests", 1.0)

        # Initialize LLM enricher if enabled
        self.llm_enricher = None
        if config.get("preferences.use_llm_enricher", True):
            try:
                self.llm_enricher = LLMMetadataEnricher(config, verbose=verbose)
                if self.verbose >= 2 and self.llm_enricher.enabled:
                    print("[ENRICH] LLM enricher initialized")
            except Exception as e:
                if self.verbose >= 1:
                    print(f"[ENRICH] Failed to initialize LLM enricher: {e}")

    async def enrich_entries(
        self, entries: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Enrich multiple entries concurrently.

        Args:
            entries: List of bibliography entries

        Returns:
            List of enriched entries
        """
        if self.verbose >= 1:
            print(f"\n[ENRICH] Starting enrichment for {len(entries)} entries")

        # Create async session
        async with aiohttp.ClientSession(headers=self.headers) as session:
            tasks = []
            for i, entry in enumerate(entries):
                task = self._enrich_single_entry(session, entry, i + 1, len(entries))
                tasks.append(task)

            # Process entries with limited concurrency
            enriched_entries = await self._gather_with_concurrency(
                tasks, max_concurrent=3
            )

        return enriched_entries

    async def _gather_with_concurrency(
        self, tasks: List, max_concurrent: int = 3
    ) -> List:
        """Process tasks with limited concurrency."""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def run_with_semaphore(task):
            async with semaphore:
                return await task

        return await asyncio.gather(
            *[run_with_semaphore(task) for task in tasks], return_exceptions=True
        )

    async def _enrich_single_entry(
        self,
        session: aiohttp.ClientSession,
        entry: Dict[str, Any],
        index: int,
        total: int,
    ) -> Dict[str, Any]:
        """Enrich a single bibliography entry with 2-minute timeout."""
        try:
            # Apply 2-minute timeout to the entire enrichment process
            return await asyncio.wait_for(
                self._enrich_single_entry_impl(session, entry, index, total),
                timeout=120.0,  # 2 minutes
            )
        except asyncio.TimeoutError:
            if self.verbose >= 1:
                print(
                    f"[TIMEOUT] Entry {index} timed out after 2 minutes: {entry.get('title', 'Unknown')[:60]}..."
                )
            # Return the entry with minimal metadata on timeout
            return entry
        except Exception as e:
            if self.verbose >= 1:
                print(f"[ERROR] Failed to enrich entry {index}: {e}")
            return entry

    async def _enrich_single_entry_impl(
        self,
        session: aiohttp.ClientSession,
        entry: Dict[str, Any],
        index: int,
        total: int,
    ) -> Dict[str, Any]:
        """Implementation of single entry enrichment."""
        if self.verbose >= 2:
            print(f"\n[PROGRESS] Processing entry {index}/{total}")
            print(f"[ENRICH] Title: {entry.get('title', 'Unknown')[:60]}...")

        # Create cache key
        cache_key = self.cache.create_entry_key(entry)

        # Check cache first
        cached_metadata = self.cache.get_api_response(cache_key)
        if cached_metadata:
            if self.verbose >= 2:
                print(f"[ENRICH] Using cached metadata for entry {index}")
                if cached_metadata.get("doi"):
                    print(f"  - Cached DOI: {cached_metadata['doi']}")
                if cached_metadata.get("publisher_url"):
                    print(f"  - Cached URL: {cached_metadata['publisher_url']}")
                if cached_metadata.get("keywords"):
                    print(
                        f"  - Cached keywords: {len(cached_metadata['keywords'])} found"
                    )
            entry.update(cached_metadata)
            return entry

        # Initialize metadata fields
        metadata = {
            "doi": entry.get("doi", ""),
            "abstract": "",
            "keywords": [],
            "publisher_url": "",
            "isbn": "",
            "keyword_source": "none",
        }

        # Try different enrichment strategies in order
        strategies = []

        # Add LLM enricher as first strategy if available
        if self.llm_enricher and self.llm_enricher.enabled:
            strategies.append(("llm", self._enrich_from_llm))

        # Add traditional strategies
        strategies.extend(
            [
                ("url", self._enrich_from_url),
                ("crossref", self._enrich_from_crossref),
                ("publisher", self._enrich_from_publisher),
                ("scholar", self._enrich_from_scholar),
            ]
        )

        for strategy_name, strategy_func in strategies:
            try:
                if self.verbose >= 3:
                    print(f"[ENRICH] Trying {strategy_name} strategy")

                # Apply 30-second timeout to each strategy
                success = await asyncio.wait_for(
                    strategy_func(session, entry, metadata),
                    timeout=30.0,  # 30 seconds per strategy
                )

                if success and (
                    metadata["doi"] or metadata["abstract"] or metadata["keywords"]
                ):
                    if self.verbose >= 2:
                        print(f"[SUCCESS] Enriched via {strategy_name}")
                        if metadata.get("doi"):
                            print(f"  - Found DOI: {metadata['doi']}")
                        if metadata.get("publisher_url"):
                            print(f"  - Found URL: {metadata['publisher_url']}")
                        if metadata.get("abstract"):
                            print(
                                f"  - Found abstract: {len(metadata['abstract'])} chars"
                            )
                        if metadata.get("keywords"):
                            print(
                                f"  - Found keywords: {', '.join(metadata['keywords'][:5])}"
                            )
                            if len(metadata["keywords"]) > 5:
                                print(
                                    f"    ... and {len(metadata['keywords']) - 5} more"
                                )
                    break

            except asyncio.TimeoutError:
                if self.verbose >= 2:
                    print(
                        f"[TIMEOUT] {strategy_name} strategy timed out after 30 seconds"
                    )
            except Exception as e:
                if self.verbose >= 2:
                    print(f"[ERROR] {strategy_name} strategy failed: {e}")

        # If DOI found but no proper publisher URL (or just DOI URL), try to convert based on DOI prefix
        if metadata.get("doi"):
            current_url = metadata.get("publisher_url", "")
            # Check if we have a generic DOI URL instead of publisher-specific
            if (
                not current_url
                or "doi.org" in current_url
                or current_url == f"https://doi.org/{metadata['doi']}"
            ):
                converted_url = self._convert_doi_to_publisher_url(metadata["doi"])
                if converted_url and converted_url != current_url:
                    metadata["publisher_url"] = converted_url
                    if self.verbose >= 2:
                        print(
                            f"[ENRICH] Converted DOI to publisher URL: {converted_url}"
                        )

            # Always try to fetch abstract and keywords from publisher page if we don't have them
            if metadata.get("publisher_url") and (
                not metadata.get("abstract") or not metadata.get("keywords")
            ):
                await self._fetch_from_publisher_url(session, metadata)

        # If still no publisher URL and no DOI, try constructing URL from journal metadata
        if not metadata.get("publisher_url") and not metadata.get("doi"):
            # Initialize search engines
            search_engines = AcademicSearchEngines(session, self.verbose)
            constructed = search_engines.construct_publisher_url(entry)
            if constructed:
                if constructed.get("url"):
                    metadata["publisher_url"] = constructed["url"]
                    if self.verbose >= 2:
                        print(
                            f"[ENRICH] Constructed publisher URL from journal data: {constructed['url']}"
                        )
                if constructed.get("doi"):
                    metadata["doi"] = constructed["doi"]
                    if self.verbose >= 2:
                        print(
                            f"[ENRICH] Constructed DOI from journal data: {constructed['doi']}"
                        )

        # Detect publisher from DOI if not already found
        if metadata.get("doi") and not entry.get("publisher", {}).get("name"):
            publisher_name = self._detect_publisher_from_doi(metadata["doi"])
            if publisher_name:
                entry["publisher"]["name"] = publisher_name
                if self.verbose >= 2:
                    print(f"[ENRICH] Detected publisher from DOI: {publisher_name}")

        # Generate keywords if not found but we have title and/or abstract
        if (
            not metadata.get("keywords")
            and self.llm_enricher
            and self.llm_enricher.enabled
        ):
            title = entry.get("title", "")
            abstract = metadata.get("abstract", "") or entry.get("abstract", "")

            if title or abstract:
                if self.verbose >= 2:
                    print(
                        "[ENRICH] No keywords found, attempting to generate with LLM..."
                    )

                generated_keywords = await self.llm_enricher.generate_keywords(
                    title, abstract
                )

                if generated_keywords:
                    metadata["keywords"] = generated_keywords
                    metadata["keyword_source"] = "generated"
                    if self.verbose >= 2:
                        print(f"[ENRICH] Generated {len(generated_keywords)} keywords")
                        if self.verbose >= 3:
                            print(f"  Keywords: {', '.join(generated_keywords[:5])}")
                            if len(generated_keywords) > 5:
                                print(f"  ... and {len(generated_keywords) - 5} more")

        # Update entry with metadata
        entry.update(metadata)

        # Cache the metadata (excluding the original entry fields)
        self.cache.set_api_response(cache_key, metadata)

        return entry

    def _detect_publisher_from_doi(self, doi: str) -> Optional[str]:
        """Detect publisher from DOI prefix."""
        doi_prefixes = {
            "10.1016/": "Elsevier",
            "10.1007/": "Springer",
            "10.1002/": "Wiley",
            "10.1080/": "Taylor & Francis",
            "10.1093/": "Oxford University Press",
            "10.1017/": "Cambridge University Press",
            "10.1515/": "De Gruyter",
            "10.4018/": "IGI Global",
            "10.1201/": "CRC Press",
            "10.1142/": "World Scientific",
            "10.1103/": "American Physical Society",
        }

        for prefix, publisher in doi_prefixes.items():
            if doi.startswith(prefix):
                return publisher

        return None

    def _convert_doi_to_publisher_url(self, doi: str) -> Optional[str]:
        """Convert DOI to publisher-specific URL."""
        if doi.startswith("10.1016/"):
            # Elsevier/ScienceDirect
            pii = self._doi_to_pii(doi)
            if pii:
                return f"https://www.sciencedirect.com/science/article/abs/pii/{pii}"
        elif doi.startswith("10.1007/"):
            # Springer
            return f"https://link.springer.com/chapter/{doi}"
        elif doi.startswith("10.1002/"):
            # Wiley
            return f"https://onlinelibrary.wiley.com/doi/{doi}"
        elif doi.startswith("10.1103/"):
            # American Physical Society (APS)
            # Extract journal name and construct URL
            doi_parts = doi.split("/")
            if len(doi_parts) >= 2:
                journal_info = doi_parts[1]  # e.g., "PhysRevResearch.6.033284"

                # Map journal names to URL paths
                journal_map = {
                    "PhysRevLett": "prl",
                    "PhysRevA": "pra",
                    "PhysRevB": "prb",
                    "PhysRevC": "prc",
                    "PhysRevD": "prd",
                    "PhysRevE": "pre",
                    "PhysRevX": "prx",
                    "PhysRevResearch": "prresearch",
                    "PhysRevApplied": "prapplied",
                    "PhysRevFluids": "prfluids",
                    "PhysRevMaterials": "prmaterials",
                    "PhysRevAccelBeams": "prab",
                    "RevModPhys": "rmp",
                    "PhysRevSTAB": "prstab",
                    "PhysRevSTPER": "prstper",
                    "PhysRev": "pr",  # Legacy Physical Review
                }

                # Extract journal name from DOI
                for journal_name, url_code in journal_map.items():
                    if journal_info.startswith(journal_name):
                        return f"https://journals.aps.org/{url_code}/abstract/{doi}"

                # If no specific match, try generic format
                return f"https://journals.aps.org/abstract/{doi}"

        # Default to DOI.org resolver
        return f"https://doi.org/{doi}"

    async def _enrich_from_url(
        self,
        session: aiohttp.ClientSession,
        entry: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> bool:
        """Try to enrich from provided URL."""
        url = entry.get("url", "")
        if not url:
            return False

        # Try to extract DOI from URL first
        doi_from_url = self._extract_doi_from_url(url)
        if doi_from_url:
            metadata["doi"] = doi_from_url
            if self.verbose >= 2:
                print(f"[ENRICH] Extracted DOI from URL: {doi_from_url}")
            # Convert to publisher URL if possible
            publisher_url = self._convert_doi_to_publisher_url(doi_from_url)
            if publisher_url:
                metadata["publisher_url"] = publisher_url
                # If it's a publisher URL (not just doi.org), we found useful metadata
                if "doi.org" not in publisher_url:
                    return True

        # For preprint URLs, we'll now try to extract metadata
        # The paper_content_extractor will handle these specifically

        try:
            await self._rate_limit("url")

            async with session.get(
                url, timeout=self.config.get("preferences.timeout", 30)
            ) as response:
                if response.status != 200:
                    return False

                html = await response.text()
                soup = BeautifulSoup(html, "html.parser")

                # Extract DOI
                doi_meta = soup.find("meta", {"name": "citation_doi"}) or soup.find(
                    "meta", {"name": "DC.Identifier", "scheme": "doi"}
                )
                if doi_meta:
                    metadata["doi"] = doi_meta.get("content", "")

                # Extract abstract
                abstract_meta = soup.find(
                    "meta", {"name": "citation_abstract"}
                ) or soup.find("meta", {"name": "DC.Description"})
                if abstract_meta:
                    metadata["abstract"] = abstract_meta.get("content", "")

                # Extract keywords
                keywords = []
                keyword_metas = soup.find_all(
                    "meta", {"name": "citation_keywords"}
                ) + soup.find_all("meta", {"name": "keywords"})
                for meta in keyword_metas:
                    content = meta.get("content", "")
                    keywords.extend(
                        [k.strip() for k in content.split(",") if k.strip()]
                    )

                if keywords:
                    metadata["keywords"] = list(set(keywords))  # Remove duplicates
                    metadata["keyword_source"] = "publisher"

                metadata["publisher_url"] = url

                return bool(
                    metadata["doi"] or metadata["abstract"] or metadata["keywords"]
                )

        except Exception as e:
            if self.verbose >= 3:
                print(f"[ENRICH] URL fetch error: {e}")
            return False

    async def _enrich_from_crossref(
        self,
        session: aiohttp.ClientSession,
        entry: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> bool:
        """Try to enrich from CrossRef API."""
        try:
            # Build search query
            title = entry.get("title", "")
            authors = entry.get("authors", [])
            book_title = entry.get("book_title", "")

            if not title:
                return False

            # Search using crossref-commons
            await self._rate_limit("crossref")

            # First try: Search with chapter title + book title (if available)
            if book_title:
                query_params = {
                    "query.bibliographic": f"{title} {book_title}",
                    "rows": 5,
                    "sort": "relevance",
                    "filter": "type:book-chapter",
                }
            else:
                query_params = {
                    "query.bibliographic": title,
                    "rows": 5,
                    "sort": "relevance",
                }

            # Add author if available
            if authors:
                author_name = f"{authors[0].get('firstName', '')} {authors[0].get('lastName', '')}".strip()
                if author_name:
                    query_params["query.author"] = author_name

            # Add year filter if available
            year = entry.get("year")
            if year:
                query_params["filter"] = (
                    query_params.get("filter", "")
                    + f",from-pub-date:{year},until-pub-date:{year}"
                )

            # Search CrossRef using aiohttp (async)
            email = self.config.get("api_keys.crossref_email", "")
            headers = dict(self.headers)  # Copy headers
            if email:
                headers["mailto"] = email

            async with session.get(
                "https://api.crossref.org/works",
                params=query_params,
                headers=headers,
                timeout=aiohttp.ClientTimeout(
                    total=self.config.get("preferences.timeout", 30)
                ),
            ) as response:
                if response.status != 200:
                    return False

                data = await response.json()
            items = data.get("message", {}).get("items", [])

            # Check results
            for work in items[:5]:  # Check first 5 results
                # Check title similarity
                work_titles = work.get("title", [])
                if not work_titles:
                    continue

                work_title = work_titles[0].lower()

                # For book chapters, also check container title
                container_titles = work.get("container-title", [])
                container_match = False
                if book_title and container_titles:
                    container_title = container_titles[0].lower()
                    container_match = (
                        self._calculate_similarity(book_title.lower(), container_title)
                        > 0.7
                    )

                # Check if titles match (either chapter title or we have a container match)
                title_match = (
                    self._calculate_similarity(title.lower(), work_title) > 0.8
                )

                if title_match or container_match:
                    # Extract metadata
                    metadata["doi"] = work.get("DOI", "")

                    # Get abstract if available
                    if "abstract" in work:
                        metadata["abstract"] = work["abstract"]

                    # Get URL
                    if "URL" in work:
                        metadata["publisher_url"] = work["URL"]

                    # Get ISBN
                    isbn_list = work.get("ISBN", [])
                    if isbn_list:
                        metadata["isbn"] = isbn_list[0]

                    if self.verbose >= 3:
                        print(f"[ENRICH] CrossRef match found: {work_title[:50]}...")

                    return True

            # Second try: If no match found and we have book title, try searching just the book
            if not items and book_title:
                if self.verbose >= 3:
                    print("[ENRICH] No chapter match, trying book search...")

                query_params = {
                    "query.bibliographic": book_title,
                    "rows": 3,
                    "filter": "type:book",
                }

                async with session.get(
                    "https://api.crossref.org/works",
                    params=query_params,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(
                        total=self.config.get("preferences.timeout", 30)
                    ),
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        items = data.get("message", {}).get("items", [])

                        for work in items[:2]:
                            container_titles = work.get("title", [])
                            if container_titles:
                                container_title = container_titles[0].lower()
                                if (
                                    self._calculate_similarity(
                                        book_title.lower(), container_title
                                    )
                                    > 0.8
                                ):
                                    # Found the book, extract its DOI and try to construct chapter DOI
                                    book_doi = work.get("DOI", "")
                                    if book_doi and book_doi.startswith("10.1007/"):
                                        # For Springer, chapter DOIs often follow pattern book_doi + _chapternum
                                        if self.verbose >= 2:
                                            print(
                                                f"[ENRICH] Found book DOI: {book_doi}, searching for chapter..."
                                            )
                                        # We'll rely on publisher enrichment to find the exact chapter
                                        return False

            return False

        except Exception as e:
            if self.verbose >= 3:
                print(f"[ENRICH] CrossRef error: {e}")
            return False

    async def _enrich_from_publisher(
        self,
        session: aiohttp.ClientSession,
        entry: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> bool:
        """Try to enrich from publisher-specific sources."""
        publisher = entry.get("publisher", {}).get("name", "").lower()

        if "springer" in publisher:
            return await self._enrich_from_springer(session, entry, metadata)
        elif "elsevier" in publisher:
            return await self._enrich_from_elsevier(session, entry, metadata)
        elif "wiley" in publisher:
            return await self._enrich_from_wiley(session, entry, metadata)
        else:
            return False

    async def _enrich_from_springer(
        self,
        session: aiohttp.ClientSession,
        entry: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> bool:
        """Enrich from Springer/SpringerLink."""
        try:
            # If URL is already provided, try to use it first
            url = entry.get("url", "")
            if url and "springer.com" in url:
                if self.verbose >= 2:
                    print(f"[ENRICH] Found existing Springer URL: {url}")
                # Extract DOI from URL if possible
                doi_match = re.search(r"10\.1007/[^\s/]+", url)
                if doi_match:
                    metadata["doi"] = doi_match.group(0)
                    metadata["publisher_url"] = (
                        f"https://link.springer.com/chapter/{metadata['doi']}"
                    )
                    if self.verbose >= 2:
                        print(f"[ENRICH] Extracted DOI from URL: {metadata['doi']}")
                    return True

            # Search by title and book title
            title = entry.get("title", "")
            book_title = entry.get("book_title", "")

            # Try searching with both chapter and book title
            if book_title:
                search_query = f"{title} {book_title}"
            else:
                search_query = title

            search_url = (
                f"https://link.springer.com/search?query={quote_plus(search_query)}"
            )

            if self.verbose >= 3:
                print(f"[ENRICH] Searching Springer with query: {search_query}")

            await self._rate_limit("springer")

            async with session.get(
                search_url, timeout=self.config.get("preferences.timeout", 30)
            ) as response:
                if response.status != 200:
                    return False

                html = await response.text()
                soup = BeautifulSoup(html, "html.parser")

                # Find search results - try different selectors
                results = soup.find_all("a", {"class": "title"}) or soup.find_all(
                    "h3", class_=re.compile("result-item-title")
                )

                if not results and self.verbose >= 3:
                    print(
                        "[ENRICH] No results found with standard selectors, trying alternative"
                    )
                    # Try to find any links that might be chapter links
                    results = soup.find_all("a", href=re.compile(r"/chapter/10\.1007/"))

                found_match = False
                for result in results[:5]:  # Check first 5 results
                    if hasattr(result, "get_text"):
                        result_title = result.get_text(strip=True).lower()
                    else:
                        result_title = result.string.lower() if result.string else ""

                    # Check similarity with chapter title
                    title_sim = self._calculate_similarity(title.lower(), result_title)

                    # Also check if it's the right book
                    book_match = False
                    if book_title:
                        # Check if result mentions the book title
                        parent_elem = result.find_parent(
                            "div", class_=re.compile("result")
                        )
                        if parent_elem:
                            parent_text = parent_elem.get_text(strip=True).lower()
                            book_match = book_title.lower() in parent_text

                    if title_sim > 0.7 or book_match:
                        # Get the chapter URL
                        if result.name == "a":
                            href = result.get("href", "")
                        else:
                            # Find the link within the element
                            link = result.find("a")
                            href = link.get("href", "") if link else ""

                        if href:
                            chapter_url = (
                                href
                                if href.startswith("http")
                                else f"https://link.springer.com{href}"
                            )

                            # Extract DOI from URL
                            doi_match = re.search(r"10\.1007/[^/\s]+", chapter_url)
                            if doi_match:
                                metadata["doi"] = doi_match.group(0)
                                metadata["publisher_url"] = (
                                    f"https://link.springer.com/chapter/{metadata['doi']}"
                                )

                                if self.verbose >= 2:
                                    print(
                                        f"[ENRICH] Found Springer chapter DOI: {metadata['doi']}"
                                    )

                                # Always fetch the chapter page for abstract and keywords
                                if not metadata.get("abstract") or not metadata.get(
                                    "keywords"
                                ):
                                    try:
                                        async with session.get(
                                            chapter_url,
                                            timeout=self.config.get(
                                                "preferences.timeout", 30
                                            ),
                                        ) as chapter_response:
                                            if chapter_response.status == 200:
                                                chapter_html = (
                                                    await chapter_response.text()
                                                )
                                                self._parse_springer_chapter(
                                                    chapter_html, metadata
                                                )
                                                # Set publisher URL to the chapter URL
                                                metadata["publisher_url"] = chapter_url
                                    except Exception as e:
                                        if self.verbose >= 3:
                                            print(
                                                f"[ENRICH] Failed to fetch chapter page: {e}"
                                            )

                                found_match = True
                                break

                # If no match found but we have a book title, try searching just for the book
                if not found_match and book_title and not title_sim > 0.5:
                    if self.verbose >= 3:
                        print("[ENRICH] No chapter match, searching for book...")

                    book_search_url = f"https://link.springer.com/search?query={quote_plus(book_title)}&facet-content-type=%22Book%22"

                    async with session.get(
                        book_search_url,
                        timeout=self.config.get("preferences.timeout", 30),
                    ) as book_response:
                        if book_response.status == 200:
                            book_html = await book_response.text()
                            book_soup = BeautifulSoup(book_html, "html.parser")

                            # Find book results
                            book_results = book_soup.find_all("a", {"class": "title"})[
                                :3
                            ]

                            for book_result in book_results:
                                book_result_title = book_result.get_text(
                                    strip=True
                                ).lower()
                                if (
                                    self._calculate_similarity(
                                        book_title.lower(), book_result_title
                                    )
                                    > 0.8
                                ):
                                    book_href = book_result.get("href", "")
                                    if book_href:
                                        # Extract book DOI and try to construct chapter DOI
                                        book_doi_match = re.search(
                                            r"10\.1007/(978-[0-9-]+)", book_href
                                        )
                                        if book_doi_match:
                                            book_isbn = book_doi_match.group(1)
                                            if self.verbose >= 2:
                                                print(
                                                    f"[ENRICH] Found book ISBN: {book_isbn}, inferring chapter DOI..."
                                                )
                                            # For known patterns, we might be able to infer chapter DOI
                                            # but this is unreliable, so we just note it
                                            return False

                return found_match

        except Exception as e:
            if self.verbose >= 3:
                print(f"[ENRICH] Springer error: {e}")
            return False

    def _parse_springer_chapter(self, html: str, metadata: Dict[str, Any]) -> bool:
        """Parse Springer chapter page."""
        soup = BeautifulSoup(html, "html.parser")

        # Extract DOI - try multiple selectors
        doi_elem = soup.find(
            "span",
            {"class": "bibliographic-information__value", "data-test": "doi-value"},
        )
        if not doi_elem:
            # Try meta tag
            doi_meta = soup.find("meta", {"name": "citation_doi"})
            if doi_meta:
                metadata["doi"] = doi_meta.get("content", "")
        else:
            metadata["doi"] = doi_elem.get_text(strip=True)

        # Extract abstract
        abstract_elem = soup.find("div", {"class": "c-chapter-abstract"})
        if not abstract_elem:
            abstract_elem = soup.find("section", {"class": "Abstract"})
        if not abstract_elem:
            # Try additional selectors
            abstract_elem = (
                soup.find("div", {"class": "chapter-abstract-container"})
                or soup.find("section", {"id": "Abs1"})
                or soup.find("div", {"class": "AbstractSection"})
            )

        if abstract_elem:
            # Try to find paragraphs first
            abstract_paras = abstract_elem.find_all("p")
            if abstract_paras:
                abstract_text = " ".join(
                    [p.get_text(strip=True) for p in abstract_paras]
                )
            else:
                abstract_text = abstract_elem.get_text(strip=True)

            # Remove "Abstract" label if present
            abstract_text = re.sub(r"^Abstract:?\s*", "", abstract_text, flags=re.I)

            if abstract_text and len(abstract_text) > 50:
                metadata["abstract"] = abstract_text
                if self.verbose >= 3:
                    print(f"[PARSE] Found Springer abstract: {abstract_text[:100]}...")
        else:
            if self.verbose >= 3:
                print("[PARSE] No abstract found in Springer chapter page")

        # Extract keywords - try multiple selectors
        keywords = []
        keywords_section = soup.find("div", {"class": "c-chapter-keywords"})
        if not keywords_section:
            keywords_section = soup.find("div", {"class": "KeywordGroup"})

        if keywords_section:
            # Try different keyword element selectors
            keyword_elems = keywords_section.find_all(
                "span", {"class": "c-chapter-keyword"}
            )
            if not keyword_elems:
                keyword_elems = keywords_section.find_all("span", {"class": "Keyword"})

            for kw in keyword_elems:
                kw_text = kw.get_text(strip=True)
                if kw_text and kw_text not in keywords:
                    keywords.append(kw_text)

        # Also check meta tags for keywords
        if not keywords:
            keyword_metas = soup.find_all("meta", {"name": "citation_keywords"})
            for meta in keyword_metas:
                content = meta.get("content", "")
                keywords.extend([k.strip() for k in content.split(";") if k.strip()])

        if keywords:
            metadata["keywords"] = keywords
            metadata["keyword_source"] = "publisher"
            if self.verbose >= 2:
                print(f"[ENRICH] Found {len(keywords)} keywords from Springer page")

        return bool(metadata["doi"] or metadata["abstract"] or metadata["keywords"])

    async def _enrich_from_elsevier(
        self,
        session: aiohttp.ClientSession,
        entry: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> bool:
        """Enrich from Elsevier/ScienceDirect."""
        try:
            # If we already have a DOI that looks like Elsevier, convert it
            doi = metadata.get("doi", "")
            if doi.startswith("10.1016/"):
                # Extract PII from DOI
                pii = self._doi_to_pii(doi)
                if pii:
                    sciencedirect_url = (
                        f"https://www.sciencedirect.com/science/article/abs/pii/{pii}"
                    )
                    metadata["publisher_url"] = sciencedirect_url
                    if self.verbose >= 2:
                        print(
                            f"[ENRICH] Converted Elsevier DOI to ScienceDirect URL: {sciencedirect_url}"
                        )

                    # Try to fetch abstract and keywords from ScienceDirect
                    await self._rate_limit("elsevier")

                    try:
                        async with session.get(
                            sciencedirect_url,
                            timeout=self.config.get("preferences.timeout", 30),
                        ) as response:
                            if response.status == 200:
                                html = await response.text()
                                self._parse_sciencedirect_page(html, metadata)
                    except Exception as e:
                        if self.verbose >= 3:
                            print(f"[ENRICH] Failed to fetch ScienceDirect page: {e}")

                    return True

            # If no DOI, try searching by title
            title = entry.get("title", "")
            if not title:
                return False

            # Search ScienceDirect
            search_url = (
                f"https://www.sciencedirect.com/search/advanced?qs={quote_plus(title)}"
            )

            await self._rate_limit("elsevier")

            async with session.get(
                search_url, timeout=self.config.get("preferences.timeout", 30)
            ) as response:
                if response.status != 200:
                    return False

                html = await response.text()
                soup = BeautifulSoup(html, "html.parser")

                # Find search results
                results = soup.find_all("a", {"class": "result-list-title-link"})

                for result in results[:3]:  # Check first 3 results
                    result_title = result.get_text(strip=True).lower()

                    if self._calculate_similarity(title.lower(), result_title) > 0.8:
                        chapter_url = "https://www.sciencedirect.com" + result.get(
                            "href", ""
                        )
                        metadata["publisher_url"] = chapter_url

                        # Extract DOI if present
                        doi_elem = result.find_parent().find("span", {"class": "doi"})
                        if doi_elem:
                            metadata["doi"] = (
                                doi_elem.get_text(strip=True)
                                .replace("DOI:", "")
                                .strip()
                            )

                        return True

                return False

        except Exception as e:
            if self.verbose >= 3:
                print(f"[ENRICH] Elsevier error: {e}")
            return False

    def _extract_doi_from_url(self, url: str) -> Optional[str]:
        """Extract DOI from various URL formats."""
        # Direct DOI patterns
        doi_patterns = [
            r"doi\.org/(10\.\d{4,}/[^\s]+)",  # doi.org URLs
            r"dx\.doi\.org/(10\.\d{4,}/[^\s]+)",  # dx.doi.org URLs
            r"/chapter/(10\.\d{4,}/[^\s]+)",  # Springer chapter URLs
            r"/doi/(10\.\d{4,}/[^\s]+)",  # Wiley URLs
            r"/abstract/(10\.\d{4,}/[^\s]+)",  # APS abstract URLs
            r"journals\.aps\.org/[^/]+/(?:abstract|pdf)/(10\.\d{4,}/[^\s]+)",  # APS journal URLs
            r"link\.aps\.org/doi/(10\.\d{4,}/[^\s]+)",  # APS link URLs
            r"/(10\.\d{4,}/[^\s&?#]+)",  # General DOI pattern
        ]

        for pattern in doi_patterns:
            match = re.search(pattern, url, re.IGNORECASE)
            if match:
                doi = match.group(1)
                # Clean up the DOI
                doi = doi.rstrip("/")
                # Remove any trailing file extensions or parameters
                doi = re.sub(r"[?#].*$", "", doi)
                doi = re.sub(r"\.(pdf|html|xml)$", "", doi)
                return doi

        return None

    def _doi_to_pii(self, doi: str) -> Optional[str]:
        """Convert Elsevier DOI to PII (Publication Item Identifier)."""
        # Example: 10.1016/b978-0-323-90476-6.00019-4 -> B9780323904766000194
        if not doi.startswith("10.1016/"):
            return None

        # Remove prefix
        suffix = doi.replace("10.1016/", "")

        # For book chapters, handle the format carefully
        if suffix.lower().startswith("b978"):
            # Remove 'b' prefix and process
            isbn_and_chapter = suffix[1:]  # Remove 'b'
            # Split on the dot before chapter number
            parts = isbn_and_chapter.split(".")
            if len(parts) == 2:
                isbn_part = parts[0].replace("-", "")  # Remove hyphens from ISBN
                chapter_part = parts[1].replace("-", "")  # Remove hyphens from chapter
                return f"B{isbn_part}{chapter_part}".upper()

        # For journal articles (starting with S)
        elif suffix.lower().startswith("s"):
            return suffix.replace("-", "").replace(".", "").upper()

        # Default handling
        return suffix.replace("-", "").replace(".", "").upper()

    def _parse_sciencedirect_page(self, html: str, metadata: Dict[str, Any]) -> None:
        """Parse ScienceDirect page for metadata."""
        soup = BeautifulSoup(html, "html.parser")

        # Extract abstract
        if not metadata.get("abstract"):
            # Try multiple selectors for abstract
            abstract_selectors = [
                ("div", {"class": "abstract author"}),
                ("div", {"id": "abstracts"}),
                ("section", {"id": "abs0010"}),
                ("div", {"class": "abstract-content"}),
                ("div", {"class": "Abstracts"}),
            ]

            for tag, attrs in abstract_selectors:
                abstract_elem = soup.find(tag, attrs)
                if abstract_elem:
                    # Find the actual abstract text, skipping headers
                    abstract_paras = abstract_elem.find_all("p")
                    if abstract_paras:
                        abstract_text = " ".join(
                            [p.get_text(strip=True) for p in abstract_paras]
                        )
                    else:
                        abstract_text = abstract_elem.get_text(strip=True)

                    # Remove "Abstract" label if present
                    abstract_text = re.sub(
                        r"^Abstract:?\s*", "", abstract_text, flags=re.I
                    )

                    if abstract_text and len(abstract_text) > 50:
                        metadata["abstract"] = abstract_text
                        break

        # Extract keywords
        if not metadata.get("keywords"):
            keywords = []
            # Try different keyword containers
            keyword_containers = soup.find_all("div", {"class": "keyword"})
            if not keyword_containers:
                keyword_containers = soup.find_all("span", {"class": "keyword"})

            for kw_elem in keyword_containers:
                kw_text = kw_elem.get_text(strip=True)
                if kw_text and kw_text not in keywords:
                    keywords.append(kw_text)

            if keywords:
                metadata["keywords"] = keywords
                metadata["keyword_source"] = "publisher"

    async def _enrich_from_wiley(
        self,
        session: aiohttp.ClientSession,
        entry: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> bool:
        """Enrich from Wiley Online Library."""
        try:
            title = entry.get("title", "")
            if not title:
                return False

            # Search Wiley
            search_url = f"https://onlinelibrary.wiley.com/action/doSearch?AllField={quote_plus(title)}"

            await self._rate_limit("wiley")

            async with session.get(
                search_url, timeout=self.config.get("preferences.timeout", 30)
            ) as response:
                if response.status != 200:
                    return False

                html = await response.text()
                soup = BeautifulSoup(html, "html.parser")

                # Find first matching result
                results = soup.find_all("a", {"class": "publication_title"})

                for result in results[:3]:
                    result_title = result.get_text(strip=True).lower()

                    if self._calculate_similarity(title.lower(), result_title) > 0.8:
                        chapter_url = "https://onlinelibrary.wiley.com" + result.get(
                            "href", ""
                        )
                        metadata["publisher_url"] = chapter_url

                        # Extract DOI from URL
                        doi_match = re.search(r"10\.1002/[^/\s]+", chapter_url)
                        if doi_match:
                            metadata["doi"] = doi_match.group(0)

                        return True

                return False

        except Exception as e:
            if self.verbose >= 3:
                print(f"[ENRICH] Wiley error: {e}")
            return False

    def _parse_wiley_page(self, html: str, metadata: Dict[str, Any]) -> None:
        """Parse Wiley chapter page for metadata."""
        soup = BeautifulSoup(html, "html.parser")

        # Extract abstract
        if not metadata.get("abstract"):
            abstract_section = soup.find(
                "section", {"class": "article-section__abstract"}
            )
            if not abstract_section:
                abstract_section = soup.find("div", {"class": "abstract"})
            if abstract_section:
                # Remove the abstract heading
                heading = abstract_section.find(["h2", "h3"])
                if heading:
                    heading.decompose()
                abstract_text = abstract_section.get_text(strip=True)
                if abstract_text:
                    metadata["abstract"] = abstract_text

        # Extract keywords
        if not metadata.get("keywords"):
            keywords = []
            keywords_section = soup.find(
                "section", {"class": "article-section__keywords"}
            )
            if keywords_section:
                keyword_links = keywords_section.find_all("a")
                for link in keyword_links:
                    kw_text = link.get_text(strip=True)
                    if kw_text and kw_text not in keywords:
                        keywords.append(kw_text)

            if keywords:
                metadata["keywords"] = keywords
                metadata["keyword_source"] = "publisher"

    async def _enrich_from_scholar(
        self,
        session: aiohttp.ClientSession,
        entry: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> bool:
        """Try to enrich from Google Scholar as fallback."""
        try:
            title = entry.get("title", "")
            if not title:
                return False

            await self._rate_limit("scholar")

            # Run scholarly in executor since it's synchronous
            loop = asyncio.get_event_loop()

            def search_scholar():
                try:
                    # Search Google Scholar
                    search_query = scholarly.search_pubs(title)

                    # Get first result only
                    result = next(search_query, None)
                    if not result:
                        return None

                    # Check title similarity
                    result_title = result.get("bib", {}).get("title", "").lower()
                    if self._calculate_similarity(title.lower(), result_title) > 0.8:
                        # Get full publication info
                        pub = scholarly.fill(result)
                        return pub

                    return None
                except Exception as e:
                    if self.verbose >= 3:
                        print(f"[ENRICH] Scholar search error: {e}")
                    return None

            # Run in executor with timeout
            pub = await asyncio.wait_for(
                loop.run_in_executor(None, search_scholar),
                timeout=20.0,  # 20 seconds for scholar search
            )

            if pub:
                # Extract metadata
                bib = pub.get("bib", {})

                if "abstract" in bib:
                    metadata["abstract"] = bib["abstract"]

                if "pub_url" in bib:
                    metadata["publisher_url"] = bib["pub_url"]

                return True

            return False

        except asyncio.TimeoutError:
            if self.verbose >= 2:
                print("[ENRICH] Scholar search timed out")
            return False
        except Exception as e:
            if self.verbose >= 3:
                print(f"[ENRICH] Scholar error: {e}")
            return False

    async def _enrich_from_llm(
        self,
        session: aiohttp.ClientSession,
        entry: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> bool:
        """Try to enrich using LLM-based intelligent extraction."""
        if not self.llm_enricher or not self.llm_enricher.enabled:
            return False

        try:
            if self.verbose >= 2:
                print("[ENRICH] Using LLM enricher for intelligent extraction")

            # Define search function using multiple academic search engines
            async def search_academic_sources(query: str) -> List[Dict[str, Any]]:
                """Search multiple academic sources and return combined results."""
                try:
                    # Initialize search engines
                    search_engines = AcademicSearchEngines(session, self.verbose)

                    # Run all searches in parallel
                    all_results = await search_engines.search_all(entry)

                    # Combine results from all sources
                    combined_results = []

                    # Prioritize results: general_web (first result) > CrossRef > Semantic Scholar > others
                    # Give special priority to the first web search result as user mentioned
                    if "general_web" in all_results and all_results["general_web"]:
                        # Take the first web result with highest priority
                        combined_results.extend(all_results["general_web"][:2])

                    for source in ["crossref", "semantic_scholar", "unpaywall"]:
                        if source in all_results:
                            combined_results.extend(
                                all_results[source][:3]
                            )  # Take top 3 from each

                    # Add remaining web results
                    if (
                        "general_web" in all_results
                        and len(all_results["general_web"]) > 2
                    ):
                        combined_results.extend(all_results["general_web"][2:5])

                    # Add other sources
                    for source in ["direct_publisher", "restricted_web"]:
                        if source in all_results:
                            combined_results.extend(all_results[source][:2])

                    # Remove duplicates based on title similarity
                    seen_titles = set()
                    unique_results = []
                    for result in combined_results:
                        title_lower = result.get("title", "").lower()
                        if title_lower and title_lower not in seen_titles:
                            seen_titles.add(title_lower)
                            unique_results.append(result)

                    if self.verbose >= 3:
                        print(
                            f"[LLM_ENRICHER] Found {len(unique_results)} unique results from all sources"
                        )
                        for source, results in all_results.items():
                            if results:
                                print(f"  - {source}: {len(results)} results")

                    return unique_results[:10]  # Return top 10 results

                except Exception as e:
                    if self.verbose >= 3:
                        print(f"[LLM_ENRICHER] Search error: {e}")
                    return []

            # Define fetch function
            async def fetch_page(url: str) -> Optional[str]:
                """Fetch page content."""
                try:
                    await self._rate_limit("url")

                    async with session.get(
                        url, timeout=self.config.get("preferences.timeout", 30)
                    ) as response:
                        if response.status == 200:
                            return await response.text()
                        return None
                except Exception as e:
                    if self.verbose >= 3:
                        print(f"[LLM_ENRICHER] Fetch error: {e}")
                    return None

            # Use LLM enricher's complete workflow
            llm_metadata = await self.llm_enricher.search_and_extract(
                entry, search_academic_sources, fetch_page
            )

            # Update metadata with LLM results
            success = False
            for key in ["doi", "abstract", "keywords", "publisher_url", "isbn"]:
                if llm_metadata.get(key):
                    metadata[key] = llm_metadata[key]
                    success = True

            if llm_metadata.get("keyword_source"):
                metadata["keyword_source"] = llm_metadata["keyword_source"]

            if success and self.verbose >= 2:
                print("[LLM_ENRICHER] Successfully enriched with LLM")
                if metadata.get("doi"):
                    print(f"  - DOI: {metadata['doi']}")
                if metadata.get("abstract"):
                    print(f"  - Abstract: {len(metadata['abstract'])} chars")
                if metadata.get("keywords"):
                    print(
                        f"  - Keywords: {len(metadata['keywords'])} ({metadata.get('keyword_source', 'unknown')} source)"
                    )

            return success

        except Exception as e:
            if self.verbose >= 1:
                print(f"[ENRICH] LLM enricher error: {e}")
            return False

    async def _rate_limit(self, service: str) -> None:
        """Apply rate limiting for a service."""
        current_time = time.time()

        if service in self.last_request_time:
            elapsed = current_time - self.last_request_time[service]
            if elapsed < self.delay:
                await asyncio.sleep(self.delay - elapsed)

        self.last_request_time[service] = time.time()

    async def _fetch_from_publisher_url(
        self, session: aiohttp.ClientSession, metadata: Dict[str, Any]
    ) -> None:
        """Fetch abstract and keywords from publisher URL."""
        url = metadata.get("publisher_url", "")
        if not url:
            return

        doi = metadata.get("doi", "")

        try:
            if self.verbose >= 2:
                print(f"[ENRICH] Fetching metadata from publisher URL: {url}")

            await self._rate_limit("publisher")

            async with session.get(
                url, timeout=self.config.get("preferences.timeout", 30)
            ) as response:
                if response.status == 200:
                    html = await response.text()

                    if self.verbose >= 3:
                        print(
                            f"[ENRICH] Successfully fetched publisher page ({len(html)} bytes)"
                        )

                    # Determine publisher and use appropriate parser
                    if doi.startswith("10.1007/") or "springer.com" in url:
                        if self.verbose >= 3:
                            print("[ENRICH] Using Springer parser")
                        self._parse_springer_chapter(html, metadata)
                    elif doi.startswith("10.1016/") or "sciencedirect.com" in url:
                        if self.verbose >= 3:
                            print("[ENRICH] Using ScienceDirect parser")
                        self._parse_sciencedirect_page(html, metadata)
                    elif doi.startswith("10.1002/") or "wiley.com" in url:
                        if self.verbose >= 3:
                            print("[ENRICH] Using Wiley parser")
                        self._parse_wiley_page(html, metadata)
                    else:
                        if self.verbose >= 3:
                            print("[ENRICH] Using generic parser")
                        self._parse_generic_publisher_page(html, metadata)

                    if self.verbose >= 2:
                        if metadata.get("abstract"):
                            print(
                                f"[ENRICH] Found abstract: {len(metadata['abstract'])} chars"
                            )
                        else:
                            print("[ENRICH] No abstract found on publisher page")
                        if metadata.get("keywords"):
                            print(
                                f"[ENRICH] Found {len(metadata['keywords'])} keywords from publisher"
                            )
                else:
                    if self.verbose >= 2:
                        if response.status == 403:
                            print(
                                "[ENRICH] Publisher blocked access (HTTP 403) - this is common for automated requests"
                            )
                            print(
                                "[ENRICH] Will use alternative sources (CrossRef, Google Scholar, etc.)"
                            )
                        else:
                            print(
                                f"[ENRICH] Failed to fetch publisher page: HTTP {response.status}"
                            )
        except Exception as e:
            if self.verbose >= 3:
                print(f"[ENRICH] Error fetching publisher page: {e}")

    def _parse_generic_publisher_page(
        self, html: str, metadata: Dict[str, Any]
    ) -> None:
        """Parse generic publisher page using common meta tags."""
        soup = BeautifulSoup(html, "html.parser")

        # Extract abstract
        if not metadata.get("abstract"):
            # Try various meta tags
            abstract_meta = (
                soup.find("meta", {"name": "citation_abstract"})
                or soup.find("meta", {"name": "DC.Description"})
                or soup.find("meta", {"property": "og:description"})
            )
            if abstract_meta:
                abstract_text = abstract_meta.get("content", "").strip()
                if (
                    abstract_text and len(abstract_text) > 100
                ):  # Ensure it's not just a short description
                    metadata["abstract"] = abstract_text

            # Try finding abstract in content
            if not metadata.get("abstract"):
                abstract_selectors = [
                    ("div", {"class": re.compile(r"abstract", re.I)}),
                    ("section", {"class": re.compile(r"abstract", re.I)}),
                    ("div", {"id": re.compile(r"abstract", re.I)}),
                    ("p", {"class": "abstract"}),
                ]

                for tag, attrs in abstract_selectors:
                    abstract_elem = soup.find(tag, attrs)
                    if abstract_elem:
                        abstract_text = abstract_elem.get_text(strip=True)
                        # Remove "Abstract" label if present
                        abstract_text = re.sub(
                            r"^Abstract:?\s*", "", abstract_text, flags=re.I
                        )
                        if abstract_text and len(abstract_text) > 100:
                            metadata["abstract"] = abstract_text
                            break

        # Extract keywords
        if not metadata.get("keywords"):
            keywords = []

            # Try meta tags
            keyword_metas = soup.find_all(
                "meta", {"name": re.compile(r"keywords?", re.I)}
            ) + soup.find_all("meta", {"name": "citation_keywords"})

            for meta in keyword_metas:
                content = meta.get("content", "")
                # Split by comma or semicolon
                kws = re.split(r"[,;]", content)
                keywords.extend(
                    [k.strip() for k in kws if k.strip() and len(k.strip()) > 2]
                )

            # Remove duplicates and limit
            if keywords:
                seen = set()
                unique_keywords = []
                for kw in keywords:
                    kw_lower = kw.lower()
                    if kw_lower not in seen:
                        seen.add(kw_lower)
                        unique_keywords.append(kw)

                metadata["keywords"] = unique_keywords[:20]  # Limit to 20 keywords
                metadata["keyword_source"] = "publisher"

    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate simple string similarity (Jaccard index)."""
        # Tokenize and normalize
        tokens1 = set(re.findall(r"\w+", str1.lower()))
        tokens2 = set(re.findall(r"\w+", str2.lower()))

        if not tokens1 or not tokens2:
            return 0.0

        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)

        return len(intersection) / len(union)
