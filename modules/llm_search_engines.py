# -*- coding: utf-8 -*-
"""
Alternative search engines for finding academic publications.

This module provides various search methods beyond Google Scholar
for finding publisher URLs and metadata.

Author: Dr. Denys Dutykh (Khalifa University of Science and Technology, Abu Dhabi, UAE)
Date: 2025-01-01
"""

import asyncio
import aiohttp
import re
from typing import List, Dict, Any, Optional
from urllib.parse import quote_plus, urlencode


class AcademicSearchEngines:
    """Collection of search methods for academic publications."""

    # Comprehensive list of academic publisher domains
    ACADEMIC_DOMAINS = [
        # Major publishers
        "springer.com",
        "link.springer.com",
        "sciencedirect.com",
        "elsevier.com",
        "wiley.com",
        "onlinelibrary.wiley.com",
        "ieee.org",
        "ieeexplore.ieee.org",
        "acm.org",
        "dl.acm.org",
        "nature.com",
        "science.org",
        "sciencemag.org",
        "cell.com",
        "tandfonline.com",  # Taylor & Francis
        "sagepub.com",
        "mdpi.com",
        "hindawi.com",
        "frontiersin.org",
        "plos.org",
        "journals.plos.org",
        # Physics and Math publishers
        "aps.org",
        "journals.aps.org",  # American Physical Society
        "iopscience.iop.org",
        "iop.org",  # IOP Publishing
        "aip.org",
        "pubs.aip.org",  # American Institute of Physics
        "ams.org",  # American Mathematical Society
        "siam.org",  # SIAM
        # Chemistry publishers
        "acs.org",
        "pubs.acs.org",  # American Chemical Society
        "rsc.org",
        "pubs.rsc.org",  # Royal Society of Chemistry
        # University presses
        "cambridge.org",
        "oup.com",
        "academic.oup.com",  # Oxford
        "mit.edu",
        "mitpressjournals.org",
        "uchicago.edu",
        "journals.uchicago.edu",
        # Other academic publishers
        "royalsocietypublishing.org",
        "iucr.org",  # International Union of Crystallography
        "annualreviews.org",
        "bioone.org",
        "degruyter.com",
        "emerald.com",
        "karger.com",
        "thieme-connect.com",
        "worldscientific.com",
        "igi-global.com",
        "inderscience.com",
        "inderscienceonline.com",
        # Open access
        "doaj.org",
        "arxiv.org",
        "biorxiv.org",
        "chemrxiv.org",
        "medrxiv.org",
        "ssrn.com",
        "hal.archives-ouvertes.fr",
        "hal.science",
        # Aggregators
        "jstor.org",
        "projecteuclid.org",
        "muse.jhu.edu",  # Project MUSE
        "ingentaconnect.com",
        "pubmed.ncbi.nlm.nih.gov",
        "europepmc.org",
    ]

    def __init__(self, session: aiohttp.ClientSession, verbose: int = 2):
        """Initialize search engines."""
        self.session = session
        self.verbose = verbose

    async def search_crossref(self, entry: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Search CrossRef for publication metadata.

        CrossRef is very reliable and has good API limits.
        """
        try:
            title = entry.get("title", "")
            if not title:
                return []

            # Build query
            params = {
                "query.bibliographic": title,
                "rows": 5,
                "select": "DOI,title,author,published-print,container-title,link,abstract",
            }

            # Add author if available
            if entry.get("authors"):
                first_author = entry["authors"][0].get("lastName", "")
                if first_author:
                    params["query.author"] = first_author

            url = f"https://api.crossref.org/works?{urlencode(params)}"

            if self.verbose >= 3:
                print(f"[SEARCH] CrossRef query: {title[:50]}...")

            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    items = data.get("message", {}).get("items", [])

                    results = []
                    for item in items:
                        # Extract authors
                        authors = []
                        for author in item.get("author", []):
                            name = f"{author.get('given', '')} {author.get('family', '')}".strip()
                            if name:
                                authors.append(name)

                        # Extract URL - prefer publisher link
                        url = ""
                        links = item.get("link", [])
                        for link in links:
                            if link.get("content-type") == "text/html":
                                url = link.get("URL", "")
                                break

                        result = {
                            "title": item.get("title", [""])[0]
                            if item.get("title")
                            else "",
                            "authors": authors,
                            "year": str(
                                item.get("published-print", {}).get(
                                    "date-parts", [[None]]
                                )[0][0]
                                or ""
                            ),
                            "doi": item.get("DOI", ""),
                            "url": url,
                            "venue": item.get("container-title", [""])[0]
                            if item.get("container-title")
                            else "",
                            "abstract": item.get("abstract", ""),
                            "source": "crossref",
                        }
                        results.append(result)

                    if self.verbose >= 3:
                        print(f"[SEARCH] CrossRef found {len(results)} results")

                    return results

        except Exception as e:
            if self.verbose >= 2:
                print(f"[SEARCH] CrossRef error: {e}")

        return []

    async def search_semantic_scholar(
        self, entry: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Search Semantic Scholar API.

        Free, no API key required, good for CS and AI papers.
        """
        try:
            title = entry.get("title", "")
            if not title:
                return []

            # Build search query
            query = quote_plus(title)
            url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&limit=5&fields=title,authors,year,abstract,url,externalIds,venue"

            if self.verbose >= 3:
                print(f"[SEARCH] Semantic Scholar query: {title[:50]}...")

            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    papers = data.get("data", [])

                    results = []
                    for paper in papers:
                        # Extract authors
                        authors = [
                            author["name"] for author in paper.get("authors", [])
                        ]

                        # Extract DOI from external IDs
                        doi = ""
                        external_ids = paper.get("externalIds", {})
                        if "DOI" in external_ids:
                            doi = external_ids["DOI"]

                        result = {
                            "title": paper.get("title", ""),
                            "authors": authors,
                            "year": str(paper.get("year", "")),
                            "doi": doi,
                            "url": paper.get("url", ""),
                            "venue": paper.get("venue", ""),
                            "abstract": paper.get("abstract", ""),
                            "source": "semantic_scholar",
                        }
                        results.append(result)

                    if self.verbose >= 3:
                        print(f"[SEARCH] Semantic Scholar found {len(results)} results")

                    return results

        except Exception as e:
            if self.verbose >= 2:
                print(f"[SEARCH] Semantic Scholar error: {e}")

        return []

    async def search_unpaywall(self, doi: str) -> Optional[Dict[str, Any]]:
        """
        Search Unpaywall for open access version and metadata.

        Great for finding free versions and publisher links.
        """
        if not doi:
            return None

        try:
            # Unpaywall requires email in query
            email = "metadata@example.com"  # Generic email for metadata extraction
            url = f"https://api.unpaywall.org/v2/{doi}?email={email}"

            if self.verbose >= 3:
                print(f"[SEARCH] Unpaywall query for DOI: {doi}")

            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()

                    # Get best OA location or publisher URL
                    best_location = data.get("best_oa_location", {})
                    publisher_url = (
                        best_location.get("url_for_landing_page", "")
                        or best_location.get("url", "")
                        or data.get("doi_url", "")
                    )

                    result = {
                        "title": data.get("title", ""),
                        "doi": data.get("doi", ""),
                        "url": publisher_url,
                        "year": str(data.get("year", "")),
                        "publisher": data.get("publisher", ""),
                        "is_oa": data.get("is_oa", False),
                        "source": "unpaywall",
                    }

                    if self.verbose >= 3:
                        print(f"[SEARCH] Unpaywall found: {result['url']}")

                    return result

        except Exception as e:
            if self.verbose >= 3:
                print(f"[SEARCH] Unpaywall error: {e}")

        return None

    async def search_microsoft_academic(
        self, entry: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Search using Bing Web Search for academic content.

        More reliable than Google Scholar for programmatic access.
        """
        try:
            # Build academic search query
            title = entry.get("title", "")
            authors = entry.get("authors", [])
            year = entry.get("year", "")

            # Construct search query
            query_parts = [f'"{title}"']
            if authors:
                first_author = authors[0].get("lastName", "")
                if first_author:
                    query_parts.append(first_author)
            if year:
                query_parts.append(str(year))

            # Add academic sites
            query_parts.append(
                "(site:springer.com OR site:sciencedirect.com OR site:ieee.org OR site:acm.org OR site:wiley.com)"
            )

            query = " ".join(query_parts)

            # Use DuckDuckGo HTML search (no API key needed)
            url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"

            if self.verbose >= 3:
                print(f"[SEARCH] Web search query: {query[:100]}...")

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }

            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    html = await response.text()

                    # Extract results using regex (simple HTML parsing)
                    results = []

                    # Find result links
                    link_pattern = r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>([^<]+)</a>'
                    matches = re.finditer(link_pattern, html, re.IGNORECASE)

                    for i, match in enumerate(matches):
                        if i >= 5:  # Limit to 5 results
                            break

                        result_url = match.group(1)
                        result_title = match.group(2)

                        # Filter for academic publishers
                        if any(
                            domain in result_url
                            for domain in [
                                "springer.com",
                                "sciencedirect.com",
                                "ieee.org",
                                "acm.org",
                                "wiley.com",
                            ]
                        ):
                            result = {
                                "title": result_title,
                                "url": result_url,
                                "source": "web_search",
                            }
                            results.append(result)

                    if self.verbose >= 3:
                        print(
                            f"[SEARCH] Web search found {len(results)} academic results"
                        )

                    return results

        except Exception as e:
            if self.verbose >= 2:
                print(f"[SEARCH] Web search error: {e}")

        return []

    async def direct_publisher_search(
        self, entry: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Search directly on publisher websites based on publisher name.
        """
        publisher = entry.get("publisher", {}).get("name", "").lower()
        title = entry.get("title", "")

        if not publisher or not title:
            return []

        results = []

        try:
            # Springer
            if "springer" in publisher:
                search_url = (
                    f"https://link.springer.com/search?query={quote_plus(title)}"
                )
                results.append(
                    {
                        "title": title,
                        "url": search_url,
                        "source": "springer_direct",
                        "search_url": True,
                    }
                )

            # Elsevier
            elif "elsevier" in publisher:
                search_url = (
                    f"https://www.sciencedirect.com/search?qs={quote_plus(title)}"
                )
                results.append(
                    {
                        "title": title,
                        "url": search_url,
                        "source": "elsevier_direct",
                        "search_url": True,
                    }
                )

            # Wiley
            elif "wiley" in publisher:
                search_url = f"https://onlinelibrary.wiley.com/action/doSearch?AllField={quote_plus(title)}"
                results.append(
                    {
                        "title": title,
                        "url": search_url,
                        "source": "wiley_direct",
                        "search_url": True,
                    }
                )

            if results and self.verbose >= 3:
                print("[SEARCH] Direct publisher search URL generated")

        except Exception as e:
            if self.verbose >= 2:
                print(f"[SEARCH] Direct publisher search error: {e}")

        return results

    async def search_all(
        self, entry: Dict[str, Any]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run all search methods in parallel and return combined results.
        """
        # Create tasks for all search methods
        tasks = [
            ("general_web", self.search_web(entry)),  # NEW: General web search first
            ("crossref", self.search_crossref(entry)),
            ("semantic_scholar", self.search_semantic_scholar(entry)),
            (
                "restricted_web",
                self.search_microsoft_academic(entry),
            ),  # Renamed for clarity
            ("direct_publisher", self.direct_publisher_search(entry)),
        ]

        # Run all searches in parallel
        results = {}
        search_results = await asyncio.gather(
            *[task[1] for task in tasks], return_exceptions=True
        )

        # Collect results
        for (name, _), result in zip(tasks, search_results):
            if isinstance(result, Exception):
                if self.verbose >= 2:
                    print(f"[SEARCH] {name} failed: {result}")
                results[name] = []
            else:
                results[name] = result

        # Process general web results to extract DOIs from first results
        if "general_web" in results and results["general_web"]:
            # Focus on the first few results, especially from academic sources
            for i, web_result in enumerate(results["general_web"][:3]):
                if not web_result.get("doi") and web_result.get("url"):
                    # Try to extract DOI from URL
                    extracted_doi = self.extract_doi_from_url(web_result["url"])
                    if extracted_doi:
                        web_result["doi"] = extracted_doi
                        if self.verbose >= 3:
                            print(
                                f"[SEARCH] Extracted DOI from web result #{i + 1}: {extracted_doi}"
                            )

        # If we found a DOI from any source, also try Unpaywall
        for source_results in results.values():
            for result in source_results:
                if result.get("doi"):
                    unpaywall_result = await self.search_unpaywall(result["doi"])
                    if unpaywall_result:
                        results["unpaywall"] = [unpaywall_result]
                    break
            if "unpaywall" in results:
                break

        return results

    def extract_doi_from_url(self, url: str) -> Optional[str]:
        """
        Extract DOI from various URL formats.

        Examples:
        - https://doi.org/10.1103/PhysRevResearch.6.033284
        - https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.6.033284
        - https://link.springer.com/article/10.1007/s11069-023-06071-1
        """
        if not url:
            return None

        # Common DOI patterns - improved to handle more formats
        doi_patterns = [
            r"(?:doi\.org/|doi/|/)(10\.\d{4,}/[-._;()/:\w]+)",
            r"/abstract/(10\.\d{4,}/[-._;()/:\w]+)",  # APS abstract URLs
            r"journals\.aps\.org/[^/]+/(?:abstract|pdf|article)/(10\.\d{4,}/[-._;()/:\w]+)",  # APS journal URLs
            r"link\.aps\.org/doi/(10\.\d{4,}/[-._;()/:\w]+)",  # APS link URLs
            r"/article/(10\.\d{4,}/[-._;()/:\w]+)",  # General article URLs
            r"/chapter/(10\.\d{4,}/[-._;()/:\w]+)",  # Chapter URLs
            r"/(10\.\d{4,}/[-._;()/:\w]+)(?:\?|$|#)",
            r"doi[:=]\s*(10\.\d{4,}/[-._;()/:\w]+)",
        ]

        for pattern in doi_patterns:
            match = re.search(pattern, url, re.IGNORECASE)
            if match:
                doi = match.group(1)
                # Clean up common artifacts
                doi = doi.rstrip("/")
                doi = doi.rstrip(".")
                # Remove any file extensions or query parameters
                doi = re.sub(r"\.(pdf|html|xml)$", "", doi)
                doi = re.sub(r"[?#].*$", "", doi)
                if self.verbose >= 3:
                    print(f"[SEARCH] Extracted DOI from URL: {doi}")
                return doi

        return None

    def is_academic_url(self, url: str) -> bool:
        """Check if URL is from a known academic publisher."""
        if not url:
            return False

        url_lower = url.lower()
        return any(domain in url_lower for domain in self.ACADEMIC_DOMAINS)

    async def search_web(self, entry: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        General web search without domain restrictions.
        Uses DuckDuckGo for better privacy and no rate limits.
        """
        try:
            title = entry.get("title", "")
            if not title:
                return []

            # Build search query
            query_parts = [f'"{title}"']

            # Add first author's last name for precision
            authors = entry.get("authors", [])
            if authors and authors[0].get("lastName"):
                query_parts.append(authors[0]["lastName"])

            # Add year if available
            year = entry.get("year")
            if year:
                query_parts.append(str(year))

            # Add journal name if available (for journal papers)
            journal = entry.get("journal")
            if journal:
                query_parts.append(f'"{journal}"')

            query = " ".join(query_parts)

            if self.verbose >= 3:
                print(f"[SEARCH] General web search: {query[:100]}...")

            # Use DuckDuckGo HTML interface
            url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }

            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    html = await response.text()

                    results = []

                    # Extract all result links
                    link_pattern = r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>([^<]+)</a>'
                    matches = re.finditer(link_pattern, html, re.IGNORECASE | re.DOTALL)

                    for i, match in enumerate(matches):
                        if i >= 10:  # Get top 10 results
                            break

                        result_url = match.group(1)
                        result_title = match.group(2).strip()

                        # Clean up HTML entities
                        result_title = result_title.replace("&amp;", "&")
                        result_title = result_title.replace("&quot;", '"')
                        result_title = result_title.replace("&#39;", "'")

                        # Check if it's an academic URL
                        is_academic = self.is_academic_url(result_url)

                        # Try to extract DOI
                        doi = self.extract_doi_from_url(result_url)

                        result = {
                            "title": result_title,
                            "url": result_url,
                            "doi": doi or "",
                            "is_academic": is_academic,
                            "source": "web_search",
                            "rank": i + 1,  # Search rank for prioritization
                        }

                        results.append(result)

                        if self.verbose >= 3 and is_academic:
                            print(
                                f"[SEARCH] Found academic result #{i + 1}: {result_url[:80]}..."
                            )

                    # Sort results: academic URLs first, then by rank
                    results.sort(key=lambda x: (not x["is_academic"], x["rank"]))

                    if self.verbose >= 2:
                        academic_count = sum(1 for r in results if r["is_academic"])
                        print(
                            f"[SEARCH] Web search found {len(results)} results ({academic_count} academic)"
                        )

                    return results

        except Exception as e:
            if self.verbose >= 2:
                print(f"[SEARCH] Web search error: {e}")

        return []

    def construct_publisher_url(
        self, entry: Dict[str, Any]
    ) -> Optional[Dict[str, str]]:
        """
        Construct publisher URL directly from journal information for known publishers.

        This is useful when we have journal name, volume, and article number but no DOI.
        Returns a dict with 'url' and optionally 'doi' if it can be constructed.
        """
        journal = entry.get("journal", "").lower()
        volume = entry.get("volume", "")
        issue = entry.get("issue", "")
        pages = entry.get("pages", {})
        year = entry.get("year", "")

        if not journal or not volume:
            return None

        # Get article number or start page
        article = pages.get("article_number") or pages.get("start", "")
        if not article:
            return None

        # APS journals (Physical Review series)
        aps_journals = {
            "phys. rev. lett.": "prl",
            "phys. rev. a": "pra",
            "phys. rev. b": "prb",
            "phys. rev. c": "prc",
            "phys. rev. d": "prd",
            "phys. rev. e": "pre",
            "phys. rev. x": "prx",
            "phys. rev. res.": "prresearch",
            "phys. rev. research": "prresearch",
            "rev. mod. phys.": "rmp",
            "physical review letters": "prl",
            "physical review a": "pra",
            "physical review b": "prb",
            "physical review c": "prc",
            "physical review d": "prd",
            "physical review e": "pre",
            "physical review x": "prx",
            "physical review research": "prresearch",
            "reviews of modern physics": "rmp",
        }

        for journal_pattern, journal_code in aps_journals.items():
            if journal_pattern in journal:
                # Construct APS URL
                # Format: https://journals.aps.org/{journal_code}/abstract/10.1103/{Journal}.{volume}.{article}
                journal_name_parts = {
                    "prl": "PhysRevLett",
                    "pra": "PhysRevA",
                    "prb": "PhysRevB",
                    "prc": "PhysRevC",
                    "prd": "PhysRevD",
                    "pre": "PhysRevE",
                    "prx": "PhysRevX",
                    "prresearch": "PhysRevResearch",
                    "rmp": "RevModPhys",
                }
                journal_full = journal_name_parts.get(journal_code, "")
                if journal_full:
                    doi = f"10.1103/{journal_full}.{volume}.{article}"
                    url = f"https://journals.aps.org/{journal_code}/abstract/{doi}"
                    if self.verbose >= 3:
                        print(f"[SEARCH] Constructed APS URL: {url}")
                    return {"url": url, "doi": doi}

        # IOP journals
        iop_journals = {
            "class. quantum gravity": "0264-9381",
            "classical and quantum gravity": "0264-9381",
            "j. phys. a": "1751-8113",
            "journal of physics a": "1751-8113",
            "new j. phys.": "1367-2630",
            "new journal of physics": "1367-2630",
            "eur. j. phys.": "0143-0807",
            "european journal of physics": "0143-0807",
        }

        for journal_pattern, issn in iop_journals.items():
            if journal_pattern in journal:
                # IOP URL format
                url = f"https://iopscience.iop.org/article/10.1088/{issn}/{volume}/{issue or '1'}/{article}"
                if self.verbose >= 3:
                    print(f"[SEARCH] Constructed IOP URL: {url}")
                return {"url": url, "doi": None}

        # Nature journals
        if "nature" in journal:
            # Nature URL format
            url = f"https://www.nature.com/articles/s41586-{year[-2:]}-{article}"
            if self.verbose >= 3:
                print(f"[SEARCH] Constructed Nature URL: {url}")
            return {"url": url, "doi": None}

        # Science journals
        if "science" in journal and "sciencedirect" not in journal:
            # Science URL format
            url = f"https://www.science.org/doi/10.1126/science.{article}"
            if self.verbose >= 3:
                print(f"[SEARCH] Constructed Science URL: {url}")
            return {"url": url, "doi": None}

        return None
