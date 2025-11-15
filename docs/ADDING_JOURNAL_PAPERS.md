# Adding Journal Papers to `papers.json`

> Author: Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science and Technology, Abu Dhabi, UAE)

Use this guide when appending a new journal article to `output_data/papers.json`. Matching the structure that `src/pub2json.py` produces keeps automation (keyword generation, publisher identification, cache lookups) reliable across the toolchain.

## Workflow Overview
- **1. Gather metadata.** Collect every item in the checklist below before touching the JSON file.
- **2. Fill the template.** Copy `templates/journal_paper_template.jsonc`, replace placeholders with real values, and delete the inline comments.
- **3. Insert the entry.** Add the finished object to the `journal_papers` array in `output_data/papers.json`, keeping the list sorted with the most recent publications first.
- **4. Validate.** Confirm the result is valid JSON and follows the two-space indentation style. Run `python3 -m json.tool output_data/papers.json > /dev/null` (or `jq . output_data/papers.json >/dev/null`) before committing.

## Metadata Checklist

| Data Point | Required | Notes / Source Suggestions |
| --- | --- | --- |
| Ordered author list (`firstName`, `lastName`) | ✅ | Use the published author order; initials are welcome. |
| Paper title | ✅ | Copy exactly from the journal site (case/punctuation matter). |
| Journal name/abbreviation | ✅ | Match the style already used in the dataset for that journal. |
| Volume & issue | ⚠️ | Store as strings to support prefixes; use `null` when absent. |
| Page range or article number | ⚠️ | Fill `pages.start` / `pages.end` as integers; set `pages.article_number` for e-only articles. |
| Publication year | ✅ | Four-digit integer. |
| DOI | ✅ | Canonical lowercase string; if not assigned, set `null`. |
| Canonical URL (`url`) | ⚠️ | Prefer an open-access link if available. |
| Preprint links (`arxiv_url`, `hal_url`) | ⚠️ | Keep `null` when unused. |
| Abstract text & source | ⚠️ | Paste the official abstract and record whether it came from the publisher, arXiv, etc. |
| Keywords + `keyword_source` | ⚠️ | Use publisher keywords when possible; otherwise label as generated/manual. |
| Open-access flag | ⚠️ | Boolean describing the final journal version’s accessibility. |
| Publisher fields (`publisher_identified`, `publisher_platform`, `publisher_url`) | ⚠️ | Follow the naming used by `src/pub2json.py` (e.g., “Elsevier”, “ScienceDirect”). |
| Confidence scores (`search_confidence`, `publisher_confidence`) | ⚠️ | Floats between 0 and 1 that encode how confident the tooling is about the sources. |
| Provenance (`extraction_method`, `abstract_source`, `doi_source`) | ⚠️ | Text labels such as `crossref`, `manual`, `publisher_scrape`, `preprint`. |
| Alternative URLs list | ⚠️ | Each entry stores `{url, type, confidence}` for mirrors (publisher, preprint, repository, dataset, etc.). |
| Additional metadata | ⚠️ | Flexible `{}` object for ISSNs, submission dates, licenses, arXiv subjects, etc.; keep the field even if empty. |
| ISBN | ⚠️ | Rare for journal issues; keep the `isbn` field with `null` or `""` when not provided. |

> ⚠️ **Tip:** Preserve every field even when you lack the data. Set numbers/strings to `null` (or an empty string where legacy entries already do so) so that downstream code can rely on the schema.

## JSON Structure

`output_data/papers.json` is a single object shaped like:

```json
{
  "journal_papers": [
    {
      /* individual journal entries */
    }
  ]
}
```

Maintain two-space indentation and JSON commas that match the surrounding entries.

## Field Reference

### Core Bibliography

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `authors` | array of `{ firstName, lastName }` | ✅ | Ordered author list. |
| `title` | string | ✅ | Full article title. |
| `journal` | string | ✅ | Journal name (abbreviated or full, consistent with prior entries). |
| `volume` | string\|int\|null | ⚠️ | Free-form volume identifier. |
| `issue` | string\|int\|null | ⚠️ | Issue or number label. |
| `pages` | object | ⚠️ | Supports `start`, `end`, and/or `article_number`. Use integers when known; set missing keys to `null`. |
| `year` | int | ✅ | Year of publication. |
| `doi` | string\|null | ✅ | DOI normalized to lowercase. |
| `isbn` | string\|null | ⚠️ | Used for special issues; keep `null` if unused. |

### Access & Links

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `url` | string\|null | ⚠️ | Canonical article link (publisher or open version). |
| `arxiv_url` | string\|null | ⚠️ | arXiv preprint link. |
| `hal_url` | string\|null | ⚠️ | HAL archive link. |
| `publisher_url` | string\|null | ⚠️ | Landing page used by the publisher. |
| `alternative_urls` | array | ⚠️ | Each item is `{ "url": "...", "type": "preprint|publisher|repository|dataset|other", "confidence": 0.9 }`. |
| `open_access` | boolean | ⚠️ | Indicates whether the final article is OA. |

### Content & Keywords

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `abstract` | string\|null | ⚠️ | Single-string abstract text. |
| `keywords` | array of strings | ⚠️ | Ordered keyword list; `[]` if unavailable. |
| `keyword_source` | string\|null | ⚠️ | `publisher`, `generated`, `manual`, etc. |
| `abstract_source` | string\|null | ⚠️ | Where the abstract was fetched from (`publisher`, `preprint`, `manual`). |

### Provenance & Confidence

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `extraction_method` | string\|null | ⚠️ | Process that produced the record (`crossref`, `arxiv_api`, `manual`). |
| `doi_source` | string\|null | ⚠️ | Origin of the DOI (publisher site, preprint, manual). |
| `publisher_identified` | string\|null | ⚠️ | Publisher name inferred by enrichment. |
| `publisher_platform` | string\|null | ⚠️ | Hosting platform (e.g., “ScienceDirect”). |
| `publisher_confidence` | number\|null | ⚠️ | Confidence score (0–1) for publisher identification. |
| `search_confidence` | number\|null | ⚠️ | Confidence score for the web search result. |

### Optional Extras

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `additional_metadata` | object | ⚠️ | Flexible holder for ISSNs, submission dates, licenses, arXiv IDs, etc. Keep it as `{}` if nothing extra exists. |

## Using the Template

1. Copy `templates/journal_paper_template.jsonc`.
2. Replace every placeholder with the real metadata. Unknown values should be `null` (or another neutral value already used in the dataset).
3. Remove all `//` comment lines so the snippet becomes valid JSON.
4. Paste the finished object into the `journal_papers` array, keeping the list ordered by `year` (newest first, and then by publication date where known).
5. Ensure commas/indentation match the surrounding entries.

## Validating Your Changes

```bash
python3 -m json.tool output_data/papers.json > /dev/null
# or
jq . output_data/papers.json > /dev/null
```

Fix any reported line before committing—downstream scripts expect strict JSON.

## Optional: Regenerate via Parser

You can also update `input_data/list_of_papers.tex` and regenerate the entire file:

```bash
python3 src/pub2json.py -i input_data/list_of_papers.tex -o output_data/papers.json --force
```

This reruns the full enrichment stack (LLM parsing, publisher detection, keyword generation). Review the resulting diff carefully to ensure only the intended changes are introduced.
