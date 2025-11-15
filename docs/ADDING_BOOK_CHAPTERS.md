# Adding Book Chapters to `bkchpts.json`

> Author: Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science and Technology, Abu Dhabi, UAE)

This guide walks you through the process of appending a new book chapter entry to `output_data/bkchpts.json`. The goal is to keep the JSON database consistent with what `src/chpts2json.py` emits, so that downstream tooling (keyword generators, web enrichment, etc.) keeps working without surprises.

## Workflow Overview
- **1. Collect metadata.** Gather everything listed in the checklist below before editing the JSON file.
- **2. Fill out the template.** Copy `templates/book_chapter_template.jsonc`, replace the placeholders with real values, and remove the inline comments.
- **3. Insert the entry.** Paste the filled template into the `book_chapters` array inside `output_data/bkchpts.json`, keeping the array in reverse chronological order (most recent years first).
- **4. Validate.** Ensure the file stays valid JSON and matches the indentation style (two spaces). Run `python3 -m json.tool output_data/bkchpts.json > /dev/null` or `jq . output_data/bkchpts.json >/dev/null` before committing.

## Metadata Checklist

| Data Point | Required | Notes / Suggested Source |
| --- | --- | --- |
| Ordered author list (firstName + lastName) | ✅ | Use the publication’s author order; initials are acceptable. |
| Chapter title | ✅ | Exact casing/punctuation from the publisher’s site. |
| Editor list | ✅ | Include all editors credited for the host volume. |
| Book / volume title | ✅ | Often differs from chapter title; include volume numbering if in the official title. |
| Series name & volume number | ⚠️ | Optional. Use `null` if the chapter does not belong to a series or the volume number is unknown. |
| Page range | ✅ | Store as integers under `pages.start` and `pages.end`. |
| Publisher name & location | ✅ | City should match the imprint listed on the publisher page. |
| Publication year | ✅ | 4-digit integer. |
| DOI | ⚠️ | Use lowercase `doi` string; if absent, set to `null`. |
| URLs | ⚠️ | `url` is for an open/preprint version, `publisher_url` for the official landing page. Either can be `null`. |
| Keywords | ⚠️ | Prefer the publisher’s keywords; otherwise generate your own and set `keyword_source` accordingly. |
| ISBNs | ⚠️ | Prefer the `{ "print": "...", "electronic": "..." }` object. Use `null` for any format you cannot find. |
| Chapter number | ⚠️ | Integer where supplied; use `null` otherwise. |
| Abstract | ⚠️ | Paste the official abstract or summary, preserving paragraphs as single strings. |

> ⚠️ **Tip:** When metadata is missing, keep the field but set its value to `null` (or an empty string only if the current dataset already uses that convention for a field, e.g. historic `isbn` entries). This keeps the schema predictable for downstream consumers.

## JSON Structure

`output_data/bkchpts.json` contains one top-level object:

```json
{
  "book_chapters": [
    {
      /* individual chapter entries live here */
    }
  ]
}
```

Each chapter entry must include the fields described below. Maintain two-space indentation and a trailing comma after every entry except the last one, matching the existing file style.

### Field Reference

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `authors` | array of objects | ✅ | Ordered contributor list. Each object requires `firstName` (initials or given name) and `lastName`. |
| `title` | string | ✅ | Chapter title. |
| `editors` | array of objects | ✅ | Same structure as `authors`. Use `[]` if truly no editor is credited. |
| `book_title` | string | ✅ | Title of the host book/volume. |
| `series` | string\|null | ⚠️ | Name of the publisher series (e.g., "Springer Water"). |
| `volume` | int\|string\|null | ⚠️ | Volume number/label within the series. Use `null` when absent. |
| `pages` | object | ✅ | `{ "start": <int>, "end": <int> }`. Use absolute page numbers. |
| `publisher` | object | ✅ | `{ "name": "Springer", "location": "Cham" }`. City/state/country go into `location`. |
| `year` | int | ✅ | Four-digit publication year. |
| `url` | string\|null | ⚠️ | Optional open-access, repository, or preprint URL. |
| `doi` | string\|null | ⚠️ | Digital Object Identifier in canonical form (lowercase `10.` prefix). |
| `chapter_number` | int\|string\|null | ⚠️ | Use when the chapter is numbered; otherwise `null`. |
| `abstract` | string\|null | ⚠️ | Full abstract text. Keep newline-free unless multiple paragraphs are essential. |
| `keywords` | array of strings | ⚠️ | Ordered list of topical keywords. Use `[]` when no keywords exist. |
| `publisher_url` | string\|null | ⚠️ | Official publisher landing page for the chapter. |
| `isbn` | object\|string\|null | ⚠️ | Prefer `{ "print": "...", "electronic": "..." }`. If only one ISBN exists, store it as a string. |
| `keyword_source` | string\|null | ⚠️ | Keep track of the keyword provenance (e.g., `"publisher"`, `"generated"`, `"manual"`). |

## Using the Template

1. Copy `templates/book_chapter_template.jsonc` somewhere convenient.
2. Fill in every placeholder. Keep `null` for unknown values instead of deleting fields.
3. Remove the lines starting with `//` (they are comments for the template only and make the JSON invalid).
4. Paste the finished object (including outer braces) into the `book_chapters` array. Keep entries sorted by `year` descending; if several share the same year, place the newest publication (by release date) first.
5. Confirm that commas and indentation match the surrounding entries.

## Validating Your Changes

Run at least one of the following before committing:

```bash
python3 -m json.tool output_data/bkchpts.json > /dev/null
# or
jq . output_data/bkchpts.json > /dev/null
```

If either command errors, fix the reported line. Because the entire book chapter pipeline assumes strict JSON, even tiny formatting mistakes will break `src/chpts2json.py` consumers.

## Optional: Regenerating via Parser

If you prefer not to edit JSON manually, you can update the LaTeX source in `input_data/list_of_chpts.tex` and re-run the parser:

```bash
python3 src/chpts2json.py -i input_data/list_of_chpts.tex -o output_data/bkchpts.json --force
```

This will rebuild the JSON from scratch using the latest automation (keyword generation, metadata enrichment, caches, etc.). Review the diff carefully before committing to ensure no unintended changes slip in.

Once the JSON validates and the new entry appears in the correct place, your book chapter addition is complete.
