# Adding Collaborators to `collaborators.json`

> Author: Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science and Technology, Abu Dhabi, UAE)

Use this guide when manually appending new collaborators to `output_data/collaborators.json`. Keeping the structure aligned with what `src/collab2json.py` emits ensures that geocoding, verification, and downstream analytics continue to function without special handling.

## Workflow Overview
- **1. Collect metadata.** Gather everything in the checklist below before editing the JSON file.
- **2. Fill the template.** Copy `templates/collaborator_template.jsonc`, replace the placeholders with real data, and remove the inline comments.
- **3. Append the entry.** Insert the completed object into the JSON array. Keep `id` values sequential and preserve the two-space indentation.
- **4. Validate JSON.** Run `python3 -m json.tool output_data/collaborators.json >/dev/null` (or `jq . output_data/collaborators.json >/dev/null`) to confirm valid formatting.
- **5. Optional refresh.** If you want `_entry_hash`, geocoding, or cached metadata regenerated, run `python3 src/collab2json.py <your_source>.tex --verify --output output_data/collaborators.json --force` against the authoritative LaTeX source.

## Metadata Checklist

| Data Point | Required | Notes |
| --- | --- | --- |
| Legal name (first, last, display) | ✅ | Use official spelling; `fullName` is typically `"First Last"`. |
| Affiliation & department | ✅ | Include both a short `affiliation` and the associated `university`. |
| Institution type / research group | ⚠️ | `institutionType` can be `university`, `institute`, `company`, etc. |
| City / state / country | ✅ | Also record a formatted `locationString`. |
| Coordinates | ⚠️ | `coordinates.lat` / `coordinates.lng` in decimal degrees. Use `null` if unknown and rerun geocoder later. |
| Contact info | ⚠️ | Email and homepage URL when available. |
| Academic title | ⚠️ | Professor, Postdoctoral, PhD Candidate, etc. |
| Profiles | ⚠️ | ORCID, Google Scholar ID, ResearchGate, LinkedIn, GitHub, Twitter, Scopus, Loop. Use `null` for unknown handles. |
| Academic metrics | ⚠️ | Populate `h_index`, `citations`, `i10_index`, `publications_count`, `source`, `scholar_id` when available. |
| Research interests | ⚠️ | Array of keywords summarizing expertise. |
| Search confidence | ⚠️ | Float between 0 and 1 expressing trust in the aggregated data. |
| Nested entries (`collaboratorEntries`) | ⚠️ | Use when multiple affiliations/locations exist. Include name, affiliation, location, enhanced location data, and context. |
| Hash & flags | ⚠️ | `_entry_hash` (set to `null` if unknown), `llm_enhanced` boolean. |

> ⚠️ **Tip:** Keep every field present, setting values to `null` (or an empty list/object) when you do not have data. Downstream consumers depend on a predictable schema.

## JSON Structure

`output_data/collaborators.json` is an array:

```json
[
  {
    /* collaborator entry */
  },
  {
    /* next collaborator */
  }
]
```

Maintain two-space indentation, trailing commas between objects, and sequential integer `id` values.

## Field Reference

### Identity & Affiliation

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `id` | int | ✅ | Sequential identifier unique across the array. |
| `firstName`, `lastName`, `fullName` | string | ✅ | Primary name fields. |
| `nameVariants` | array of strings \| null | ⚠️ | Alternate spellings (initials, maiden names, etc.). |
| `affiliation` | string | ✅ | Short description, e.g., `"Department of Mathematics"`. |
| `university` | string | ✅ | Host institution or organization. |
| `department`, `researchGroup` | string \| null | ⚠️ | Additional sub-units. |
| `institutionType` | string \| null | ⚠️ | e.g., `"university"`, `"institute"`, `"company"`. |
| `academicTitle` | string \| null | ⚠️ | Rank or role such as `"Professor"`. |
| `specialNotes` | string \| null | ⚠️ | Free-form clarifications. |

### Location & Geodata

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `city`, `state`, `country` | string \| null | ✅ | Geographic fields; `state` optional. |
| `locationString` | string | ✅ | Preformatted `"City, Country"` label. |
| `coordinates` | object | ⚠️ | `{ "lat": <float>, "lng": <float> }`. |
| `alternateLocations` | string \| array \| null | ⚠️ | Additional notes about former locations. |
| `llm_enhanced` | bool | ⚠️ | Indicates whether LLM augmentation was applied. |
| `_entry_hash` | string \| null | ⚠️ | Integrity hash produced by `collab2json`. Leave `null` if not known and regenerate later. |

### Contact & Profiles

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `email`, `homepage` | string \| null | ⚠️ | Contact channels. |
| `profiles` | object | ⚠️ | Keys: `orcid`, `google_scholar`, `researchgate`, `linkedin`, `github`, `twitter`, `scopus`, `loop`. Values are strings or `null`. |

### Academic Metrics & Interests

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `academicMetrics` | object | ⚠️ | Contains `h_index`, `citations`, `i10_index`, `publications_count`, `source`, `scholar_id`. Values default to `null` when unknown. |
| `researchInterests` | array of strings | ⚠️ | Descriptive keywords. |
| `searchConfidence` | float \| null | ⚠️ | Confidence level (0–1) assigned by your enrichment pipeline. |

### Nested `collaboratorEntries`

Use `collaboratorEntries` (array) to record multiple affiliations/locations per collaborator. Each entry object supports the following structure:

| Field | Description |
| --- | --- |
| `nameInformation` | `{ "firstName", "lastName", "middleName?", "nameVariants": [] }`. |
| `affiliationDetails` | `{ "affiliation", "university", "institutionType", "department", "researchGroup" }`. |
| `locationInformation` | `{ "city", "state?", "country", "alternateLocations" }`. |
| `enhancedLocationData` | `{ "improvedLocationString", "locationConfidenceScore", "ambiguousLocation", "ambiguityNotes?", "suggestedLocationString?" }`. |
| `additionalContext` | `{ "isFormerAffiliation", "academicTitle", "specialNotes" }`. |

If you only track a single affiliation, keeping a one-element array still preserves schema consistency.

## Using the Template

1. Copy `templates/collaborator_template.jsonc`.
2. Replace placeholders with real data. Keep `null`/empty placeholders for unknown values.
3. Remove all `//` lines so the snippet becomes valid JSON.
4. Insert the object into `output_data/collaborators.json`. Update `id` to the next available integer.
5. Validate the file with `python3 -m json.tool output_data/collaborators.json`.

## Optional: Rebuild via Parser

When possible, prefer editing the source LaTeX list and regenerating JSON:

```bash
python3 src/collab2json.py input_data/collab_excerpts.tex --output output_data/collaborators.json --force -v 2
```

This route guarantees that hashes, geocoding, search caches, and profile verifications stay in sync.
