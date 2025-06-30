# -*- coding: utf-8 -*-
"""
This script extracts collaborator information from a LaTeX file, geocodes their locations, and saves the data as a JSON file.

Author: Dr. Denys Dutykh (Khalifa University of Science and Technology, Abu Dhabi, UAE)
Date: 2025-06-30
"""

import re
import json
import time
import argparse
import requests
from pathlib import Path
from typing import List, Dict, Optional

# --- Configuration ---

# Base directory of the project
BASE_DIR = Path(__file__).parent.resolve()

# Directories for input and output data
INPUT_DIR = BASE_DIR / "input_data"
OUTPUT_DIR = BASE_DIR / "output_data"

# Geocoding cache file, stored in a temporary location
GEOCODE_CACHE_FILE = BASE_DIR / ".geocode_cache.json"

# Set to True to disable geocoding (e.g., for faster testing)
SKIP_GEOCODING = False

# --- Geocoding ---

def geocode_location(location: str, cache: Dict, skip: bool = SKIP_GEOCODING) -> Optional[Dict[str, float]]:
    if skip:
        print(f"[Geocode] Skipping geocoding for: {location}")
        return None
    
    cache_key = location.lower()
    if cache_key in cache:
        coords = cache[cache_key]
        if coords:
            print(f"[Geocode] Cache hit for '{location}': {coords}")
        else:
            print(f"[Geocode] Cache hit for failed location: '{location}'")
        return coords

    print(f"[Geocode] Cache miss. Querying API for: {location}")
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": location,
        "format": "json",
        "limit": 1
    }
    try:
        response = requests.get(url, params=params, headers={"User-Agent": "latex-to-json-script"})
        response.raise_for_status()
        data = response.json()
        if data:
            coords = {"lat": float(data[0]["lat"]), "lng": float(data[0]["lon"])}
            print(f"[Geocode] Success: {coords}")
            cache[cache_key] = coords
            return coords
        else:
            print("[Geocode] No results found.")
            cache[cache_key] = None
    except Exception as e:
        print(f"[Geocode] Error for '{location}': {e}")
        cache[cache_key] = None
    return None


def clean_latex_string(text: str) -> str:
    """Replaces common LaTeX commands with their Unicode equivalents."""
    # Define replacements for commands that take arguments, e.g., \v{s}
    # This will handle them even if they are inside another set of braces.
    text = re.sub(r"\\textsc\{(.*?)\}", r"\1", text)
    text = re.sub(r"\\v\{s\}", "š", text)
    text = re.sub(r"\\v\{S\}", "Š", text)
    text = re.sub(r"\\v\{c\}", "č", text)
    text = re.sub(r"\\v\{C\}", "Č", text)
    text = re.sub(r"\\v\{z\}", "ž", text)
    text = re.sub(r"\\v\{Z\}", "Ž", text)
    text = re.sub(r"\\c\{c\}", "ç", text)
    text = re.sub(r"\\c\{C\}", "Ç", text)

    # Define replacements for simple accent commands, e.g., \'a
    replacements = {
        "\\'e": "é", "\\`e": "è", "\\^e": "ê", '\\"e': "ë",
        "\\'a": "á", "\\`a": "à", "\\^a": "â", '\\"a': "ä",
        "\\'i": "í", "\\`i": "ì", "\\^i": "î", '\\"i': "ï",
        "\\'o": "ó", "\\`o": "ò", "\\^o": "ô", '\\"o': "ö",
        "\\'u": "ú", "\\`u": "ù", "\\^u": "û", '\\"u': "ü",
        "\\'E": "É", "\\`E": "È", "\\^E": "Ê", '\\"E': "Ë",
        "\\'A": "Á", "\\`A": "À", "\\^A": "Â", '\\"A': "Ä",
        "\\'I": "Í", "\\`I": "Ì", "\\^I": "Î", '\\"I': "Ï",
        "\\'O": "Ó", "\\`O": "Ò", "\\^O": "Ô", '\\"O': "Ö",
        "\\'U": "Ú", "\\`U": "Ù", "\\^U": "Û", '\\"U': "Ü",
        "\\o": "ø", "\\O": "Ø",
        "\\&": "&",
        "--": "-",
        "~": " ",
    }

    # Apply the simple replacements, also handling the case where they are in braces, e.g., {\'a}
    for latex, unicode_char in replacements.items():
        text = text.replace(latex, unicode_char)
        text = text.replace("{" + latex + "}", unicode_char)

    # Clean up any remaining braces that might be left
    text = text.replace("{", "").replace("}", "")

    return text

# Improved parser for lines like:
#   \item[First \textsc{Last}:] Affiliation, University, City, Country

def parse_latex(tex_content: str, geocode_cache: Dict, skip_geocoding: bool):
    print("\n--- Extracting Collaborators ---")
    pattern = re.compile(r"^\s*\\item\[(.*?):\]\s*(.*)$", re.MULTILINE)
    results = []
    failed_geocode = []
    for idx, match in enumerate(pattern.finditer(tex_content), 1):
        full_name_raw, rest = match.groups()

        # Clean the full name and split it into first and last names
        full_name_clean = re.sub(r'\\textsc\{(.*?)\}', r'\1', full_name_raw)
        full_name_clean = clean_latex_string(full_name_clean).strip()
        
        name_parts = full_name_clean.split()
        first, last = "", ""
        if len(name_parts) > 1:
            first = " ".join(name_parts[:-1])
            last = name_parts[-1]
        else:
            last = full_name_clean

        print(f"\n--- Collaborator #{idx} ---")
        print(f"  Name: {full_name_clean}")

        # Clean LaTeX commands from the string
        rest = clean_latex_string(rest)

        # Remove '(formerly at)' and similar phrases
        rest = re.sub(r"\(formerly at\) ?", "", rest, flags=re.IGNORECASE).strip()
        country, city, university, affiliation = "", "", "", ""

        # A list of countries to look for. This helps avoid treating parts of an
        # affiliation as a country name.
        known_countries = [
            "Algeria", "Australia", "Austria", "Belgium", "Brazil", "Cameroon", "Canada", "Chile", "China", "Cyprus", "Czech Republic",
            "Denmark", "Estonia", "Finland", "France", "Germany", "Greece", "Hong Kong", "India", "Indonesia", "Iran",
            "Ireland", "Italy", "Japan", "Kazakhstan", "Lebanon", "Malaysia", "Morocco", "Netherlands",
            "New Zealand", "Norway", "Pakistan", "Poland", "Portugal", "Qatar", "Romania", "Russia",
            "Saudi Arabia", "Singapore", "Slovak Republic", "South Korea", "Spain", "Sweden", "Switzerland",
            "Taiwan", "Thailand", "Tunisia", "Turkey", "UAE", "UK", "Ukraine", "United Arab Emirates", "United Kingdom",
            "USA", "Vietnam"
        ]

        # Normalize country names to their long form
        country_map = {
            "UAE": "United Arab Emirates",
            "UK": "United Kingdom",
            "USA": "United States of America"
        }

        # Extract country using the known_countries list
        # Sort by length descending to match longer names first (e.g., "United Arab Emirates" before "UAE")
        for c in sorted(known_countries, key=len, reverse=True):
            # Use word boundaries to avoid matching substrings like 'UK' in 'Ukraine'
            if re.search(r'\b' + re.escape(c) + r'\b', rest, re.IGNORECASE):
                country = c
                rest = rest.replace(country, "").strip().rstrip(',')
                break

        country = country_map.get(country, country)

        # 2. Extract City
        parts = [p.strip() for p in rest.split(',') if p.strip()]
        if parts:
            # The last part is the most likely candidate for the city
            city_candidate = parts[-1]
            # A city name is unlikely to contain these keywords or be very long
            if "university" not in city_candidate.lower() and \
               "institute" not in city_candidate.lower() and \
               "center" not in city_candidate.lower() and \
               "laboratoire" not in city_candidate.lower() and \
               "R&D" not in city_candidate and \
               len(city_candidate.split()) < 4:
                city = city_candidate
                parts.pop(-1)

        # Special case for Košice, which might be part of the university name
        if not city and "košice" in full_details.lower():
            city = "Košice"

        # 3. Extract University and Affiliation from remaining parts
        remaining_details = ", ".join(parts)

        # Find the most specific affiliation part (often the first part)
        if parts:
            affiliation = parts[0]

        # Find a university name in the remaining parts
        for p in parts:
            if "university" in p.lower() or "universit" in p.lower() or "polytech" in p.lower() or "école" in p.lower():
                university = p
                break # Take the first one found

        # If affiliation is empty but university is not, use university as affiliation
        if not affiliation and university:
            affiliation = university
        elif not affiliation and remaining_details:
             affiliation = remaining_details

        # If university is not found, but affiliation is, use affiliation as university
        if not university and affiliation:
            university = affiliation

        print(f"  Affiliation: {affiliation}")
        print(f"  University: {university}")
        print(f"  City: {city}")
        print(f"  Country: {country}")

        # Improved location string logic
        location_str = ""
        if city and country:
            location_str = f"{city}, {country}"
        elif university and country:
            location_str = f"{university}, {country}"
        elif country:
            location_str = country
        else:
            location_str = ""
        reason = ""
        if not location_str or location_str == ",":
            reason = "Insufficient location data"
        
        coords = geocode_location(location_str, geocode_cache, skip=skip_geocoding) if location_str and not reason else None

        if coords is None:
            if not reason:
                reason = "Geocoding API failed"
            failed_geocode.append({
                "id": idx,
                "firstName": first,
                "lastName": last,
                "locationString": location_str,
                "reason": reason
            })
        results.append({
            "id": idx,
            "firstName": first,
            "lastName": last,
            "affiliation": affiliation,
            "university": university,
            "city": city,
            "country": country,
            "locationString": location_str,
            "coordinates": coords or {"lat": None, "lng": None},
            "collaborationYears": [],
            "publicationCount": None
        })
    return results, failed_geocode

def main():
    """Main function to parse arguments and run the script."""
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="Extract collaborator data from a LaTeX file and save it as JSON.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Name of the input .tex file located in the 'input_data' directory."
    )
    parser.add_argument(
        "--skip-geocoding",
        action="store_true",
        help="If set, skip the geocoding process to run the script faster."
    )
    args = parser.parse_args()

    # --- File Paths ---
    tex_file_path = INPUT_DIR / args.input_file
    output_json_path = OUTPUT_DIR / f"{tex_file_path.stem}.json"

    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(exist_ok=True)

    start_time = time.time()
    print("--- Starting Collaborator Processing ---")

    # Load geocoding cache
    geocode_cache = {}
    if GEOCODE_CACHE_FILE.exists():
        print(f"Loading geocode cache from: {GEOCODE_CACHE_FILE}")
        with open(GEOCODE_CACHE_FILE, "r", encoding="utf-8") as f:
            try:
                geocode_cache = json.load(f)
            except json.JSONDecodeError:
                print("Warning: Geocode cache is corrupted. Starting fresh.")
                geocode_cache = {}
    else:
        print("No geocode cache found. Starting fresh.")

    print(f"Reading LaTeX file: {tex_file_path}")
    if not tex_file_path.exists():
        print(f"Error: LaTeX file not found at {tex_file_path}")
        return

    if output_json_path.exists():
        print(f"Removing previous JSON file: {output_json_path}")
        output_json_path.unlink()

    tex_content = tex_file_path.read_text(encoding="utf-8")
    collaborators, failed_geocode = parse_latex(tex_content, geocode_cache, args.skip_geocoding)

    print(f"\n--- Writing Output ---")
    print(f"Writing {len(collaborators)} collaborators to {output_json_path}")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(collaborators, f, ensure_ascii=False, indent=2)
    
    # Save updated geocoding cache
    print(f"Saving updated geocode cache to: {GEOCODE_CACHE_FILE}")
    with open(GEOCODE_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(geocode_cache, f, ensure_ascii=False, indent=2)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print("\n--- Processing Summary ---")
    print(f"Total collaborators processed: {len(collaborators)}")
    print(f"Successfully geocoded: {len(collaborators) - len(failed_geocode)}")
    print(f"Failed to geocode: {len(failed_geocode)}")
    print(f"Total execution time: {elapsed_time:.2f} seconds")

    if failed_geocode:
        print("\n--- Geocoding Failures ---")
        for collaborator in failed_geocode:
            print(f"  - #{collaborator['id']} {collaborator['firstName']} {collaborator['lastName']}: {collaborator['locationString']} (Reason: {collaborator['reason']})")
    else:
        print("\nAll collaborators were successfully geocoded!")


if __name__ == "__main__":
    main()
