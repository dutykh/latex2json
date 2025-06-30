# LaTeX to JSON Converter

This project provides a Python script to parse a list of collaborators from a LaTeX (`.tex`) file, geocode their affiliations to find geographic coordinates, and export the structured data into a JSON file.

## Features

- Extracts names and affiliations from a LaTeX `description` environment.
- Cleans LaTeX-specific commands and characters (e.g., `\textsc`, `\'{e}`).
- Geocodes locations using the OpenStreetMap Nominatim API.
- Caches geocoding results to avoid redundant API calls and speed up subsequent runs.
- Generates a clean, well-structured JSON file.

## Prerequisites

- Python 3.6+
- `requests` library

To install the required library, run:
```bash
pip install requests
```

## Usage

1.  Place your input `.tex` file inside the `input_data/` directory.
2.  Run the script from the command line, providing the name of your input file as an argument.

```bash
python3 latex_to_json.py <your_input_file.tex>
```

### Example

To process the sample file `collab_excerpts.tex`:

```bash
python3 latex_to_json.py collab_excerpts.tex
```

The script will generate a corresponding JSON file in the `output_data/` directory (e.g., `output_data/collab_excerpts.json`).

### Optional Arguments

- `--skip-geocoding`: Use this flag to disable the geocoding process. This is useful for quickly parsing the LaTeX file without making external API calls.

```bash
python3 latex_to_json.py collab_excerpts.tex --skip-geocoding
```

## File Structure

- `latex_to_json.py`: The main Python script.
- `input_data/`: Directory for your input `.tex` files.
- `output_data/`: Directory where the output `.json` files are saved.
- `.gitignore`: Specifies files and directories to be ignored by Git.
- `LICENSE`: The license file for the project.
- `README.md`: This documentation file.

## Author

- **Dr. Denys Dutykh** - *Initial work* - [Khalifa University of Science and Technology, Abu Dhabi, UAE]

## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for more details.
