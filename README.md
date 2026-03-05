# Plant-Based TPA Analyzer (TUI)
A lightweight Terminal User Interface for analyzing Double Compression (TPA) data from Zwick ZS10 machines.

## Setup Instructions
1. Install requirements: `pip install -r requirements.txt`
2. Run the application: `python app.py`

## Usage
- The app will automatically try to parse `.csv` files in the current directory matching the Zwick export format.
- Click or use keyboard navigation in the terminal to interact with the interface.

## Build macOS + Windows executables (GitHub Actions)
This repository includes a workflow at `.github/workflows/build-binaries.yml` that creates native one-file binaries:
- `tpa-analyzer` on macOS
- `tpa-analyzer.exe` on Windows

How to run:
1. Push this repository to GitHub.
2. Open **Actions** -> **Build Binaries**.
3. Click **Run workflow** (or push a tag like `v1.0.0`).
4. Download artifacts:
   - `tpa-analyzer-macOS`
   - `tpa-analyzer-Windows`
