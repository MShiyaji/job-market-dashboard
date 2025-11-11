# Job Market Dashboard

A minimal Streamlit app and utilities to scrape job listings, process the data, and (optionally) match them to your resume.

## Structure
- `app.py` – Streamlit UI
- `scraper.py` – Scrape jobs to `data/raw_jobs.csv`
- `data_processor.py` – Clean to `data/processed_jobs.csv`
- `resume_analyzer.py` – Stub for resume matching
- `utils/config.py` – Env-based settings
- `data/` – CSVs and resume file

## Quickstart
1. Create a virtual environment and install requirements.
2. Put your resume PDF at `data/resume.pdf` (optional for now).
3. Generate demo data and run the app.

### Install
```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Demo data and processing
```powershell
python scraper.py
python data_processor.py
```

### Run app
```powershell
streamlit run app.py
```

## Environment
Copy `.env.example` to `.env` and set variables.

## Notes
- This is a scaffold: replace demo parts with real scraping and analysis.
- `.env` is intentionally git-ignored. Never commit secrets.
