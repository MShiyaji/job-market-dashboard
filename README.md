# Job Market Dashboard

A minimal Streamlit app and utilities to scrape job listings, process the data, and (optionally) match them to your resume.

## Structure
- `app.py` – Streamlit UI
- `scraper.py` – Scrape jobs to `data/raw_jobs.csv`
- `data_processor.py` – Clean to `data/processed_jobs.csv`
- `resume_analyzer.py` – Stub for resume matching
- `utils/config.py` – Env-based settings
- `data/` – CSVs and resume file

View the full app here: https://data-science-jobs.streamlit.app/
