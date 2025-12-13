"""
Progressive backfill script to retrieve historical job data in 2-day windows.

This script is designed to run hourly and fetch one 2-day window at a time,
working backwards from 30 days ago to the present. It tracks progress using
a state file to avoid re-scraping the same time window.

Usage (PowerShell):
  python scripts/backfill_historical.py --search-terms "Data Scientist,Machine learning engineer" --locations "United States,Remote" --results-per-site 400

Environment variables (via .env):
  S3_BUCKET_NAME          - required
  S3_RAW_KEY              - optional (default: raw_jobs.csv)
  S3_PROCESSED_KEY        - optional (default: processed_jobs.csv)
  AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION - required for S3

State tracking:
- Progress is saved in data/backfill_state.json
- Each run processes one 2-day window
- Automatically stops when all 15 windows (30 days) are complete
"""
from __future__ import annotations

import argparse
import os
import sys
import json
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from scraper import JobScraper
from data_processor import DataProcessor
import pandas as pd
import boto3
from botocore.exceptions import ClientError, BotoCoreError
from typing import Optional
import re
from urllib.parse import urlsplit, urlunsplit, unquote


# State file to track progress
STATE_FILE = os.path.join("data", "backfill_state.json")


def _first_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _norm_text(val) -> Optional[str]:
    if pd.isna(val):
        return None
    s = str(val).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s or None


def _norm_url(val) -> Optional[str]:
    if pd.isna(val):
        return None
    u = str(val).strip()
    try:
        parts = urlsplit(u)
        scheme = parts.scheme or "https"
        netloc = parts.netloc.lower()
        path = unquote(parts.path or "/")
        if path.endswith("/") and len(path) > 1:
            path = path.rstrip("/")
        return urlunsplit((scheme, netloc, path, "", ""))
    except Exception:
        return u.lower()


def _add_norm_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    title_col = _first_col(df, ["title", "job_title"])
    company_col = _first_col(df, ["company", "company_name"])
    location_col = _first_col(df, ["location"])
    link_col = _first_col(df, ["link", "job_url", "job_url_direct"])

    df["norm_title"] = df[title_col].map(_norm_text) if title_col else None
    df["norm_company"] = df[company_col].map(_norm_text) if company_col else None
    df["norm_location"] = df[location_col].map(_norm_text) if location_col else None
    df["norm_link"] = df[link_col].map(_norm_url) if link_col else None
    return df


def load_state() -> dict:
    """Load backfill state from JSON file."""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠ Failed to load state file: {e}. Starting fresh.")
    return {"completed_windows": [], "current_window": 0, "total_windows": 15}


def save_state(state: dict) -> None:
    """Save backfill state to JSON file."""
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)


def get_next_window(state: dict, results_per_site: int) -> Optional[dict]:
    """
    Calculate the next pagination window to scrape.
    Returns None if backfill is complete.
    
    Uses pagination (offset) to go back in time, assuming results are date-descending.
    - Window 0: Offset 0 (Newest jobs)
    - Window 1: Offset N (Slightly older)
    - ...
    """
    current = state["current_window"]
    total = state["total_windows"]
    
    if current >= total:
        return None
    
    # Calculate offset
    offset = current * results_per_site
    
    return {
        "window_num": current,
        "hours_old": 720,  # Always look at last 30 days
        "offset": offset,
        "description": f"Pagination Offset {offset} (Jobs {offset}-{offset+results_per_site})"
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Progressive backfill of historical job data (2-day windows)")
    p.add_argument("--search-terms", default=os.getenv("SEARCH_TERMS", "Data Scientist,Senior Data Scientist,Machine Learning Engineer,ML Engineer,AI Engineer,Artificial Intelligence Engineer,Data Analyst,Business Data Analyst,Data Engineer,Big Data Engineer,Analytics Engineer,Research Scientist,Applied Scientist"), help="Comma-separated search terms")
    p.add_argument("--locations", default=os.getenv("LOCATIONS", "United States,Remote"), help="Comma-separated locations")
    p.add_argument("--results-per-site", type=int, default=int(os.getenv("RESULTS_PER_SITE", "50")), help="Max results per site per window")
    p.add_argument("--reset", action="store_true", help="Reset backfill state and start over")
    return p.parse_args()


def main() -> int:
    load_dotenv()
    
    args = parse_args()
    search_terms = [s.strip() for s in args.search_terms.split(",") if s.strip()]
    locations = [s.strip() for s in args.locations.split(",") if s.strip()]
    results_per_site = args.results_per_site

    bucket = os.getenv("S3_BUCKET_NAME")
    if not bucket:
        print("❌ S3_BUCKET_NAME not set in environment. Aborting.")
        return 1

    raw_key_default = os.getenv("S3_RAW_KEY", "raw_jobs.csv")
    processed_key_default = os.getenv("S3_PROCESSED_KEY", "processed_jobs.csv")

    # Local paths
    raw_local = os.path.join("data", "raw_jobs.csv")
    processed_local = os.path.join("data", "processed_jobs.csv")

    # Load or reset state
    if args.reset:
        state = {"completed_windows": [], "current_window": 0, "total_windows": 15}
        save_state(state)
        print("🔄 Reset backfill state")
    else:
        state = load_state()
        # Ensure state file exists on disk so artifact upload doesn't fail on first run/error
        if not os.path.exists(STATE_FILE):
            save_state(state)
            print("ℹ Initialized backfill state file")

    # Check if backfill is complete
    window_info = get_next_window(state, results_per_site)
    if window_info is None:
        print("✅ Backfill complete! All 15 windows (30 days) have been processed.")
        return 0

    print(f"\n{'='*60}")
    print(f"📊 Backfill Progress: Window {window_info['window_num'] + 1}/15")
    print(f"📅 Time Range: {window_info['description']}")
    print(f"{'='*60}\n")

    # Prepare S3 client
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION"),
    )

    # Load existing data from S3
    existing_raw: Optional[pd.DataFrame] = None
    try:
        obj = s3.get_object(Bucket=bucket, Key=raw_key_default)
        existing_raw = pd.read_csv(obj["Body"])  # type: ignore[arg-type]
        print(f"✓ Loaded existing raw from S3: {len(existing_raw)} rows")
    except (ClientError, BotoCoreError, Exception) as e:
        print(f"⚠ Could not read from S3: {e}")
        if os.path.exists(raw_local):
            try:
                existing_raw = pd.read_csv(raw_local)
                print(f"✓ Loaded from local file: {len(existing_raw)} rows")
            except Exception:
                existing_raw = None
        else:
            existing_raw = None
            print("ℹ No existing data found; starting fresh")

    # Scrape this window
    print(f"🔍 Scraping window {window_info['window_num']}: {window_info['description']}")
    print(f"   Terms: {search_terms}")
    print(f"   Locations: {locations}")
    print(f"   Results/site: {results_per_site}")

    time_windows = [{
        "hours_old": window_info["hours_old"],
        "offset": window_info["offset"],
        "results_wanted": results_per_site
    }]

    scraper = JobScraper(
        search_terms=search_terms,
        locations=locations,
        results_per_site=results_per_site,
        time_windows=time_windows
    )
    
    try:
        new_raw = scraper.scrape_jobs()
        print(f"✓ Scraped {len(new_raw)} jobs from this window")
    except Exception as e:
        print(f"❌ Scraping failed: {e}")
        return 1

    if new_raw.empty:
        print("⚠ No jobs found in this window, but marking as complete")
        # Still mark as complete to move forward
        state["completed_windows"].append(window_info["window_num"])
        state["current_window"] += 1
        save_state(state)
        return 0

    # Normalize and merge
    new_raw_norm = _add_norm_columns(new_raw)
    
    if existing_raw is not None and not existing_raw.empty:
        existing_raw_norm = _add_norm_columns(existing_raw)
        merged_raw = pd.concat([existing_raw_norm, new_raw_norm], ignore_index=True)
        total_before = len(merged_raw)

        # Deduplicate by ID
        removed_by_id = 0
        if "id" in merged_raw.columns and merged_raw["id"].notna().any():
            tmp_before = len(merged_raw)
            merged_raw = merged_raw.drop_duplicates(subset=["id"], keep="first")
            removed_by_id = tmp_before - len(merged_raw)

        # Deduplicate by normalized composite keys
        subset_cols = [c for c in ["norm_title", "norm_company", "norm_location", "norm_link"] 
                      if c in merged_raw.columns]
        removed_by_composite = 0
        if subset_cols:
            tmp_before = len(merged_raw)
            merged_raw = merged_raw.drop_duplicates(subset=subset_cols, keep="first")
            removed_by_composite = tmp_before - len(merged_raw)

        print(f"📝 Deduplication: removed {removed_by_id} by ID, {removed_by_composite} by composite key")
        print(f"   Total: {len(merged_raw)} rows (added {len(merged_raw) - len(existing_raw_norm)} net new)")
    else:
        merged_raw = new_raw_norm.copy()
        print(f"📝 First dataset: {len(merged_raw)} rows")

    # Save merged data locally (for processing only, not uploaded to S3)
    os.makedirs(os.path.dirname(raw_local), exist_ok=True)
    norm_cols = [c for c in merged_raw.columns if str(c).startswith("norm_")]
    merged_to_save = merged_raw.drop(columns=norm_cols) if norm_cols else merged_raw
    merged_to_save.to_csv(raw_local, index=False)
    print(f"✓ Saved raw data locally: {raw_local}")

    # Process data
    print("⚙️  Processing data...")
    processor = DataProcessor(raw_local)
    processor.process_all()
    processor.save_processed_data(processed_local)
    
    # Upload only processed data to S3 (raw data not needed in S3)
    try:
        processor.upload_to_s3(processed_local, bucket, processed_key_default)
        print(f"✓ Uploaded processed data to S3: s3://{bucket}/{processed_key_default}")
    except Exception as e:
        print(f"❌ Processed data upload failed: {e}")
        return 1

    # Update state
    state["completed_windows"].append(window_info["window_num"])
    state["current_window"] += 1
    save_state(state)
    
    remaining = state["total_windows"] - state["current_window"]
    print(f"\n✅ Window {window_info['window_num'] + 1}/15 complete!")
    print(f"📊 Progress: {len(state['completed_windows'])}/{state['total_windows']} windows done")
    if remaining > 0:
        print(f"⏳ Remaining: {remaining} windows (~{remaining} hours at 1 window/hour)")
    else:
        print("🎉 Backfill complete!")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
