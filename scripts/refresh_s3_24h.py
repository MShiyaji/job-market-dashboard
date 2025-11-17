"""
Repopulate the S3 bucket with newly scraped and processed jobs from the last 24 hours.

Usage (PowerShell):
  python scripts/refresh_s3_24h.py --search-terms "Data Scientist,Machine learning engineer,AI engineer,Data Analyst,Data Engineer" --locations "United States,Remote" --results-per-site 400 --s3-prefix "" --timestamp

Environment variables (via .env):
  S3_BUCKET_NAME          - required
  S3_RAW_KEY              - optional (default: raw_jobs.csv)
  S3_PROCESSED_KEY        - optional (default: processed_jobs.csv)
  AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION - required for S3

Notes:
- Optimized for recent job analysis: scrapes only the last 24 hours for the most current job postings
- If --timestamp is provided, files will be uploaded with a datetime suffix under optional --s3-prefix.
- Local CSVs will still be written to data/ for debugging.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

from dotenv import load_dotenv

# Ensure project root is on sys.path when running as a script (python scripts/refresh_s3_24h.py)
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
        # drop query & fragment; normalize trailing slash (except root)
        if path.endswith("/") and len(path) > 1:
            path = path.rstrip("/")
        return urlunsplit((scheme, netloc, path, "", ""))
    except Exception:
        return u.lower()


def _add_norm_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    title_col = _first_col(df, ["title", "job_title"])
    company_col = _first_col(df, ["company", "company_name"])
    location_col = _first_col(df, ["location"])  # jobspy typically uses 'location'
    link_col = _first_col(df, ["link", "job_url", "job_url_direct"])  # prefer explicit link, fallback to job urls

    df["norm_title"] = df[title_col].map(_norm_text) if title_col else None
    df["norm_company"] = df[company_col].map(_norm_text) if company_col else None
    df["norm_location"] = df[location_col].map(_norm_text) if location_col else None
    df["norm_link"] = df[link_col].map(_norm_url) if link_col else None
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Repopulate S3 with scraped/processed job data from last 24 hours")
    p.add_argument("--search-terms", default=os.getenv("SEARCH_TERMS", "Data Scientist,Machine learning engineer,AI engineer,Data Analyst,Data Engineer"), help="Comma-separated search terms")
    p.add_argument("--locations", default=os.getenv("LOCATIONS", "United States,Remote"), help="Comma-separated locations")
    p.add_argument("--results-per-site", type=int, default=int(os.getenv("RESULTS_PER_SITE", "400")), help="Max results per site (default: 400 for recent job analysis)")
    p.add_argument("--s3-prefix", default=os.getenv("S3_PREFIX", ""), help="Optional key prefix in S3 (e.g., 'jobs/')")
    p.add_argument("--timestamp", action="store_true", help="Append timestamp to S3 object keys")
    return p.parse_args()


def main() -> int:
    load_dotenv()

    args = parse_args()
    search_terms = [s.strip() for s in args.search_terms.split(",") if s.strip()]
    locations = [s.strip() for s in args.locations.split(",") if s.strip()]
    results_per_site = args.results_per_site

    bucket = os.getenv("S3_BUCKET_NAME")
    if not bucket:
        print("S3_BUCKET_NAME not set in environment. Aborting.")
        return 1

    raw_key_default = os.getenv("S3_RAW_KEY", "raw_jobs.csv")
    processed_key_default = os.getenv("S3_PROCESSED_KEY", "processed_jobs.csv")

    # Local paths
    raw_local = os.path.join("data", "raw_jobs.csv")
    processed_local = os.path.join("data", "processed_jobs.csv")

    # Prepare S3 client
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION"),
    )

    # Try to load existing raw from S3 (or local fallback) for incremental merge
    existing_raw: Optional[pd.DataFrame] = None
    try:
        obj = s3.get_object(Bucket=bucket, Key=raw_key_default)
        existing_raw = pd.read_csv(obj["Body"])  # type: ignore[arg-type]
        print(f"✓ Loaded existing raw from s3://{bucket}/{raw_key_default}: {len(existing_raw)} rows")
    except (ClientError, BotoCoreError, Exception) as e:
        print(f"⚠ Failed to read from S3 (s3://{bucket}/{raw_key_default}): {e}")
        if os.path.exists(raw_local):
            try:
                existing_raw = pd.read_csv(raw_local)
                print(f"✓ Falling back to local file: {raw_local} ({len(existing_raw)} rows)")
            except Exception as local_e:
                existing_raw = None
                print(f"⚠ Local file read failed: {local_e}")
        else:
            existing_raw = None
            print(f"ℹ No local file found at {raw_local}; starting fresh with new scrape")

    # Scrape fresh data from last 24 hours
    print(f"Scraping last 24 hours with terms={search_terms}, locations={locations}, results/site={results_per_site}")

    # Configure time windows for last 24 hours only
    time_windows_24h = [
        {"hours_old": 48, "hours_new": 0, "results_wanted": results_per_site}
    ]

    scraper = JobScraper(search_terms=search_terms, locations=locations, results_per_site=results_per_site, time_windows=time_windows_24h)
    new_raw = scraper.scrape_jobs()

    # Normalize for robust de-duplication
    new_raw_norm = _add_norm_columns(new_raw)
    if existing_raw is not None and not existing_raw.empty:
        existing_raw_norm = _add_norm_columns(existing_raw)
    else:
        existing_raw_norm = None

    # Merge then de-duplicate using multiple strategies
    if existing_raw_norm is None or existing_raw_norm.empty:
        merged_raw = new_raw_norm.copy()
        removed_by_id = 0
        removed_by_composite = 0
        added_count = len(merged_raw)
    else:
        merged_raw = pd.concat([existing_raw_norm, new_raw_norm], ignore_index=True)
        total_before = len(merged_raw)

        # 1) Dedupe by id when available in merged set
        removed_by_id = 0
        if "id" in merged_raw.columns and merged_raw["id"].notna().any():
            tmp_before = len(merged_raw)
            merged_raw = merged_raw.drop_duplicates(subset=["id"], keep="first")
            removed_by_id = tmp_before - len(merged_raw)

        # 2) Dedupe by normalized composite keys
        subset_cols = [c for c in ["norm_title", "norm_company", "norm_location", "norm_link"] if c in merged_raw.columns]
        removed_by_composite = 0
        if subset_cols:
            tmp_before = len(merged_raw)
            merged_raw = merged_raw.drop_duplicates(subset=subset_cols, keep="first")
            removed_by_composite = tmp_before - len(merged_raw)

        total_after = len(merged_raw)
        print(
            f"De-duplication summary -> by id: {removed_by_id}, by normalized composite: {removed_by_composite}, "
            f"kept: {total_after}/{total_before}"
        )

        # Added count relative to previous dataset size
        added_count = max(0, len(merged_raw) - len(existing_raw_norm))

    print(f"Added {max(0, added_count)} new rows. Total raw rows: {len(merged_raw)}")

    # Save merged raw locally (for processing only, not uploaded to S3)
    os.makedirs(os.path.dirname(raw_local), exist_ok=True)
    # Drop normalization helper columns before persisting
    norm_cols = [c for c in merged_raw.columns if str(c).startswith("norm_")]
    merged_to_save = merged_raw.drop(columns=norm_cols) if norm_cols else merged_raw
    merged_to_save.to_csv(raw_local, index=False)
    print(f"Saved raw data locally: {raw_local} (not uploaded to S3)")

    # Process
    processor = DataProcessor(raw_local)
    processor.process_all()
    processor.save_processed_data(processed_local)
    
    # Upload only processed data to S3 (raw data not needed in S3)
    try:
        processor.upload_to_s3(processed_local, bucket, processed_key_default)
        print(f"Uploaded processed data to S3: s3://{bucket}/{processed_key_default}")
    except Exception as e:
        print(f"Processed data upload failed: {e}")

    # Optionally upload processed with custom prefix/timestamp
    if args.timestamp or args.s3_prefix:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S") if args.timestamp else ""
        basename = f"processed_jobs_{ts}.csv" if ts else os.path.basename(processed_key_default)
        object_name = f"{args.s3_prefix}{basename}" if args.s3_prefix else basename
        try:
            processor.upload_to_s3(processed_local, bucket, object_name)
        except Exception as e:
            print(f"Optional timestamped processed upload failed: {e}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())