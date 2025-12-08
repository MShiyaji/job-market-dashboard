"""
Job scraping utility powered by jobspy (with a demo fallback).
- Scrapes multiple sites and writes a CSV to data/raw_jobs.csv
- Optimized for trend analysis: uses smaller samples for faster execution
"""
from __future__ import annotations

import boto3
import csv
import os
import time
from datetime import datetime
from typing import Iterable, List, Optional

import pandas as pd
from botocore.exceptions import NoCredentialsError
from dotenv import load_dotenv

load_dotenv()

try:  # pragma: no cover - optional dependency
    from jobspy import scrape_jobs as jobspy_scrape_jobs  # type: ignore
except Exception:  # pragma: no cover
    jobspy_scrape_jobs = None  # type: ignore


class JobScraper:
    """Scrape job postings from multiple job boards."""

    def __init__(self, search_terms: Iterable[str], locations: Iterable[str], results_per_site: int = 200, time_windows: Optional[List[dict]] = None) -> None:
        """Initialize scraper with search parameters.

        Args:
            search_terms: Job titles/queries to search for.
            locations: Locations to search.
            results_per_site: Results per board/site.
            time_windows: Optional list of time window dicts with 'hours_old', 'hours_new', 'results_wanted'.
                         If None, uses default 30-day balanced windows.
        """
        self.search_terms = list(search_terms)
        self.locations = list(locations)
        self.results_per_site = int(results_per_site)
        self.time_windows = time_windows
        self.all_jobs: pd.DataFrame = pd.DataFrame()

    def scrape_jobs(self, site_names: Optional[List[str]] = None) -> pd.DataFrame:
        """Scrape jobs from the specified sites and return a DataFrame.

        Notes on stability:
        - Glassdoor often requires a precise city/state; wide locations like
          'United States' or 'Remote' can return 400 or parse errors via JobSpy.
          We skip Glassdoor for such broad locations to avoid noisy failures.
        - For Remote searches, we pass a stable country context and set remote=True
          (when supported by JobSpy) to improve coverage and reduce country parsing errors.
        """
        if site_names is None:
            site_names = ["indeed", "linkedin", "glassdoor"]

        all_jobs_list: List[pd.DataFrame] = []

        for search_term in self.search_terms:
            for location in self.locations:
                print(f"Scraping {search_term} jobs in {location}…")

                if jobspy_scrape_jobs is None:
                    raise ImportError(
                        "jobspy is not installed. Install it with 'pip install jobspy' to scrape real data."
                    )

                aggregated: List[pd.DataFrame] = []

                for site in site_names:
                    if site.lower() == "glassdoor" and location.strip().lower() in {"united states", "remote"}:
                        print("Skipping Glassdoor for broad location; requires city/state-level location")
                        continue

                    if self.time_windows:
                        time_windows = self.time_windows
                    else:
                        time_windows = [
                            {"hours_old": 168, "hours_new": 0, "results_wanted": self.results_per_site // 4},      # 0-7 days ago (~100 results)
                            {"hours_old": 336, "hours_new": 168, "results_wanted": self.results_per_site // 4},   # 7-14 days ago (~100 results)
                            {"hours_old": 504, "hours_new": 336, "results_wanted": self.results_per_site // 4},   # 14-21 days ago (~100 results)
                            {"hours_old": 720, "hours_new": 504, "results_wanted": self.results_per_site // 4},   # 21-30 days ago (~100 results)
                        ]

                    for window in time_windows:
                        # Base kwargs for this time window
                        kwargs = dict(
                            site_name=[site],
                            search_term=search_term,
                            linkedin_fetch_description=True,
                        )
                        kwargs.update(window)

                        # Location handling
                        effective_location = location
                        # Improve Remote stability by providing a country context and remote=True when supported
                        is_remote = location.strip().lower() == "remote"
                        if is_remote:
                            effective_location = "United States"
                            kwargs["remote"] = True  # type: ignore[typeddict-item]
                        kwargs["location"] = effective_location

                        # Country hints (Some sites use specific country params; JobSpy will ignore unknown ones)
                        kwargs["country_indeed"] = "USA"
                        kwargs["linkedin_country"] = "us"

                        try:
                            site_jobs = jobspy_scrape_jobs(**kwargs)
                            df_site = pd.DataFrame(site_jobs)
                            if not df_site.empty:
                                aggregated.append(df_site)
                        except Exception as site_err:
                            print(f"Warning: {site} scrape failed for '{search_term}' in '{location}' (window {window['hours_old']}h): {site_err}")


                if aggregated:
                    jobs = pd.concat(aggregated, ignore_index=True)
                else:
                    raise RuntimeError(f"All site scrapes failed for '{search_term}' in '{location}'")

                # Add search metadata
                jobs["search_term"] = search_term
                jobs["search_location"] = location
                jobs["scraped_date"] = datetime.now()

                all_jobs_list.append(jobs)
                print(f"Found {len(jobs)} jobs for {search_term} in {location}")

        # Combine all results
        if all_jobs_list:
            self.all_jobs = pd.concat(all_jobs_list, ignore_index=True)
            print(f"\nTotal jobs scraped: {len(self.all_jobs)}")
            return self.all_jobs
        else:
            print("No jobs found!")
            return pd.DataFrame()

    def upload_to_s3(self, file_path: str, bucket: str, object_name: str = None) -> bool:
        """Upload a file to an S3 bucket."""
        if object_name is None:
            object_name = os.path.basename(file_path)
        s3 = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION')
        )
        s3.upload_file(file_path, bucket, object_name)
        print(f"Uploaded {file_path} to s3://{bucket}/{object_name}")
        return True