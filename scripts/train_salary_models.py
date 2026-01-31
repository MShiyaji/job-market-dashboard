#!/usr/bin/env python
"""
Pre-train Salary Prediction Models

Run this script to generate pre-trained models that will be saved to data/models/.
These models are loaded automatically when the dashboard starts.

Usage:
    python scripts/train_salary_models.py
"""

import sys
import os
from io import StringIO

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import boto3
from dotenv import load_dotenv
from salary_predictor import SalaryPredictor

# Load environment variables
load_dotenv()


def load_data_from_s3():
    """Load processed jobs data from S3 bucket."""
    bucket = os.getenv("S3_BUCKET_NAME")
    key = os.getenv("S3_PROCESSED_KEY", "processed_jobs.csv")
    
    if not bucket:
        print("ERROR: S3_BUCKET_NAME not set in environment.")
        return None
    
    print(f"\nLoading data from S3: s3://{bucket}/{key}...")
    
    try:
        s3 = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION", "us-east-2"),
        )
        obj = s3.get_object(Bucket=bucket, Key=key)
        data = obj["Body"].read().decode("utf-8")
        df = pd.read_csv(StringIO(data), low_memory=False)
        print(f"Loaded {len(df)} jobs from S3")
        return df
    except Exception as e:
        print(f"ERROR: Failed to load from S3: {e}")
        return None


def main():
    print("=" * 60)
    print("Salary Prediction Model Training")
    print("=" * 60)
    
    # Load data from S3
    df = load_data_from_s3()
    
    if df is None:
        print("\nERROR: Could not load data from S3. Aborting.")
        sys.exit(1)
    
    # Add role_category if not present
    if 'role_category' not in df.columns:
        def categorize_job_title(title):
            if not isinstance(title, str):
                return "Other"
            t = title.lower().strip()
            if 'data scientist' in t or 'research scientist' in t:
                return 'Data Scientist'
            elif 'machine learning' in t or 'ml engineer' in t or 'applied scientist' in t or 'deep learning' in t:
                return 'Machine Learning Engineer'
            elif 'ai engineer' in t or 'artificial intelligence' in t:
                return 'AI Engineer'
            elif 'data engineer' in t or 'etl' in t or 'analytics engineer' in t or 'analytics specialist' in t:
                return 'Data Engineer'
            elif 'data analyst' in t or 'analyst' in t or 'analytics' in t or 'business intelligence' in t:
                return 'Data Analyst'
            elif 'software engineer' in t or 'developer' in t:
                return 'Software Engineer'
            else:
                return 'Other'
        
        df['role_category'] = df['title'].apply(categorize_job_title)
        print("Added role_category column")
    
    # Train models
    print("\nTraining models for each role category...")
    predictor = SalaryPredictor()
    results = predictor.train(df)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    
    for role, metrics in results.items():
        print(f"\n{role}:")
        print(f"  Best Model: {metrics['model_name']}")
        print(f"  R² Score:   {metrics['r2']:.3f}")
        print(f"  MAE:        ${metrics['mae']:,.0f}")
        print(f"  RMSE:       ${metrics['rmse']:,.0f}")
    
    print(f"\n✅ Models saved to data/models/")
    print("These will be loaded automatically when the dashboard starts.")


if __name__ == "__main__":
    main()
