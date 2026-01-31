#!/usr/bin/env python
"""
Diagnose Low R2 Scores

Investigates why Data Analyst, Data Scientist, and ML Engineer models have low performance.
checks:
- Data quality (missing values)
- target distribution (salary outliers)
- Feature correlations
- Model comparison (Linear vs Tree-based)
"""

import sys
import os
import pandas as pd
import numpy as np
from io import StringIO
import boto3
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from salary_predictor import SalaryPredictor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import r2_score

load_dotenv()

def load_data_from_s3():
    """Load processed jobs data from S3 bucket."""
    bucket = os.getenv("S3_BUCKET_NAME")
    key = os.getenv("S3_PROCESSED_KEY", "processed_jobs.csv")
    
    if not bucket:
        print("ERROR: S3_BUCKET_NAME not set in environment.")
        return None
    
    print(f"Loading data from S3: s3://{bucket}/{key}...")
    
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
        return df
    except Exception as e:
        print(f"ERROR: Failed to load from S3: {e}")
        return None

def diagnose_role(role, df_clean, predictor):
    print(f"\n{'='*20} DIAGNOSING: {role} {'='*20}")
    
    role_df = df_clean[df_clean['role_category'] == role]
    print(f"Sample size: {len(role_df)}")
    
    if len(role_df) < 15:
        print("Insufficient data.")
        return

    # 1. Check Target Distribution
    salaries = role_df['average_salary']
    print("\n--- Salary Distribution ---")
    print(salaries.describe().to_string())
    print(f"Skewness: {salaries.skew():.2f}")
    
    # Check for potential bad data (e.g. hourly rates mixed in?)
    low_salary = (salaries < 30000).sum()
    high_salary = (salaries > 500000).sum()
    print(f"Salaries < 30k: {low_salary}")
    print(f"Salaries > 500k: {high_salary}")

    # 2. Check Feature Quality
    print("\n--- Feature Quality ---")
    print("Education 'No education':", (role_df['education_level'] == 'No education').sum(), 
          f"({(role_df['education_level'] == 'No education').mean():.1%})")
    print("Experience missing (filled):", (role_df['years_experience_required'].isna()).sum())
    
    # 3. Model Comparison
    print("\n--- Model Comparison (5-fold CV R2) ---")
    y = salaries.values
    
    # Get features using predictor logic
    top_skills = predictor._select_top_skills(role_df, y, n_skills=10)
    print(f"Top Skills used: {[s.replace('skill_', '') for s in top_skills[:5]]}...")
    
    X_df = predictor._build_features(role_df, fit_encoders=True, skill_cols=top_skills)
    X = X_df.values
    
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42)
    }
    
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        print(f"{name:20s}: Mean R2 = {scores.mean():.4f} (Std: {scores.std():.4f})")

def main():
    df = load_data_from_s3()
    if df is None:
        return
        
    predictor = SalaryPredictor()
    # Use predictor's prep logic (which now has the 'No education' fix)
    df_clean = predictor.prepare_data(df)
    
    target_roles = ['Data Analyst', 'Data Scientist', 'Machine Learning Engineer']
    
    for role in target_roles:
        diagnose_role(role, df_clean, predictor)

if __name__ == "__main__":
    main()
