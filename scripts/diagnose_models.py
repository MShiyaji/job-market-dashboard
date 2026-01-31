#!/usr/bin/env python
"""
Diagnose low-accuracy salary prediction models.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import boto3
from io import StringIO
from dotenv import load_dotenv

load_dotenv()

def load_data_from_s3():
    bucket = os.getenv("S3_BUCKET_NAME")
    key = os.getenv("S3_PROCESSED_KEY", "processed_jobs.csv")
    
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION", "us-east-2"),
    )
    obj = s3.get_object(Bucket=bucket, Key=key)
    data = obj["Body"].read().decode("utf-8")
    return pd.read_csv(StringIO(data), low_memory=False)

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

def main():
    print("Loading data from S3...")
    df = load_data_from_s3()
    
    # Add role category
    if 'role_category' not in df.columns:
        df['role_category'] = df['title'].apply(categorize_job_title)
    
    # Filter to valid salaries
    df_valid = df[df['average_salary'].notna()]
    df_valid = df_valid[(df_valid['average_salary'] >= 30000) & (df_valid['average_salary'] <= 500000)]
    
    print("\n" + "=" * 70)
    print("DIAGNOSIS: Why Data Analyst & Data Scientist Have Low Accuracy")
    print("=" * 70)
    
    roles_to_check = ['Data Analyst', 'Data Scientist', 'Data Engineer', 'Software Engineer']
    
    for role in roles_to_check:
        role_df = df_valid[df_valid['role_category'] == role]
        salaries = role_df['average_salary']
        
        print(f"\n{'=' * 50}")
        print(f"  {role}")
        print(f"{'=' * 50}")
        print(f"  Sample size:      {len(role_df)}")
        print(f"  Salary range:     ${salaries.min():,.0f} - ${salaries.max():,.0f}")
        print(f"  Salary mean:      ${salaries.mean():,.0f}")
        print(f"  Salary median:    ${salaries.median():,.0f}")
        print(f"  Salary std dev:   ${salaries.std():,.0f}")
        print(f"  Coef of Variation:{(salaries.std() / salaries.mean() * 100):.1f}%")
        
        # Check experience distribution
        if 'years_experience_required' in role_df.columns:
            exp = role_df['years_experience_required'].dropna()
            print(f"\n  Experience range: {exp.min():.0f} - {exp.max():.0f} years")
            print(f"  Exp median:       {exp.median():.1f} years")
        
        # Check remote distribution
        if 'is_remote' in role_df.columns:
            remote_pct = role_df['is_remote'].fillna(False).mean() * 100
            print(f"  Remote jobs:      {remote_pct:.1f}%")
        
        # Show salary by experience correlation
        if 'years_experience_required' in role_df.columns:
            exp_salary_corr = role_df[['years_experience_required', 'average_salary']].corr().iloc[0, 1]
            print(f"\n  Exp-Salary corr:  {exp_salary_corr:.3f}")
        
        # Show salary percentiles
        print(f"\n  Salary percentiles:")
        for p in [10, 25, 50, 75, 90]:
            print(f"    {p}th: ${salaries.quantile(p/100):,.0f}")
        
        # Check for outliers (IQR method)
        q1, q3 = salaries.quantile(0.25), salaries.quantile(0.75)
        iqr = q3 - q1
        outliers = ((salaries < q1 - 1.5*iqr) | (salaries > q3 + 1.5*iqr)).sum()
        print(f"\n  Outliers (IQR):   {outliers} ({outliers/len(salaries)*100:.1f}%)")

if __name__ == "__main__":
    main()
