
import sys
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from salary_predictor import SalaryPredictor

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

# Load data
print("Loading data from S3...")
predictor = SalaryPredictor()
df = load_data_from_s3()
df_clean = predictor.prepare_data(df)

# Fix experience outliers locally to test impact
print("\nEffect of capping experience at 25 years:")
df_clean['years_experience_required'] = df_clean['years_experience_required'].clip(upper=25)

roles = ['Data Analyst', 'Data Scientist', 'Machine Learning Engineer']

for role in roles:
    print(f"\n--- Analysis for {role} ---")
    role_df = df_clean[df_clean['role_category'] == role].copy()
    
    if len(role_df) < 20:
        print("Not enough data.")
        continue
        
    y = role_df['average_salary'].values
    y_log = np.log1p(y)
    
    # Get features
    top_skills = predictor._select_top_skills(role_df, y, n_skills=10)
    X_df = predictor._build_features(role_df, fit_encoders=True, skill_cols=top_skills)
    X = X_df.values
    
    print(f"Sample size: {len(role_df)}")
    
    # KFold with shuffle
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Baseline Model
    rf = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
    scores = cross_val_score(rf, X, y, cv=cv, scoring='r2')
    print(f"Baseline R2: {scores.mean():.4f} (+/- {scores.std():.4f})")
    
    # Feature Importance
    rf.fit(X, y)
    imp = pd.DataFrame({
        'feature': X_df.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\nTop 5 Features (Baseline):")
    print(imp.head(5))
    
    # Log Transform Model
    scores_log = cross_val_score(rf, X, y_log, cv=cv, scoring='r2')
    print(f"\nLog-Target R2: {scores_log.mean():.4f} (+/- {scores_log.std():.4f})")
    
    # Check if experience is broken
    exp_corr = role_df['years_experience_required'].corr(role_df['average_salary'])
    print(f"\nExperience Correlation: {exp_corr:.4f}")
