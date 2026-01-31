
import sys
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_validate, KFold
from sklearn.metrics import make_scorer, r2_score

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from salary_predictor import SalaryPredictor

# Load data
import boto3
from io import StringIO
from dotenv import load_dotenv

load_dotenv()

def load_data():
    if os.path.exists('data/processed_jobs.csv'):
        return pd.read_csv('data/processed_jobs.csv', low_memory=False)
    
    # Fallback to S3
    print("Loading from S3...")
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

df = load_data()

predictor = SalaryPredictor()
df_clean = predictor.prepare_data(df)

roles = ['Data Scientist', 'Machine Learning Engineer', 'Data Engineer', 'AI Engineer', 'Software Engineer']

print(f"{'Role':<25} {'Model':<20} {'CV R2 (Mean)':<15} {'CV R2 (Std)':<15} {'Train R2 (Mean)':<15}")
print("-" * 95)

for role in roles:
    role_df = df_clean[df_clean['role_category'] == role]
    if len(role_df) < 20:
        continue
        
    y = role_df['average_salary'].values
    
    # Get top skills like in training
    top_skills = predictor._select_top_skills(role_df, y, n_skills=10)
    X_df = predictor._build_features(role_df, fit_encoders=True, skill_cols=top_skills)
    X = X_df.values
    
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_split=5, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    }
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        results = cross_validate(model, X, y, cv=cv, scoring='r2', return_train_score=True)
        print(f"{role:<25} {name:<20} {results['test_score'].mean():<15.4f} {results['test_score'].std():<15.4f} {results['train_score'].mean():<15.4f}")
    
    print("-" * 95)
