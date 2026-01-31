import sys
import os
from io import StringIO
sys.path.insert(0, '.')

import pandas as pd
import boto3
from dotenv import load_dotenv
from salary_predictor import SalaryPredictor

load_dotenv()

bucket = os.getenv('S3_BUCKET_NAME')
key = os.getenv('S3_PROCESSED_KEY', 'processed_jobs.csv')

s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION', 'us-east-2'),
)
obj = s3.get_object(Bucket=bucket, Key=key)
data = obj['Body'].read().decode('utf-8')
df = pd.read_csv(StringIO(data), low_memory=False)

sp = SalaryPredictor()

# Test Regex
print('=== Testing Regex Update ===')
test_cases = [
    "Must have an advanced degree in Engineering.",
    "Postgraduate qualification preferred.",
    "Masters degree in CS.",
    "No degree required."
]
for text in test_cases:
    extracted = sp._extract_education_level(text)
    print(f'Text: "{text}" -> {extracted}')

print('\n=== Education Level Distribution in Training Data (After Update) ===')
df_clean = sp.prepare_data(df)
edu_counts = df_clean['education_level'].value_counts(dropna=False)
print(edu_counts)
