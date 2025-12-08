import pandas as pd
import os

print("Starting analysis...")
file_path = 'data/processed_jobs.csv'
if not os.path.exists(file_path):
    print("No processed data found.")
else:
    df = pd.read_csv(file_path, low_memory=False)
    if 'date_posted' in df.columns:
        df['date_posted'] = pd.to_datetime(df['date_posted'], errors='coerce')
        daily_counts = df.groupby(df['date_posted'].dt.date).size()
        print("Daily Job Counts (Last 7 Days):")
        print(daily_counts.tail(7))
        print(f"\nAverage jobs per day (last 7 days): {daily_counts.tail(7).mean():.1f}")
    else:
        print("No 'date_posted' column found.")
