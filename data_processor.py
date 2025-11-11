import pandas as pd
import numpy as np
import re
import os
import boto3
from datetime import datetime
from botocore.exceptions import NoCredentialsError
from dotenv import load_dotenv

# Load environment variables (for AWS credentials and bucket)
load_dotenv()

class DataProcessor:
    """Process and clean scraped job data"""

    
    def __init__(self, raw_data_path='data/raw_jobs.csv'):
        """Load raw job data"""
        self.df = pd.read_csv(raw_data_path)
        self.processed_df = None
        
    def clean_salary_data(self):
        """Standardize salary information to annual USD"""
        
        def convert_to_annual(row):
            """Convert salary to annual amount"""
            if pd.isna(row['min_amount']) or pd.isna(row['max_amount']):
                return None, None
            
            min_sal = float(row['min_amount'])
            max_sal = float(row['max_amount'])
            interval = row['interval']
            
            # Convert to annual
            if interval == 'hourly':
                min_sal *= 2080  # 40 hours/week * 52 weeks
                max_sal *= 2080
            elif interval == 'monthly':
                min_sal *= 12
                max_sal *= 12
            elif interval == 'weekly':
                min_sal *= 52
                max_sal *= 52
            
            return min_sal, max_sal
        
        # Apply conversion
        self.df[['annual_min_salary', 'annual_max_salary']] = \
            self.df.apply(convert_to_annual, axis=1, result_type='expand')
        
        # Calculate average salary
        self.df['average_salary'] = \
            (self.df['annual_min_salary'] + self.df['annual_max_salary']) / 2

    def extract_skills(self):
        """Extract common data science skills from job descriptions"""
        
        # Define skill keywords
        skills_dict = {
            'Python': r'\bpython\b',
            'R': r'\b r\b|\br\s',
            'SQL': r'\bsql\b',
            'Machine Learning': r'machine learning|ml\b',
            'Deep Learning': r'deep learning|neural network',
            'NLP': r'\bnlp\b|natural language',
            'TensorFlow': r'tensorflow',
            'PyTorch': r'pytorch',
            'Scikit-learn': r'scikit-learn|sklearn',
            'Pandas': r'\bpandas\b',
            'NumPy': r'numpy',
            'Tableau': r'tableau',
            'Power BI': r'power bi|powerbi',
            'AWS': r'\baws\b|amazon web services',
            'Azure': r'\bazure\b',
            'GCP': r'\bgcp\b|google cloud',
            'Spark': r'\bspark\b',
            'Hadoop': r'hadoop',
            'Docker': r'docker',
            'Kubernetes': r'kubernetes|k8s',
            'Git': r'\bgit\b',
            'A/B Testing': r'a/b test|ab test',
            'Statistics': r'statistics|statistical',
            'Data Visualization': r'data visualization|data viz',
        }
        
        # Create binary columns for each skill
        for skill, pattern in skills_dict.items():
            self.df[f'skill_{skill}'] = \
                self.df['description'].str.contains(
                    pattern, 
                    case=False, 
                    na=False, 
                    regex=True
                ).astype(int)

    def extract_experience_level(self):
        """Extract required years of experience"""
        
        def parse_experience(description):
            if pd.isna(description):
                return None
            
            # Look for experience patterns
            patterns = [
                r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
                r'(\d+)-(\d+)\s*years',
                r'minimum\s*(?:of\s*)?(\d+)\s*years?',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, description.lower())
                if match:
                    return int(match.group(1))
            
            # Check for entry level keywords
            if re.search(r'entry.level|junior|new grad|0-2 years', description.lower()):
                return 0
            
            return None
        
        self.df['years_experience_required'] = \
            self.df['description'].apply(parse_experience)

    def categorize_companies(self):
        """Categorize companies by size if data available"""
        # This would require additional data sources
        # For now, we'll leave this as a placeholder
        pass

    def process_all(self):
        """Run all processing steps"""
        print("Processing job data...")
        
        # Remove duplicates
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates(subset=['title', 'company', 'location'], keep='first')
        print(f"Removed {initial_count - len(self.df)} duplicates")
        
        # Clean data
        self.clean_salary_data()
        self.extract_skills()
        self.extract_experience_level()
        
        # Create processed dataframe
        self.processed_df = self.df.copy()
        
        print(f"Processed {len(self.processed_df)} jobs")
        return self.processed_df

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
        try:
            s3.upload_file(file_path, bucket, object_name)
            print(f"Uploaded {file_path} to s3://{bucket}/{object_name}")
            return True
        except NoCredentialsError:
            print("AWS credentials not available.")
            return False
        except Exception as e:
            print(f"Failed to upload to S3: {e}")
            return False

    def save_processed_data(self, filename='data/processed_jobs.csv'):
        """Save processed data and upload to S3"""
        if self.processed_df is not None:
            os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
            self.processed_df.to_csv(filename, index=False)
            print(f"Processed data saved to {filename}")
            
            # Upload to S3
            bucket = os.getenv('S3_BUCKET_NAME')
            if bucket:
                self.upload_to_s3(filename, bucket)
            else:
                print("S3_BUCKET_NAME not set in environment; skipping S3 upload.")
        else:
            print("No processed data to save! Run process_all() first.")

    def get_summary_statistics(self):
        """Generate summary statistics"""
        if self.processed_df is None:
            print("No processed data! Run process_all() first.")
            return
        
        stats = {
            'total_jobs': len(self.processed_df),
            'unique_companies': self.processed_df['company'].nunique(),
            'avg_salary': self.processed_df['average_salary'].mean(),
            'median_salary': self.processed_df['average_salary'].median(),
            'jobs_with_salary': self.processed_df['average_salary'].notna().sum(),
            'remote_jobs': (self.processed_df['is_remote'] == True).sum(),
        }
        
        return stats

# Example usage
if __name__ == "__main__":
    processor = DataProcessor('data/raw_jobs.csv')
    processed_df = processor.process_all()
    processor.save_processed_data()

    # Print summary
    stats = processor.get_summary_statistics()
    print("\nSummary Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")