import fitz  # PyMuPDF
import google.generativeai as genai
import os
from dotenv import load_dotenv
import json
import pandas as pd
import time
import pickle
from pathlib import Path

# Load environment variables
load_dotenv()

# Cache directory for resume skills
CACHE_DIR = Path('data/.cache')
CACHE_DIR.mkdir(parents=True, exist_ok=True)

class ResumeAnalyzer:
    """Parse resume and match against job requirements using AI"""
    
    def __init__(self, resume_path='data/resume.pdf'):
        """Initialize with resume path and Gemini API"""
        self.resume_path = resume_path
        self.resume_text = None
        self.skills = {}
        
        # Initialize Gemini client - try RESUME_GEMINI_API_KEY first, fallback to GEMINI_API_KEY
        api_key = os.getenv('RESUME_GEMINI_API_KEY') or os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY or RESUME_GEMINI_API_KEY not found in .env file")
        
        genai.configure(api_key=api_key)
    
    def extract_resume_text(self):
        """Extract text from PDF resume"""
        try:
            doc = fitz.open(self.resume_path)
            text = ""
            for page in doc:
                text += page.get_text()
            self.resume_text = text
            print(f"Extracted {len(text)} characters from resume")
            return text
        except Exception as e:
            print(f"Error extracting resume text: {e}")
            return None
    
    def extract_resume_skills(self):
        """Use Gemini to extract skills from resume (with caching)"""
        if not self.resume_text:
            self.extract_resume_text()
        
        # Generate cache key based on resume content hash
        import hashlib
        resume_hash = hashlib.md5(self.resume_text.encode()).hexdigest()
        cache_file = CACHE_DIR / f'resume_skills_{resume_hash}.pkl'
        
        # Try to load from cache first
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    skills_data = pickle.load(f)
                    self.skills = skills_data
                    print("✓ Loaded skills from cache (saved API call)")
                    return skills_data
            except Exception as e:
                print(f"Cache read failed: {e}")
        
        # If not cached, extract with API
        prompt = f"""
Analyze this resume and extract:
1. Technical skills (programming languages, tools, frameworks)
2. Years of experience in data science/AI
3. Education level and field
4. Key projects and achievements

Resume:
{self.resume_text}

Return ONLY valid JSON with keys: 
technical_skills (list), years_experience (number), 
education (string), key_projects (list).
"""
        
        try:
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            response = model.generate_content(prompt)
            
            # Parse JSON response
            result_text = response.text.strip()
            # Extract JSON from response (handle markdown code blocks)
            if '```json' in result_text:
                result_text = result_text.split('```json')[1].split('```')[0].strip()
            elif '```' in result_text:
                result_text = result_text.split('```')[1].split('```')[0].strip()
            
            skills_data = json.loads(result_text)
            self.skills = skills_data
            
            # Cache the result
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(skills_data, f)
                print("Cached extracted skills for future use")
            except Exception as e:
                print(f"Cache write failed: {e}")
            
            return skills_data
        
        except Exception as e:
            error_msg = str(e)
            if '429' in error_msg or 'quota' in error_msg.lower():
                print("\n" + "="*70)
                print("⚠️  GEMINI API QUOTA EXCEEDED")
                print("="*70)
                print("\nYou've hit the Gemini API free tier daily limit (200 requests/day).")
                print("\nYour options:")
                print("  1. WAIT: Quota resets at midnight Pacific Time")
                print("  2. UPGRADE: Enable billing to get Tier 1 limits")
                print("     - Visit: https://aistudio.google.com/app/apikey")
                print("     - Tier 1: 125,000 TPM, much higher daily limits")
                print("  3. NEW KEY: Get a fresh API key (temporary solution)")
                print("     - Visit: https://aistudio.google.com/app/apikey")
                print("\nCheck your usage: https://aistudio.google.com/usage?tab=rate-limit")
                print("="*70 + "\n")
            print(f"Error extracting skills with Gemini: {e}")
            return None
    
    def match_job_to_resume(self, job_description, job_title, required_skills):
        """
        DEPRECATED: This method is no longer used.
        Use analyze_job_market_fit() instead - it processes all jobs in a single batch API call.
        """
        raise NotImplementedError("Use analyze_job_market_fit() for batch processing instead")
    
    def analyze_job_market_fit(self, jobs_df, sample_size=10):
        """
        Analyze resume fit against multiple jobs using a SINGLE batch API call.
        Optimized for Gemini free tier (1M TPM limit).
        
        Args:
            jobs_df: DataFrame with job postings
            sample_size: Number of jobs to analyze (default 10 to stay under token limits)
        
        Returns:
            DataFrame with match scores and insights
        """
        print(f"Starting batch analysis with {len(jobs_df)} total jobs")
        
        if not self.skills:
            print("Extracting resume skills first...")
            self.extract_resume_skills()
            if not self.skills:
                print("ERROR: Failed to extract resume skills")
                return pd.DataFrame()
        
        # Sample jobs if too many
        if len(jobs_df) > sample_size:
            sample_jobs = jobs_df.sample(n=sample_size, random_state=42)
        else:
            sample_jobs = jobs_df
        
        print(f"Analyzing {len(sample_jobs)} jobs in single API call...")
        print(f"Note: Using sample_size={sample_size} to stay within free tier token limits")
        
        # Prepare aggregated job data with reduced description length
        jobs_summary = []
        for idx, job in sample_jobs.iterrows():
            skill_cols = [col for col in job.index if col.startswith('skill_')]
            required_skills = [col.replace('skill_', '') for col in skill_cols if job[col] == 1]
            
            # Limit to top 10 skills to reduce token usage
            required_skills = required_skills[:10]
            
            jobs_summary.append({
                'id': idx,
                'title': job.get('title', 'Unknown'),
                'company': job.get('company', 'Unknown'),
                'location': job.get('location', 'Unknown'),
                'required_skills': required_skills,
                'description_snippet': str(job.get('description', ''))[:200]  # Reduced from 500 to 200
            })
        
        # Single API call for all jobs
        prompt = f"""
You are an expert career advisor. Analyze how well this candidate's resume matches each of the following job postings.

Candidate Resume Summary:
Technical Skills: {self.skills.get('technical_skills', [])}
Years of Experience: {self.skills.get('years_experience', 'Unknown')}
Education: {self.skills.get('education', 'Unknown')}

Job Postings to Analyze:
{json.dumps(jobs_summary, indent=2)}

For EACH job, provide a match analysis. Return ONLY valid JSON (no markdown) as an array:
[
  {{
    "id": <job_id_from_input>,
    "match_percentage": <number 0-100>,
    "matched_skills": ["skill1", "skill2"],
    "missing_skills": ["skill3", "skill4"],
    "explanation": "brief explanation"
  }},
  ...
]
"""
        
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                model = genai.GenerativeModel('gemini-2.0-flash-exp')
                response = model.generate_content(prompt)
                
                result_text = response.text.strip()
                
                # Extract JSON from response
                if '```json' in result_text:
                    result_text = result_text.split('```json')[1].split('```')[0].strip()
                elif '```' in result_text:
                    result_text = result_text.split('```')[1].split('```')[0].strip()
                
                # Parse JSON array
                matches_data = json.loads(result_text)
                
                if not isinstance(matches_data, list):
                    print("ERROR: Expected array of matches from API")
                    return pd.DataFrame()
                
                # Build results dataframe
                results = []
                for match in matches_data:
                    job_id = match.get('id')
                    if job_id is None:
                        continue
                    
                    try:
                        job = sample_jobs.loc[job_id]
                    except KeyError:
                        print(f"Warning: Job ID {job_id} not found in sample")
                        continue
                    
                    results.append({
                        'job_title': job.get('title'),
                        'company': job.get('company'),
                        'location': job.get('location'),
                        'match_percentage': match.get('match_percentage', 0),
                        'matched_skills': match.get('matched_skills', []),
                        'missing_skills': match.get('missing_skills', []),
                        'explanation': match.get('explanation', ''),
                        'salary': job.get('average_salary'),
                    })
                
                print(f"✓ Successfully analyzed {len(results)} jobs in single API call")
                return pd.DataFrame(results)
            
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                print(f"Response preview: {result_text[:300]}")
                return pd.DataFrame()
            except Exception as e:
                error_msg = str(e)
                if '429' in error_msg or 'quota' in error_msg.lower():
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)
                        print(f"Rate limit hit, waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
                        time.sleep(wait_time)
                        continue
                    else:
                        print("\n" + "="*70)
                        print("⚠️  GEMINI API QUOTA EXCEEDED")
                        print("="*70)
                        print("\nYou've exhausted your Gemini API quota for today.")
                        print("\nFree Tier Limits (Gemini 2.0 Flash):")
                        print("  • 15 requests per minute")
                        print("  • 1,000,000 tokens per minute")
                        print("  • 200 requests per day ← You hit this!")
                        print("\nSolutions:")
                        print("  1. Wait until midnight Pacific Time (quota resets)")
                        print("  2. Upgrade to paid tier (Tier 1 = 125,000 TPM)")
                        print("     → https://aistudio.google.com/app/apikey")
                        print("  3. Use a different API key (temporary fix)")
                        print("\nMonitor usage: https://aistudio.google.com/usage?tab=rate-limit")
                        print("="*70 + "\n")
                        return pd.DataFrame()
                else:
                    print(f"ERROR in batch analysis: {e}")
                    return pd.DataFrame()
        
        return pd.DataFrame()
    
    def generate_insights(self, matches_df):
        """Generate personalized insights from match analysis"""
        
        if matches_df.empty:
            return "No match data available"
        
        # Calculate statistics
        avg_match = matches_df['match_percentage'].mean()
        good_fit_count = (matches_df['match_percentage'] >= 70).sum()
        total_jobs = len(matches_df)
        
        # Find most common missing skills
        all_missing = []
        for skills in matches_df['missing_skills']:
            if isinstance(skills, list):
                all_missing.extend(skills)
        
        from collections import Counter
        missing_counts = Counter(all_missing)
        top_missing = missing_counts.most_common(5)
        
        # Generate insight text
        insight_prompt = f"""
        Based on this job market analysis, provide personalized career advice:
        
        - Average match score: {avg_match:.1f}%
        - Jobs with good fit (70%+): {good_fit_count} out of {total_jobs}
        - Most common missing skills: {', '.join([skill for skill, count in top_missing])}
        
        Resume skills: {self.skills.get('technical_skills', [])}
        
        Provide:
        1. Overall assessment (2-3 sentences)
        2. Top 3 skills to develop
        3. Specific learning resources or actions
        4. Role positioning advice
        
        Keep response concise and actionable.
        """
        
        try:
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            response = model.generate_content(insight_prompt)
            return response.text
        except Exception as e:
            print(f"Error generating insights: {e}")
            return "Unable to generate insights"

# Example usage
if __name__ == "__main__":
    analyzer = ResumeAnalyzer('data/resume.pdf')
    
    # Extract resume skills
    skills = analyzer.extract_resume_skills()
    print("Extracted skills:", skills)
    
    # Load processed jobs
    jobs_df = pd.read_csv('data/processed_jobs.csv')
    
    # Analyze market fit
    matches = analyzer.analyze_job_market_fit(jobs_df, sample_size=10)
    matches.to_csv('data/job_matches.csv', index=False)
    
    # Generate insights
    insights = analyzer.generate_insights(matches)
    print("\nPersonalized Insights:")
    print(insights)