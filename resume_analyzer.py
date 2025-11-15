import fitz  # PyMuPDF
import google.generativeai as genai
import os
from dotenv import load_dotenv
import json
import pandas as pd

# Load environment variables
load_dotenv()

class ResumeAnalyzer:
    """Parse resume and match against job requirements using Gemini AI"""
    
    def __init__(self, resume_path='data/resume.pdf'):
        """Initialize with resume path and Gemini API"""
        self.resume_path = resume_path
        self.resume_text = None
        self.skills = {}
        
        # Initialize Gemini client
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file")
        
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
        """Use Gemini to extract skills from resume"""
        if not self.resume_text:
            self.extract_resume_text()
        
        prompt = f"""
        Analyze this resume and extract:
        1. Technical skills (programming languages, tools, frameworks)
        2. Years of experience in data science/AI
        3. Education level and field
        4. Key projects and achievements
        
        Resume:
        {self.resume_text}
        
        Return the information in JSON format with keys: 
        technical_skills (list), years_experience (number), 
        education (string), key_projects (list).
        """
        
        try:
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            response = model.generate_content(prompt)
            
            # Parse JSON response
            result_text = response.text
            # Extract JSON from response (handle markdown code blocks)
            if '```json' in result_text:
                result_text = result_text.split('```json')[1].split('```')[0]
            elif '```' in result_text:
                result_text = result_text.split('```')[1].split('```')[0]
            
            skills_data = json.loads(result_text.strip())
            self.skills = skills_data
            return skills_data
        
        except Exception as e:
            print(f"Error extracting skills with Gemini: {e}")
            return None
    
    def match_job_to_resume(self, job_description, job_title, required_skills):
        """
        Match a single job against resume using Gemini
        
        Returns match percentage and explanation
        """
        if not self.resume_text:
            self.extract_resume_text()
        
        prompt = f"""
        You are an expert career advisor. Analyze how well this candidate's resume 
        matches the following job posting.
        
        Job Title: {job_title}
        
        Job Description:
        {job_description[:2000]}  # Limit length for API
        
        Required Skills:
        {required_skills}
        
        Candidate Resume (Summary):
        Technical Skills: {self.skills.get('technical_skills', [])}
        Years of Experience: {self.skills.get('years_experience', 'Unknown')}
        Education: {self.skills.get('education', 'Unknown')}
        
        Provide:
        1. Match percentage (0-100)
        2. Matched skills
        3. Missing skills
        4. Brief explanation
        
        Return as JSON with keys: match_percentage, matched_skills, 
        missing_skills, explanation
        """
        
        try:
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            response = model.generate_content(prompt)
            
            result_text = response.text
            # Extract JSON
            if '```json' in result_text:
                result_text = result_text.split('```json')[1].split('```')[0]
            elif '```' in result_text:
                result_text = result_text.split('```')[1].split('```')[0]
            
            match_data = json.loads(result_text.strip())
            return match_data
        
        except Exception as e:
            print(f"Error matching job: {e}")
            return None
    
    def analyze_job_market_fit(self, jobs_df, sample_size=20):
        """
        Analyze fit for multiple jobs and generate insights
        
        Args:
            jobs_df: DataFrame with job postings
            sample_size: Number of jobs to analyze (API limits)
        
        Returns:
            DataFrame with match scores and insights
        """
        if not self.skills:
            self.extract_resume_skills()
        
        # Sample jobs if too many
        if len(jobs_df) > sample_size:
            sample_jobs = jobs_df.sample(n=sample_size, random_state=42)
        else:
            sample_jobs = jobs_df
        
        matches = []
        
        print(f"Analyzing fit for {len(sample_jobs)} jobs...")
        
        for idx, job in sample_jobs.iterrows():
            # Extract required skills from job
            skill_cols = [col for col in job.index if col.startswith('skill_')]
            required_skills = [col.replace('skill_', '') for col in skill_cols if job[col] == 1]
            
            match_result = self.match_job_to_resume(
                job_description=job.get('description', ''),
                job_title=job.get('title', ''),
                required_skills=required_skills
            )
            
            if match_result:
                matches.append({
                    'job_title': job.get('title'),
                    'company': job.get('company'),
                    'location': job.get('location'),
                    'match_percentage': match_result.get('match_percentage', 0),
                    'matched_skills': match_result.get('matched_skills', []),
                    'missing_skills': match_result.get('missing_skills', []),
                    'explanation': match_result.get('explanation', ''),
                    'salary': job.get('average_salary'),
                })
        
        matches_df = pd.DataFrame(matches)
        return matches_df
    
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