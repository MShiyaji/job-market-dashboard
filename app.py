import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import io
import boto3
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv
import numpy as np
from sklearn.linear_model import LinearRegression
# Import custom modules
from scraper import JobScraper
from data_processor import DataProcessor
# Lazy import for resume analyzer to avoid startup failures if optional deps missing

# Load environment variables (for AWS credentials and bucket)
load_dotenv()

# --- S3 helpers -------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=300)
def read_csv_from_s3(bucket: str, key: str) -> pd.DataFrame | None:
    """Read a CSV from S3 into a DataFrame; return None on failure."""
    try:
        s3 = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION"),
        )
        obj = s3.get_object(Bucket=bucket, Key=key)
        data = obj["Body"].read()
        return pd.read_csv(io.BytesIO(data))
    except (BotoCoreError, ClientError, Exception) as e:
        print(f"S3 load failed for s3://{bucket}/{key}: {e}")
        return None

@st.cache_data(show_spinner=False, ttl=300)
def read_csv_local(path: str) -> pd.DataFrame | None:
    """Read a CSV from local disk; return None on failure."""
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"Local CSV load failed for {path}: {e}")
        return None

def autoload_jobs_from_s3() -> bool:
    """Attempt to auto-load processed jobs from S3 into session_state."""
    bucket = os.getenv("S3_BUCKET_NAME")
    key = os.getenv("S3_PROCESSED_KEY", "processed_jobs.csv")
    if not bucket:
        return False
    df = read_csv_from_s3(bucket, key)
    if df is not None and not df.empty:
        st.session_state.jobs_df = df
        st.session_state.data_loaded = True
        return True
    return False
# Page configuration
st.set_page_config(
page_title="Data Science Job Market Dashboard",
page_icon="",
layout="wide",
initial_sidebar_state="expanded"
)
# Default Plotly config (avoid deprecated keyword args; control mode bar, zoom, logo)
PLOTLY_CONFIG = {
    "displaylogo": False,
    "scrollzoom": True,
}
# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)
# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'jobs_df' not in st.session_state:
    st.session_state.jobs_df = None
if 'matches_df' not in st.session_state:
    st.session_state.matches_df = None
# Always try to autoload from S3 on app start
loaded = autoload_jobs_from_s3()
if not loaded and not st.session_state.data_loaded:
    # Fallback to local if S3 fails and no data loaded yet
    if os.path.exists('data/processed_jobs.csv'):
        df_local = read_csv_local('data/processed_jobs.csv')
        if df_local is not None and not df_local.empty:
            st.session_state.jobs_df = df_local
            st.session_state.data_loaded = True
            print("Loaded data from local data/processed_jobs.csv as fallback")
        else:
            print("Local fallback load failed or returned empty dataframe")
# Main title
st.markdown('<p> Data Science Job Market Dashboard</p>', 
        unsafe_allow_html=True)
# Sidebar
st.sidebar.title(" Dashboard Controls")
# Data loading section
st.sidebar.header("1. Data Management")
if st.sidebar.button(" Scrape New Jobs"):
    with st.spinner("Scraping job boards... This may take several minutes"):
        scraper = JobScraper(
            search_terms=["data scientist", "machine learning engineer"],
            locations=["San Francisco, CA", "Remote"],
            results_per_site=500  # Comprehensive coverage for 30 days
        )
        jobs = scraper.scrape_jobs()
        scraper.save_jobs()
        
        # Process data
        processor = DataProcessor('data/raw_jobs.csv')
        processed = processor.process_all()
        processor.save_processed_data()
        
        st.sidebar.success(f"✅ Scraped {len(jobs)} jobs!")
if st.sidebar.button(" Load Existing Data"):
    try:
        st.session_state.jobs_df = pd.read_csv('data/processed_jobs.csv')
        st.session_state.data_loaded = True
        st.sidebar.success(f"✅ Loaded {len(st.session_state.jobs_df)} jobs!")
    except FileNotFoundError:
        st.sidebar.error("No data file found. Please scrape jobs first.")
# Resume analysis section
st.sidebar.header("2. Resume Analysis")
uploaded_resume = st.sidebar.file_uploader(
"Upload Your Resume (PDF)",
type=['pdf'],
help="Upload your resume to get personalized job match insights"
)
if uploaded_resume:
    # Save uploaded file
    with open('data/resume.pdf', 'wb') as f:
        f.write(uploaded_resume.getbuffer())
    st.sidebar.success("✅ Resume uploaded!")
if st.sidebar.button(" Analyze Resume Fit") and st.session_state.data_loaded:
    if os.path.exists('data/resume.pdf'):
        with st.spinner("Analyzing your fit for jobs using Gemini AI..."):
            try:
                from resume_analyzer import ResumeAnalyzer
                analyzer = ResumeAnalyzer('data/resume.pdf')
                matches = analyzer.analyze_job_market_fit(
                    st.session_state.jobs_df,
                    sample_size=15
                )
                st.session_state.matches_df = matches
                matches.to_csv('data/job_matches.csv', index=False)
                st.sidebar.success("✅ Analysis complete!")
            except Exception as e:
                st.sidebar.error(f"Resume analyzer unavailable: {e}")
                st.sidebar.info("Install dependencies and set GEMINI_API_KEY in .env to enable this feature.")
    else:
        st.sidebar.error("Please upload a resume first!")
# Main dashboard content
# If no data is loaded yet, show a welcome screen and stop executing the rest of the app
if not st.session_state.data_loaded:
    st.markdown("""
## Welcome to the Data Science Job Market Dashboard! 

This dashboard helps you:
-  Visualize data science and AI job market trends
-  Analyze salary distributions across roles and locations
-  Identify the most in-demand skills
-  Get AI-powered personalized career insights

### Get Started:
1. Click **"Load Existing Data"** in the sidebar (if you have data)
2. OR click **"Scrape New Jobs"** to collect fresh job postings
3. Upload your resume to get personalized insights
4. Explore the dashboard tabs!

--

### About This Project
Built with:
- **JobSpy** for multi-platform job scraping
- **Gemini AI** for intelligent resume matching
- **Streamlit & Plotly** for interactive visualizations
- **Pandas** for data processing
""")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Built with ❤  by [Your Name]**")
    st.sidebar.markdown(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    st.stop()

df = st.session_state.jobs_df

# Categorize job titles into broader role categories
def categorize_job_title(title):
    if not isinstance(title, str):
        return "Other"
    t = title.lower().strip()
    if 'data scientist' in t:
        return 'Data Scientist'
    elif 'machine learning' in t or 'ml engineer' in t:
        return 'Machine Learning Engineer'
    elif 'ai engineer' in t or 'artificial intelligence' in t:
        return 'AI Engineer'
    elif 'data analyst' in t or 'analyst' in t or 'analytics' in t:
        return 'Data Analyst'
    elif 'data engineer' in t:
        return 'Data Engineer'
    elif 'software engineer' in t or 'developer' in t:
        return 'Software Engineer'
    else:
        return 'Other'

if 'title' in df.columns:
    df['role_category'] = df['title'].apply(categorize_job_title)

# Filters
st.sidebar.header("3. Filters")

# Normalize job type values to four canonical options for consistent filtering
def _normalize_job_type(val):
    if not isinstance(val, str):
        return None
    v = val.strip().lower().replace('_', '-')
    # Map common variants to canonical set
    if 'intern' in v:
        return 'internship'
    if 'part' in v:
        return 'part-time'
    if 'contract' in v or 'temp' in v:
        return 'contract'
    if 'full' in v or v == 'ft' or 'full-time' in v or 'fulltime' in v:
        return 'fulltime'
    return None

# Add a normalized job type column (non-destructive to original)
if 'job_type' in df.columns:
    try:
        df['job_type_norm'] = df['job_type'].apply(_normalize_job_type)
    except Exception:
        df['job_type_norm'] = None
else:
    df['job_type_norm'] = None

# Location filter
locations = ['All'] + sorted(df['location'].dropna().unique().tolist())
selected_location = st.sidebar.selectbox("Location", locations)

# Job type filter (restricted to four options)
allowed_job_types = ['fulltime', 'internship', 'part-time', 'contract']
job_types = ['All'] + allowed_job_types
selected_job_type = st.sidebar.selectbox("Job Type", job_types, index=0,
    help="Filter by job type. Options are limited to fulltime, internship, part-time, and contract.")

# Remote filter
remote_only = st.sidebar.checkbox("Remote Only")

# Experience filter
exp_levels = ['All', 'Entry / New Grad (0-1 yrs)', 'Mid-Level (2-4 yrs)']
selected_exp = st.sidebar.selectbox("Experience Level", exp_levels)

# Role filter
roles = ['All'] + sorted(df['role_category'].dropna().unique().tolist())
selected_role = st.sidebar.selectbox("Role", roles)

# Apply filters
filtered_df = df.copy()
if selected_location != 'All':
    filtered_df = filtered_df[filtered_df['location'] == selected_location]
if selected_job_type != 'All':
    # Use normalized job type for reliable matching
    filtered_df = filtered_df[filtered_df.get('job_type_norm').fillna('') == selected_job_type]
if remote_only:
    filtered_df = filtered_df[filtered_df['is_remote'] == True]

# Apply experience level filter (if available)
if 'years_experience_required' in filtered_df.columns and selected_exp != 'All':
    years = pd.to_numeric(filtered_df['years_experience_required'], errors='coerce')
    if selected_exp.startswith('Entry'):
        filtered_df = filtered_df[years <= 1]
    elif selected_exp.startswith('Mid-Level'):
        filtered_df = filtered_df[(years >= 2) & (years <= 4)]

# Apply role filter
if selected_role != 'All':
    filtered_df = filtered_df[filtered_df['role_category'] == selected_role]

# Key Metrics
st.header(" Key Metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Jobs", len(filtered_df))
with col2:
    st.metric("Unique Companies", filtered_df['company'].nunique())
with col3:
    avg_salary = filtered_df['average_salary'].mean()
    st.metric("Avg Salary", f"${avg_salary:,.0f}" if pd.notna(avg_salary) else "N/A")
with col4:
    remote_pct = (filtered_df['is_remote'].sum() / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
    st.metric("Remote %", f"{remote_pct:.1f}%")

# Tabs for different views
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Market Trends",
    "💰 Salary Analysis",
    "🛠️ Skills Demand",
    "🤖 AI Insights",
    "📈 Role Analytics"
])

with tab1:
    st.subheader("Job Market Trends")
    
    # Jobs by location
    col1, col2 = st.columns(2)
    
    with col1:
        location_counts = filtered_df['location'].value_counts().head(10)
        total_jobs = len(filtered_df)
        location_pct = (location_counts / total_jobs * 100).round(2)
        fig = px.bar(
            x=location_pct.values,
            y=location_pct.index,
            orientation='h',
            title="Top 10 Locations",
            labels={'x': '% of Total Jobs', 'y': 'Location'}
        )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
    
    with col2:
        company_counts = filtered_df['company'].value_counts().head(10)
        total_jobs = len(filtered_df)
        company_pct = (company_counts / total_jobs * 100).round(2)
        fig = px.bar(
            x=company_pct.values,
            y=company_pct.index,
            orientation='h',
            title="Top 10 Hiring Companies",
            labels={'x': '% of Total Jobs', 'y': 'Company'}
        )
    fig.update_layout(height=400, yaxis=dict(categoryorder="array", categoryarray=company_pct.index.tolist()))
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
    
    # (Jobs posted over time chart removed per request)

with tab2:
    st.subheader("Salary Analysis")
    
    # Salary distribution
    salary_data = filtered_df['average_salary'].dropna()
    
    if len(salary_data) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                salary_data,
                nbins=30,
                title="Salary Distribution",
                labels={'value': 'Salary ($)', 'count': 'Frequency'}
            )
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
        
        with col2:
            fig = px.box(
                filtered_df,
                y='average_salary',
                title="Salary Box Plot",
                labels={'average_salary': 'Salary ($)'}
            )
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
        
        # Salary by job title (sorted highest to lowest)
        salary_by_title = (
            filtered_df.groupby('search_term')['average_salary']
            .mean()
            .sort_values(ascending=False)
        )
        fig = px.bar(
            x=salary_by_title.values,
            y=salary_by_title.index,
            orientation='h',
            title="Average Salary by Job Title",
            labels={'x': 'Average Salary ($)', 'y': 'Job Title'}
        )
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
    else:
        st.warning("No salary data available in filtered results")

with tab3:
    st.subheader("Skills in Demand")

    # Role-specific popular skills
    st.markdown("### 1) Popular skills")
    role_df = filtered_df

    # Extract skill columns
    skill_cols = [col for col in role_df.columns if col.startswith('skill_')]

    if skill_cols and len(role_df) > 0:
        skill_demand = {}
        total_jobs = len(role_df)
        for col in skill_cols:
            skill_name = col.replace('skill_', '')
            skill_demand[skill_name] = (role_df[col].sum() / total_jobs * 100)

        skill_demand_sorted = dict(sorted(skill_demand.items(), key=lambda x: x[1], reverse=True))

        # Top skills bar chart
        top_n = 15
        top_skills = dict(list(skill_demand_sorted.items())[:top_n])

        fig = px.bar(
            x=list(top_skills.values()),
            y=list(top_skills.keys()),
            orientation='h',
            title=(f"Top {top_n} Skills - {selected_role}" if selected_role != 'All' else f"Top {top_n} In-Demand Skills"),
            labels={'x': '% of Jobs Requiring Skill', 'y': 'Skill'},
            color=list(top_skills.values()),
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

        # Skills co-occurrence (within the selected role scope)
        st.subheader("Skill Combinations")
        st.write("Most common skill pairs in job postings:")

        from itertools import combinations
        skill_pairs = {}
        for _, row in role_df.iterrows():
            present_skills = [col.replace('skill_', '') for col in skill_cols if col in row and row[col] == 1]
            if len(present_skills) >= 2:
                for pair in combinations(present_skills, 2):
                    pair_key = ' + '.join(sorted(pair))
                    skill_pairs[pair_key] = skill_pairs.get(pair_key, 0) + 1

        if skill_pairs:
            top_pairs = sorted(skill_pairs.items(), key=lambda x: x[1], reverse=True)[:10]
            pairs_df = pd.DataFrame(top_pairs, columns=['Skill Combination', 'Frequency'])
            st.dataframe(pairs_df, use_container_width=True)
        else:
            st.info("Not enough co-occurring skills to show combinations for this selection.")
    else:
        st.warning("No skill data available for this selection")

    # Hot skills: skills increasing in frequency the most
    st.markdown("### 2) Hot Skills (Rising Fast)")
    if 'date_posted' in filtered_df.columns:
        df_dates = filtered_df.copy()
        df_dates['date_posted'] = pd.to_datetime(df_dates['date_posted'], errors='coerce')
        df_dates = df_dates.dropna(subset=['date_posted'])

        if len(df_dates) > 0:
            days = 30
            cutoff = datetime.now() - timedelta(days=days)
            df_30 = df_dates[df_dates['date_posted'] >= cutoff]

            skill_cols_all = [c for c in df_30.columns if c.startswith('skill_')]
            if skill_cols_all:
                # Calculate daily counts and percentages
                daily_counts = df_30.groupby(df_30['date_posted'].dt.date)[skill_cols_all].sum()
                daily_totals = df_30.groupby(df_30['date_posted'].dt.date).size()
                
                # Convert to percentages
                daily_pct = daily_counts.div(daily_totals, axis=0) * 100
                
                # Ensure the grouped index has a consistent name/format for melting and plotting
                try:
                    daily_pct.index = pd.to_datetime(daily_pct.index)
                except Exception:
                    pass
                daily_pct.index.name = 'date'
                
                trends = []
                for col in skill_cols_all:
                    s = daily_pct[col]
                    if s.sum() == 0 or len(s) < 5:
                        continue
                    x = np.arange(len(s))
                    try:
                        slope = float(np.polyfit(x, s.values, 1)[0])  # per-day percentage point change
                    except Exception:
                        continue
                    trends.append({
                        'Skill': col.replace('skill_', ''),
                        'Trend (% pts/day)': round(slope, 4),
                        'Current %': round(s.iloc[-1], 2),
                        '30d avg %': round(s.mean(), 2)
                    })

                if trends:
                    trends_df = pd.DataFrame(trends).sort_values('Trend (% pts/day)', ascending=False).head(10)
                    st.dataframe(trends_df.reset_index(drop=True), use_container_width=True)

                    # Optional: visualize top 5 trending skills over time
                    top_skills_names = trends_df['Skill'].head(5).tolist()
                    plot_cols = [f'skill_{n}' for n in top_skills_names]
                    melted = daily_pct[plot_cols].reset_index().melt(id_vars='date', var_name='skill', value_name='percentage')
                    melted['skill'] = melted['skill'].str.replace('skill_', '', regex=False)
                    fig_trend = px.line(
                        melted,
                        x='date', y='percentage', color='skill',
                        title='Top Trending Skills - % of Jobs (Last 30 Days)', markers=True,
                        labels={'date': 'Date', 'percentage': '% of Jobs', 'skill': 'Skill'},
                        category_orders={'skill': top_skills_names}
                    )
                    fig_trend.update_xaxes(range=[datetime(2025, 10, 1), datetime.now()])
                    st.plotly_chart(fig_trend, use_container_width=True, config=PLOTLY_CONFIG)
                else:
                    st.info("Not enough recent data to identify trending skills.")
            else:
                st.info("No skill columns found in the recent data window.")
        else:
            st.info("No valid dates available to compute trends.")
    else:
        st.info("Dataset lacks 'date_posted' column; cannot compute trends.")

    # Falling skills: skills declining in frequency the most
    st.markdown("### 3) Cooling Skills (Declining Fast)")
    if 'date_posted' in filtered_df.columns:
        df_dates = filtered_df.copy()
        df_dates['date_posted'] = pd.to_datetime(df_dates['date_posted'], errors='coerce')
        df_dates = df_dates.dropna(subset=['date_posted'])

        if len(df_dates) > 0:
            days = 30
            cutoff = datetime.now() - timedelta(days=days)
            df_30 = df_dates[df_dates['date_posted'] >= cutoff]

            skill_cols_all = [c for c in df_30.columns if c.startswith('skill_')]
            if skill_cols_all:
                # Calculate daily counts and percentages
                daily_counts = df_30.groupby(df_30['date_posted'].dt.date)[skill_cols_all].sum()
                daily_totals = df_30.groupby(df_30['date_posted'].dt.date).size()
                
                # Convert to percentages
                daily_pct = daily_counts.div(daily_totals, axis=0) * 100
                
                # Ensure the grouped index has a consistent name/format for melting and plotting
                try:
                    daily_pct.index = pd.to_datetime(daily_pct.index)
                except Exception:
                    pass
                daily_pct.index.name = 'date'
                
                declining_trends = []
                for col in skill_cols_all:
                    s = daily_pct[col]
                    if s.sum() == 0 or len(s) < 5:
                        continue
                    x = np.arange(len(s))
                    try:
                        slope = float(np.polyfit(x, s.values, 1)[0])  # per-day percentage point change
                    except Exception:
                        continue
                    declining_trends.append({
                        'Skill': col.replace('skill_', ''),
                        'Trend (% pts/day)': round(slope, 4),
                        'Current %': round(s.iloc[-1], 2),
                        '30d avg %': round(s.mean(), 2)
                    })

                if declining_trends:
                    # Sort ascending to get most negative slopes (biggest declines)
                    declining_df = pd.DataFrame(declining_trends).sort_values('Trend (% pts/day)', ascending=True).head(10)
                    st.dataframe(declining_df.reset_index(drop=True), use_container_width=True)

                    # Optional: visualize top 5 declining skills over time
                    top_declining_names = declining_df['Skill'].head(5).tolist()
                    plot_cols_decline = [f'skill_{n}' for n in top_declining_names]
                    melted_decline = daily_pct[plot_cols_decline].reset_index().melt(id_vars='date', var_name='skill', value_name='percentage')
                    melted_decline['skill'] = melted_decline['skill'].str.replace('skill_', '', regex=False)
                    fig_decline = px.line(
                        melted_decline,
                        x='date', y='percentage', color='skill',
                        title='Top Declining Skills - % of Jobs (Last 30 Days)', markers=True,
                        labels={'date': 'Date', 'percentage': '% of Jobs', 'skill': 'Skill'}
                    )
                    fig_decline.update_xaxes(range=[datetime(2025, 10, 1), datetime.now()])
                    st.plotly_chart(fig_decline, use_container_width=True, config=PLOTLY_CONFIG)
                else:
                    st.info("Not enough recent data to identify declining skills.")
            else:
                st.info("No skill columns found in the recent data window.")
        else:
            st.info("No valid dates available to compute trends.")
    else:
        st.info("Dataset lacks 'date_posted' column; cannot compute trends.")

with tab4:
    st.subheader(" AI-Powered Insights")
    
    if st.session_state.matches_df is not None:
        matches_df = st.session_state.matches_df
        
        # Overall match statistics
        st.markdown("### Your Job Market Fit")
        
        col1, col2, col3 = st.columns(3)
        
        avg_match = matches_df['match_percentage'].mean()
        good_matches = (matches_df['match_percentage'] >= 70).sum()
        
        with col1:
            st.metric("Average Match Score", f"{avg_match:.1f}%")
        with col2:
            st.metric("Strong Fits (70%+)", f"{good_matches}/{len(matches_df)}")
        with col3:
            top_match = matches_df['match_percentage'].max()
            st.metric("Best Match", f"{top_match:.0f}%")
        
        # Match distribution
        fig = px.histogram(
            matches_df,
            x='match_percentage',
            nbins=20,
            title="Distribution of Match Scores",
            labels={'match_percentage': 'Match Percentage', 'count': 'Number of Jobs'}
        )
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
        
        # Top matches
        st.markdown("### Your Best Job Matches")
        top_matches = matches_df.nlargest(5, 'match_percentage')
        
        for idx, row in top_matches.iterrows():
            with st.expander(f" {row['job_title']} at {row['company']} - {row['match_percentage']:.0f}% Match"):
                st.write(f"**Location:** {row['location']}")
                if pd.notna(row.get('salary')):
                    st.write(f"**Salary:** ${row['salary']:,.0f}")
                st.write(f"**Matched Skills:** {', '.join(row['matched_skills']) if isinstance(row['matched_skills'], list) else 'N/A'}")
                st.write(f"**Skills to Develop:** {', '.join(row['missing_skills']) if isinstance(row['missing_skills'], list) else 'N/A'}")
                st.write(f"**AI Analysis:** {row['explanation']}")
        
        # Skill gap analysis
        st.markdown("### Skill Gap Analysis")
        
        all_missing = []
        for skills_list in matches_df['missing_skills']:
            if isinstance(skills_list, list):
                all_missing.extend(skills_list)
        
        if all_missing:
            from collections import Counter
            missing_counts = Counter(all_missing)
            top_missing = dict(missing_counts.most_common(10))
            
            fig = px.bar(
                x=list(top_missing.values()),
                y=list(top_missing.keys()),
                orientation='h',
                title="Skills You Should Develop",
                labels={'x': 'Frequency in Job Postings', 'y': 'Skill'},
                color=list(top_missing.values()),
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
        
        # Generate personalized insights
        if st.button("Generate Personalized Career Advice"):
            with st.spinner("Generating insights with Gemini AI..."):
                try:
                    from resume_analyzer import ResumeAnalyzer
                    analyzer = ResumeAnalyzer('data/resume.pdf')
                    insights = analyzer.generate_insights(matches_df)
                except Exception as e:
                    st.error(f"Resume analyzer unavailable: {e}")
                    st.info("Install dependencies and set GEMINI_API_KEY in .env to enable this feature.")
                    insights = ""
                
                st.markdown("###  Personalized Recommendations")
                st.markdown(insights)
    else:
        st.info("Upload your resume and click 'Analyze Resume Fit' in the sidebar to see insights.")
        
        st.markdown("""
        **What you'll get:**
        - Match percentage for each job posting
        - Skills you have vs. skills required
        - Personalized recommendations for skill development
        - Career positioning advice
        """)

with tab5:
    st.subheader("📈 Role Analytics & Predictions")
    
    # Ensure we have date data
    if 'date_posted' not in filtered_df.columns:
        st.warning("No date information available in the dataset. Cannot generate time-based analytics.")
        st.info("💡 Make sure your scraped data includes 'date_posted' field.")
        st.stop()
    
    try:
        df_with_dates = filtered_df.copy()
        df_with_dates['date_posted'] = pd.to_datetime(df_with_dates['date_posted'], errors='coerce')
        df_with_dates = df_with_dates.dropna(subset=['date_posted'])
        
        if len(df_with_dates) == 0:
            st.warning("No valid dates found in the dataset.")
            st.stop()
        
        # Define 30 days ago
        thirty_days_ago = datetime.now() - timedelta(days=30)
        df_last_30 = df_with_dates[df_with_dates['date_posted'] >= thirty_days_ago]
        
        # Chart 1: Role counts in last 30 days
        st.markdown("### 1️⃣ Role Distribution (Last 30 Days)")
        
        if len(df_last_30) > 0:
            role_counts = df_last_30['title'].value_counts().head(15)
            total_jobs = len(df_last_30)
            role_pct = (role_counts / total_jobs * 100).round(2)
            
            fig1 = px.bar(
                x=role_pct.values,
                y=role_pct.index,
                orientation='h',
                title=f"Top 15 Role Titles (Last 30 Days) - Total: {len(df_last_30)} Jobs",
                labels={'x': '% of Total Jobs', 'y': 'Role Title'},
                color=role_pct.values,
                color_continuous_scale='Viridis'
            )
            fig1.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig1, use_container_width=True, config=PLOTLY_CONFIG)
            
            # Show stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Roles (30d)", len(df_last_30))
            with col2:
                st.metric("Unique Titles", df_last_30['title'].nunique())
            with col3:
                avg_per_day = len(df_last_30) / 30
                st.metric("Avg Jobs/Day", f"{avg_per_day:.1f}")
        else:
            st.warning("No jobs found in the last 30 days")
        
        # Chart 2: Daily growth rate by role
        st.markdown("### 2️⃣ Daily Role Growth Rate")
        
        if len(df_with_dates) > 0:
            # Get top roles to track
            top_roles = df_with_dates['title'].value_counts().head(8).index.tolist()
            
            # Calculate daily counts and percentages for each top role
            daily_role_counts = df_with_dates[df_with_dates['title'].isin(top_roles)].groupby(
                [df_with_dates['date_posted'].dt.date, 'title']
            ).size().reset_index(name='count')
            
            # Calculate daily totals to get percentages
            daily_totals = df_with_dates.groupby(df_with_dates['date_posted'].dt.date).size().reset_index(name='total')
            daily_totals.columns = ['date', 'total']
            
            # Merge and calculate percentages
            daily_role_counts.columns = ['date', 'title', 'count']
            daily_role_counts = daily_role_counts.merge(daily_totals, on='date', how='left')
            daily_role_counts['percentage'] = (daily_role_counts['count'] / daily_role_counts['total'] * 100).round(2)
            
            fig2 = px.line(
                daily_role_counts,
                x='date',
                y='percentage',
                color='title',
                title="Daily Role Distribution - % of Jobs (Top 8 Roles)",
                labels={'date': 'Date', 'percentage': '% of Jobs Posted', 'title': 'Role'},
                markers=True
            )
            fig2.update_layout(height=500)
            fig2.update_xaxes(range=[datetime(2025, 10, 1), datetime.now()])
            st.plotly_chart(fig2, use_container_width=True, config=PLOTLY_CONFIG)
            
            # Calculate growth rates
            st.markdown("#### Growth Metrics (Last 7 Days)")
            seven_days_ago = datetime.now() - timedelta(days=7)
            df_last_7 = df_with_dates[df_with_dates['date_posted'] >= seven_days_ago]
            
            if len(df_last_7) > 0:
                total_7d = len(df_last_7)
                growth_data = []
                for role in top_roles:
                    count_7d = len(df_last_7[df_last_7['title'] == role])
                    pct_7d = (count_7d / total_7d * 100) if total_7d > 0 else 0
                    avg_pct_per_day = pct_7d / 7
                    growth_data.append({
                        'Role': role,
                        '% of Jobs (7d)': round(pct_7d, 2),
                        'Avg %/Day': round(avg_pct_per_day, 2)
                    })
                
                growth_df = pd.DataFrame(growth_data).sort_values('Avg %/Day', ascending=False)
                st.dataframe(growth_df, use_container_width=True)
        else:
            st.warning("No date data available for growth analysis")
        
        # Chart 3: Time-series prediction
        st.markdown("### 3️⃣ Future Role Demand Prediction")
        
        if len(df_with_dates) > 10:  # Need sufficient data for prediction
            # Get top roles for prediction
            top_roles_predict = df_with_dates['title'].value_counts().head(5).index.tolist()
            
            # Allow user to select role
            selected_role = st.selectbox("Select Role to Predict", top_roles_predict)
            
            if selected_role:
                try:
                    role_data = df_with_dates[df_with_dates['title'] == selected_role].copy()
                    
                    # Aggregate by date
                    daily_counts = role_data.groupby(role_data['date_posted'].dt.date).size().reset_index(name='count')
                    daily_counts.columns = ['date', 'count']
                    daily_counts['date'] = pd.to_datetime(daily_counts['date'])
                    daily_counts = daily_counts.sort_values('date')
                    
                    # Create cumulative sum for trend
                    daily_counts['cumulative'] = daily_counts['count'].cumsum()
                    
                    if len(daily_counts) >= 5:
                        # Prepare data for linear regression
                        daily_counts['days_since_start'] = (daily_counts['date'] - daily_counts['date'].min()).dt.days
                        
                        X = daily_counts['days_since_start'].values.reshape(-1, 1)
                        y = daily_counts['cumulative'].values
                        
                        # Train model
                        model = LinearRegression()
                        model.fit(X, y)
                        
                        # Make predictions for next 30 days
                        last_day = daily_counts['days_since_start'].max()
                        future_days = np.arange(last_day + 1, last_day + 31).reshape(-1, 1)
                        future_predictions = model.predict(future_days)
                        
                        # Create future dates
                        last_date = daily_counts['date'].max()
                        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=30)
                        
                        # Create prediction dataframe
                        predictions_df = pd.DataFrame({
                            'date': future_dates,
                            'cumulative': future_predictions,
                            'type': 'Predicted'
                        })
                        
                        historical_df = daily_counts[['date', 'cumulative']].copy()
                        historical_df['type'] = 'Historical'
                        
                        # Combine historical and predicted
                        combined_df = pd.concat([historical_df, predictions_df], ignore_index=True)
                        
                        # Plot
                        fig3 = px.line(
                            combined_df,
                            x='date',
                            y='cumulative',
                            color='type',
                            title=f"Cumulative Job Postings Forecast: {selected_role}",
                            labels={'date': 'Date', 'cumulative': 'Cumulative Jobs', 'type': 'Data Type'},
                            markers=True
                        )
                        fig3.update_xaxes(range=[datetime(2025, 10, 1), datetime.now()])
                        
                        # Add confidence interval (simple approach)
                        fig3.add_scatter(
                            x=predictions_df['date'],
                            y=future_predictions * 1.1,
                            mode='lines',
                            line=dict(dash='dash', color='rgba(0,100,200,0.3)'),
                            name='Upper Bound (+10%)',
                            showlegend=True
                        )
                        
                        fig3.add_scatter(
                            x=predictions_df['date'],
                            y=future_predictions * 0.9,
                            mode='lines',
                            line=dict(dash='dash', color='rgba(0,100,200,0.3)'),
                            name='Lower Bound (-10%)',
                            showlegend=True
                        )
                        
                        fig3.update_layout(height=500)
                        st.plotly_chart(fig3, use_container_width=True, config=PLOTLY_CONFIG)
                        
                        # Prediction insights
                        current_total = int(daily_counts['cumulative'].iloc[-1])
                        predicted_30d = int(future_predictions[-1])
                        growth = predicted_30d - current_total
                        growth_rate = (growth / current_total * 100) if current_total > 0 else 0
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Current Total", f"{current_total}")
                        with col2:
                            st.metric("Predicted (30d)", f"{predicted_30d}")
                        with col3:
                            st.metric("Expected Growth", f"+{growth}")
                        with col4:
                            st.metric("Growth Rate", f"+{growth_rate:.1f}%")
                        
                        st.info(f"📊 **Model Accuracy (R²):** {model.score(X, y):.3f} | Based on {len(daily_counts)} days of data")
                    else:
                        st.warning(f"Not enough data points for {selected_role} to make predictions (need at least 5 days)")
                
                except Exception as e:
                    st.error(f"Error generating prediction for {selected_role}: {str(e)}")
                    st.info("This role may not have enough data for reliable predictions.")
        else:
            st.warning("Not enough historical data for predictions (need at least 10 jobs with dates)")
    
    except Exception as e:
        st.error(f"Error loading role analytics: {str(e)}")
        st.info("There may be an issue with the data format. Please check that date_posted column contains valid dates.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Built by @MShiyaji**")
st.sidebar.markdown(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
