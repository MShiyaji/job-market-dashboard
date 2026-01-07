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
import re
# Import custom modules
from scraper import JobScraper
from data_processor import DataProcessor

load_dotenv()

def extract_highest_degree(text: str | None) -> str | None:
    if not isinstance(text, str):
        return None
    t = text.lower()
    # Order matters: PhD > Master's > Bachelor's
    if re.search(r"\b(ph\.?d|doctorate|doctoral)\b", t):
        return "PhD"
    if re.search(r"graduate level", t):
        return "Master's"
    if re.search(r"\b(master['’`s]*|m\.sc\.|msc\b|m\.s\.|ms\b|graduate|grad)\b", t):
        return "Master's"
    if re.search(r"\b(bachelor['’`s]*|b\.sc\.|bsc\b|b\.s\.|bs\b)\b", t):
        return "Bachelor's"
    return None

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
        return pd.read_csv(io.BytesIO(data), low_memory=False)
    except (BotoCoreError, ClientError, Exception) as e:
        print(f"S3 load failed for s3://{bucket}/{key}: {e}")
        return None

@st.cache_data(show_spinner=False, ttl=300)
def read_csv_local(path: str) -> pd.DataFrame | None:
    """Read a CSV from local disk; return None on failure."""
    try:
        return pd.read_csv(path, low_memory=False)
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

st.set_page_config(
page_title="Data Science Job Market Dashboard",
page_icon="",
layout="wide",
initial_sidebar_state="expanded"
)

if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'jobs_df' not in st.session_state:
    st.session_state.jobs_df = None
if 'matches_df' not in st.session_state:
    st.session_state.matches_df = None

bg_color = "#0E1117"
secondary_bg = "#262730"
text_color = "#FAFAFA"
border_color = "#4A4A4A"
accent_color = "#FFFFFF"
chart_template = "plotly_dark"
chart_colors = ["#FFFFFF", "#D3D3D3", "#A9A9A9", "#808080", "#696969"]

# Default Plotly config
PLOTLY_CONFIG = {
    "displaylogo": False,
    "scrollzoom": True,
}

# Helper function to style plotly charts with theme
def style_chart(fig, title="", show_grid=True):
    """Apply consistent theme styling to plotly charts with enhanced readability"""
    fig.update_layout(
        template=chart_template,
        paper_bgcolor="rgba(0,0,0,0)",  # Transparent to match dashboard background
        plot_bgcolor="rgba(0,0,0,0)",   # Transparent to match dashboard background
        font=dict(color=text_color, family="Inter, -apple-system, BlinkMacSystemFont, sans-serif", size=12),
        title=dict(
            text=title, 
            font=dict(size=20, color=accent_color, family="Inter, sans-serif"),
            x=0.5,
            xanchor='center',
            pad=dict(b=15)
        ),
        xaxis=dict(
            gridcolor=border_color if show_grid else "rgba(0,0,0,0)",
            linecolor=border_color,
            showgrid=show_grid,
            zeroline=False,
            tickfont=dict(size=11)
        ),
        yaxis=dict(
            gridcolor=border_color if show_grid else "rgba(0,0,0,0)",
            linecolor=border_color,
            showgrid=show_grid,
            zeroline=False,
            tickfont=dict(size=11)
        ),
        colorway=chart_colors,
        margin=dict(l=20, r=20, t=60, b=40),
        hovermode='closest',
        hoverlabel=dict(
            bgcolor=secondary_bg,
            font_size=12,
            font_family="Inter, sans-serif"
        )
    )
    return fig

# Custom CSS with theme support
st.markdown(f"""
<style>
.main-header {{
    font-size: 3rem;
    font-weight: bold;
    color: {accent_color};
    text-align: center;
    margin-bottom: 2rem;
}}
.metric-card {{
    background-color: {secondary_bg};
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
    border: 1px solid {border_color};
}}
.stApp {{
    background-color: {bg_color};
    color: {text_color};
}}
[data-testid="stMetricValue"] {{
    color: {text_color};
}}
[data-testid="stMetricLabel"] {{
    color: {text_color};
}}
</style>
""", unsafe_allow_html=True)
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
st.markdown('<h1 style="text-align: center; font-weight: bold; font-size: 3.5rem; margin-bottom: 2rem;"> Data Science Job Market Dashboard</h1>', 
        unsafe_allow_html=True)
# Sidebar
st.sidebar.markdown("---")
st.sidebar.title("Dashboard Controls")

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
    st.sidebar.markdown("**Built by [Your Name]**")
    st.sidebar.markdown(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    st.stop()

df = st.session_state.jobs_df

# Categorize job titles into broader role categories
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
    elif 'data analyst' in t or 'analyst' in t or 'analytics' in t or 'business intelligence' in t:
        return 'Data Analyst'
    elif 'data engineer' in t or 'etl' in t:
        return 'Data Engineer'
    elif 'software engineer' in t or 'developer' in t:
        return 'Software Engineer'
    else:
        return 'Other'

if 'title' in df.columns:
    df['role_category'] = df['title'].apply(categorize_job_title)

# Derive highest education requirement if not present
if 'education_level' not in df.columns:
    edu_source = None
    for c in ['requirements', 'description', 'job_description', 'full_description', 'posting_text']:
        if c in df.columns:
            edu_source = c
            break
    if edu_source:
        try:
            df['education_level'] = df[edu_source].apply(extract_highest_degree)
        except Exception:
            df['education_level'] = None
    else:
        df['education_level'] = None

# Filters
st.sidebar.header("Filters")

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

# Location filter with metro areas
# Define metro area groupings
METRO_AREAS = {
    'San Francisco Bay Area': ['San Francisco, CA', 'San Jose, CA', 'Oakland, CA', 'Palo Alto, CA', 
                                'Mountain View, CA', 'Sunnyvale, CA', 'Santa Clara, CA', 'Fremont, CA',
                                'Berkeley, CA', 'Redwood City, CA', 'Cupertino, CA', 'Menlo Park, CA'],
    'Dallas-Fort Worth': ['Dallas, TX', 'Fort Worth, TX', 'Arlington, TX', 'Plano, TX', 
                          'Irving, TX', 'Frisco, TX', 'McKinney, TX', 'Denton, TX'],
    'New York Metro': ['New York, NY', 'Brooklyn, NY', 'Queens, NY', 'Manhattan, NY', 
                       'Bronx, NY', 'Jersey City, NJ', 'Newark, NJ', 'Hoboken, NJ'],
    'Los Angeles Metro': ['Los Angeles, CA', 'Long Beach, CA', 'Anaheim, CA', 'Irvine, CA',
                          'Santa Ana, CA', 'Pasadena, CA', 'Glendale, CA', 'Burbank, CA'],
    'Seattle Metro': ['Seattle, WA', 'Bellevue, WA', 'Redmond, WA', 'Tacoma, WA', 
                      'Kirkland, WA', 'Everett, WA', 'Renton, WA'],
    'Boston Metro': ['Boston, MA', 'Cambridge, MA', 'Somerville, MA', 'Quincy, MA', 
                     'Newton, MA', 'Brookline, MA', 'Waltham, MA'],
    'Chicago Metro': ['Chicago, IL', 'Naperville, IL', 'Aurora, IL', 'Evanston, IL',
                      'Schaumburg, IL', 'Joliet, IL', 'Arlington Heights, IL'],
    'Washington DC Metro': ['Washington, DC', 'Arlington, VA', 'Alexandria, VA', 'Bethesda, MD',
                            'Silver Spring, MD', 'Rockville, MD', 'Fairfax, VA']
}

locations = ['All'] + sorted(df['location'].dropna().unique().tolist())
metro_options = ['Individual City'] + list(METRO_AREAS.keys())

location_type = st.sidebar.selectbox("Location Type", metro_options, index=0)

if location_type == 'Individual City':
    selected_location = st.sidebar.selectbox("Location", locations)
    selected_metro = None
else:
    selected_location = None
    selected_metro = location_type

# Job type filter (restricted to four options)
allowed_job_types = ['fulltime', 'internship', 'part-time', 'contract']
job_types = ['All'] + allowed_job_types
selected_job_type = st.sidebar.selectbox("Job Type", job_types, index=0,
    help="Filter by job type. Options are limited to fulltime, internship, part-time, and contract.")

# Remote filter
remote_only = st.sidebar.checkbox("Remote Only")

# Experience filter
exp_levels = ['All', 'New Grad (0 yrs)', 'Entry Level (0-2 yrs)', 'Mid-Level (3-5 yrs)']
selected_exp = st.sidebar.selectbox("Experience Level", exp_levels)

# Role filter
roles = ['All'] + sorted(df['role_category'].dropna().unique().tolist())
selected_role = st.sidebar.selectbox("Job Title", roles)

# Education filter
edu_levels = ["All", "Bachelor's", "Master's", "PhD"]
selected_edu = st.sidebar.selectbox(
    "Education Requirement",
    edu_levels,
    index=0,
    help="Filter by the highest degree mentioned in the posting"
)

# Apply filters
filtered_df = df.copy()

# Apply location filter (individual city or metro area)
if selected_location and selected_location != 'All':
    filtered_df = filtered_df[filtered_df['location'] == selected_location]
elif selected_metro:
    metro_cities = METRO_AREAS[selected_metro]
    filtered_df = filtered_df[filtered_df['location'].isin(metro_cities)]

if selected_job_type != 'All':
    # Use normalized job type for reliable matching
    filtered_df = filtered_df[filtered_df.get('job_type_norm').fillna('') == selected_job_type]
if remote_only:
    filtered_df = filtered_df[filtered_df['is_remote'] == True]

# Apply education level filter
if selected_edu != 'All':
    filtered_df = filtered_df[filtered_df.get('education_level').fillna('') == selected_edu]

# Apply experience level filter (if available)
if 'years_experience_required' in filtered_df.columns and selected_exp != 'All':
    years = pd.to_numeric(filtered_df['years_experience_required'], errors='coerce')
    if selected_exp == 'New Grad (0 yrs)':
        filtered_df = filtered_df[years == 0]
    elif selected_exp == 'Entry Level (0-2 yrs)':
        filtered_df = filtered_df[years <= 2]
    elif selected_exp == 'Mid-Level (3-5 yrs)':
        filtered_df = filtered_df[(years >= 3) & (years <= 5)]

# Apply role filter
if selected_role != 'All':
    filtered_df = filtered_df[filtered_df['role_category'] == selected_role]

# Key Metrics
st.header(" Key Metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_jobs = len(df)
    filtered_jobs = len(filtered_df)
    pct_of_total = (filtered_jobs / total_jobs * 100) if total_jobs > 0 else 0
    st.metric("% of Total Jobs", f"{pct_of_total:.1f}%")
with col2:
    st.metric("Unique Companies", filtered_df['company'].nunique())
with col3:
    avg_salary = filtered_df['average_salary'].mean()
    st.metric("Avg Salary", f"${avg_salary:,.0f}" if pd.notna(avg_salary) else "N/A")
with col4:
    remote_pct = (filtered_df['is_remote'].sum() / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
    st.metric("Remote %", f"{remote_pct:.1f}%")

# Tabs for different views
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Market Trends",
    "Salary Analysis",
    "Skills Demand",
    "AI Insights",
    "Role Analytics",
    "Compare Filters"
])

with tab1:
    st.subheader("Job Market Trends")
    
    # Jobs by location
    col1, col2 = st.columns(2)
    
    with col1:
        location_counts = filtered_df['location'].value_counts().head(10)
        total_jobs = len(filtered_df)
        location_pct = (location_counts / total_jobs * 100).round(2)
        
        # Create gradient colors from dark to light based on values
        colors = [f'rgba({int(26 + (i/len(location_pct)) * 229)}, {int(26 + (i/len(location_pct)) * 229)}, {int(26 + (i/len(location_pct)) * 229)}, 0.9)' 
                  for i in range(len(location_pct))]
        
        # Calculate text colors based on value (dark text for light bars, light text for dark bars)
        text_colors = ['#1A1A1A' if val > location_pct.max() * 0.5 else '#FFFFFF' for val in location_pct.values]
        
        fig = go.Figure(data=[go.Bar(
            x=location_pct.values,
            y=location_pct.index,
            orientation='h',
            marker=dict(
                color=location_pct.values,
                colorscale=[[0, '#1A1A1A'], [0.5, '#808080'], [1, '#FFFFFF']],
                line=dict(width=0),
                cornerradius=5
            ),
            hovertemplate='<b>%{y}</b><br>%{x:.1f}% of jobs<extra></extra>',
            text=[f'{val:.1f}%' for val in location_pct.values],
            textposition='inside',
            textfont=dict(size=11, family='Inter', color=text_colors)
        )])
        
        fig = style_chart(fig, "Top 10 Locations", show_grid=False)
        fig.update_layout(
            height=450,
            yaxis=dict(categoryorder='total ascending'),
            xaxis=dict(ticksuffix='%', showgrid=False)
        )
        st.plotly_chart(fig, width='stretch', config=PLOTLY_CONFIG)
    
    with col2:
        company_counts = filtered_df['company'].value_counts().head(10)
        total_jobs = len(filtered_df)
        company_pct = (company_counts / total_jobs * 100).round(2)
        
        # Calculate text colors based on value
        text_colors_comp = ['#1A1A1A' if val > company_pct.max() * 0.5 else '#FFFFFF' for val in company_pct.values]
        
        fig = go.Figure(data=[go.Bar(
            x=company_pct.values,
            y=company_pct.index,
            orientation='h',
            marker=dict(
                color=company_pct.values,
                colorscale=[[0, '#1A1A1A'], [0.5, '#808080'], [1, '#FFFFFF']],
                line=dict(width=0),
                cornerradius=5
            ),
            hovertemplate='<b>%{y}</b><br>%{x:.1f}% of jobs<extra></extra>',
            text=[f'{val:.1f}%' for val in company_pct.values],
            textposition='inside',
            textfont=dict(size=11, family='Inter', color=text_colors_comp)
        )])
        
        fig = style_chart(fig, "Top 10 Hiring Companies", show_grid=False)
        fig.update_layout(
            height=450,
            yaxis=dict(categoryorder='total ascending'),
            xaxis=dict(ticksuffix='%', showgrid=False)
        )
        st.plotly_chart(fig, width='stretch', config=PLOTLY_CONFIG)

    # Time Series Analysis
    st.markdown("---")
    if 'date_posted' in filtered_df.columns:
        trend_df = filtered_df.copy()
        trend_df['date_posted'] = pd.to_datetime(trend_df['date_posted'], errors='coerce')
        trend_df = trend_df.dropna(subset=['date_posted'])
        
        # 1. Daily Job Volume Chart - REMOVED

        # 2. Role Trends Over Time
        # Group by date and role
        daily_roles = trend_df.groupby([trend_df['date_posted'].dt.date, 'role_category']).size().reset_index(name='count')
        
        # Ensure all roles exist for all dates (fill missing with 0) to allow continuous lines
        if not daily_roles.empty:
            all_dates = daily_roles['date_posted'].unique()
            all_roles = daily_roles['role_category'].unique()
            # Create Cartesian product of all dates and roles
            idx = pd.MultiIndex.from_product([all_dates, all_roles], names=['date_posted', 'role_category'])
            daily_roles = daily_roles.set_index(['date_posted', 'role_category']).reindex(idx, fill_value=0).reset_index()

        # Calculate percentages (denominator includes 'Other' and all categories)
        daily_totals = daily_roles.groupby('date_posted')['count'].transform('sum')
        daily_roles['percentage'] = (daily_roles['count'] / daily_totals) * 100
        daily_roles['percentage'] = daily_roles['percentage'].fillna(0)
        
        vibrant_colors = [
            "#FF595E", # Red
            "#FFCA3A", # Yellow
            "#8AC926", # Green
            "#1982C4", # Blue
            "#6A4C93", # Purple
            "#F15BB5", # Pink
            "#00BBF9", # Light Blue
            "#00F5D4"  # Teal
        ]
        
        if not daily_roles.empty:
            fig_trend = go.Figure()
            
            roles = sorted(daily_roles['role_category'].unique())
            for i, role in enumerate(roles):
                role_data = daily_roles[daily_roles['role_category'] == role]
                color = vibrant_colors[i % len(vibrant_colors)]
                
                fig_trend.add_trace(go.Scatter(
                    x=role_data['date_posted'],
                    y=role_data['percentage'],
                    mode='lines',
                    name=role,
                    line=dict(width=2, color=color),
                    hovertemplate='%{y:.1f}%<extra></extra>'
                ))
            
            fig_trend = style_chart(fig_trend, "Job Roles Over Time (Daily Distribution)")
            fig_trend.update_layout(
                hovermode='x unified',
                yaxis=dict(ticksuffix='%', title="Percentage of Daily Jobs"),
                xaxis=dict(range=[datetime(2025, 11, 1), datetime.now()])
            )
            st.plotly_chart(fig_trend, width='stretch', config=PLOTLY_CONFIG)
    else:
        st.info("Date information not available for trend analysis.")

with tab2:
    st.subheader("Salary Analysis")
    
    # Salary distribution
    salary_data = filtered_df['average_salary'].dropna()
    
    if len(salary_data) > 0:
        col1, col2 = st.columns(2)

        with col1:
            fig = go.Figure(data=[go.Histogram(
                x=salary_data,
                nbinsx=30,
                marker=dict(
                    color='#FFFFFF',
                    line=dict(width=0.5, color=border_color)
                ),
                hovertemplate='$%{x:,.0f}<br>Count: %{y}<extra></extra>'
            )])
            
            fig = style_chart(fig, "Salary Distribution", show_grid=False)
            fig.update_xaxes(tickprefix='$', tickformat=',.0f')
            fig.update_layout(
                xaxis_title='Salary ($)',
                yaxis_title='Frequency',
                bargap=0.05
            )
            st.plotly_chart(fig, width='stretch', config=PLOTLY_CONFIG)

        with col2:
            # Salary by role category (sorted highest to lowest)
            salary_by_role = (
                filtered_df.groupby('role_category')['average_salary']
                .mean()
                .sort_values(ascending=False)
            )
            
            # Calculate text colors - light text for dark bars (low values), dark text for light bars (high values)
            text_colors_sal = ['#FFFFFF' if val < salary_by_role.max() * 0.5 else '#1A1A1A' for val in salary_by_role.values]
            
            fig = go.Figure(data=[go.Bar(
                x=salary_by_role.values,
                y=salary_by_role.index,
                orientation='h',
                marker=dict(
                    color=salary_by_role.values,
                    colorscale=[[0, '#1A1A1A'], [0.5, '#808080'], [1, '#FFFFFF']],
                    line=dict(width=0),
                    cornerradius=5,
                    showscale=False
                ),
                hovertemplate='<b>%{y}</b><br>$%{x:,.0f}<extra></extra>',
                text=[f'${val:,.0f}' for val in salary_by_role.values],
                textposition='inside',
                textfont=dict(size=11, family='Inter', color=text_colors_sal)
            )])
            
            fig = style_chart(fig, "Average Salary by Role", show_grid=False)
            fig.update_layout(
                yaxis=dict(categoryorder='total ascending'),
                xaxis=dict(tickprefix='$', tickformat=',.0f', showgrid=False)
            )
            st.plotly_chart(fig, width='stretch', config=PLOTLY_CONFIG)
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

        # Calculate text colors based on skill percentage
        skill_vals = list(top_skills.values())
        max_skill = max(skill_vals)
        text_colors_skills = ['#1A1A1A' if val > max_skill * 0.5 else '#FFFFFF' for val in skill_vals]
        
        fig = go.Figure(data=[go.Bar(
            x=skill_vals,
            y=list(top_skills.keys()),
            orientation='h',
            marker=dict(
                color=skill_vals,
                colorscale=[[0, '#1A1A1A'], [0.5, '#808080'], [1, '#FFFFFF']],
                line=dict(width=0),
                cornerradius=5,
                showscale=False
            ),
            hovertemplate='<b>%{y}</b><br>%{x:.1f}% of jobs<extra></extra>',
            text=[f'{val:.1f}%' for val in skill_vals],
            textposition='inside',
            textfont=dict(size=11, family='Inter', color=text_colors_skills)
        )])
        
        title_text = f"Top {top_n} Skills - {selected_role}" if selected_role != 'All' else f"Top {top_n} In-Demand Skills"
        fig = style_chart(fig, title_text, show_grid=False)
        fig.update_layout(
            height=550,
            showlegend=False,
            yaxis=dict(categoryorder='total ascending'),
            xaxis=dict(ticksuffix='%', showgrid=False)
        )
        st.plotly_chart(fig, width='stretch', config=PLOTLY_CONFIG)

    else:
        st.warning("No skill data available for this selection")

    # Education requirement mix over time
    st.markdown("### 2) Education Requirements Trend")
    if 'date_posted' in filtered_df.columns and 'education_level' in filtered_df.columns:
        edu_trend_df = filtered_df.copy()
        edu_trend_df['date_posted'] = pd.to_datetime(edu_trend_df['date_posted'], errors='coerce')
        edu_trend_df = edu_trend_df.dropna(subset=['date_posted', 'education_level'])

        # Focus trend view on the current market window
        start_date = datetime(2025, 10, 1)
        edu_trend_df = edu_trend_df[edu_trend_df['date_posted'] >= start_date]

        if len(edu_trend_df) > 0:
            edu_trend_df['date'] = edu_trend_df['date_posted'].dt.date
            daily_counts = (
                edu_trend_df
                .groupby(['date', 'education_level'])
                .size()
                .reset_index(name='count')
            )
            daily_totals = (
                edu_trend_df
                .groupby('date')
                .size()
                .reset_index(name='total')
            )
            # Sort by date for cumulative calculations
            daily_counts = daily_counts.sort_values('date')
            daily_totals = daily_totals.sort_values('date')
            
            # Compute cumulative counts per education level
            daily_counts['cum_count'] = daily_counts.groupby('education_level')['count'].cumsum()
            
            # Compute cumulative total jobs
            daily_totals['cum_total'] = daily_totals['total'].cumsum()
            
            # Merge cumulative data
            edu_trend = daily_counts.merge(daily_totals[['date', 'cum_total']], on='date')
            edu_trend['percentage'] = edu_trend['cum_count'] / edu_trend['cum_total'] * 100
            edu_trend['date'] = pd.to_datetime(edu_trend['date'])

            # Create figure with go.Scatter for each education level
            fig_edu = go.Figure()
            
            edu_levels = ["Bachelor's", "Master's", "PhD"]
            colors_map = {edu_levels[i]: chart_colors[i] for i in range(len(edu_levels))}
            
            for edu_level in edu_levels:
                edu_data = edu_trend[edu_trend['education_level'] == edu_level]
                fig_edu.add_trace(go.Scatter(
                    x=edu_data['date'],
                    y=edu_data['percentage'],
                    mode='lines+markers',
                    name=edu_level,
                    line=dict(width=3, color=colors_map.get(edu_level, chart_colors[0])),
                    marker=dict(size=7, line=dict(width=1, color=bg_color)),
                    hovertemplate=f'<b>{edu_level}</b><br>%{{x|%b %d}}<br>%{{y:.1f}}%<extra></extra>',
                    fill='tonexty' if edu_level != "Bachelor's" else None,
                    fillcolor=f'rgba({int(colors_map.get(edu_level, chart_colors[0])[1:3], 16)}, {int(colors_map.get(edu_level, chart_colors[0])[3:5], 16)}, {int(colors_map.get(edu_level, chart_colors[0])[5:7], 16)}, 0.1)'
                ))
            
            fig_edu = style_chart(fig_edu, 'Cumulative Education Requirement Mix Over Time')
            fig_edu.update_layout(
                height=450,
                legend=dict(
                    title='Highest Degree',
                    orientation='h',
                    yanchor='bottom',
                    y=1.02,
                    xanchor='right',
                    x=1
                )
            )
            fig_edu.update_yaxes(range=[0, 100], ticksuffix='%')
            st.plotly_chart(fig_edu, width='stretch', config=PLOTLY_CONFIG)
        else:
            st.info("No education data available to plot trends since Oct 2025.")
    else:
        st.info("Education or date data missing; unable to show education trend chart.")

    # Hot skills: skills increasing in frequency the most
    st.markdown("### 3) Hot Skills (Rising Fast)")
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

                    # Visualize top 5 trending skills over time
                    top_skills_names = trends_df['Skill'].head(5).tolist()
                    plot_cols = [f'skill_{n}' for n in top_skills_names]
                    melted = daily_pct[plot_cols].reset_index().melt(id_vars='date', var_name='skill', value_name='percentage')
                    melted['skill'] = melted['skill'].str.replace('skill_', '', regex=False)
                    
                    # Create figure with go.Scatter
                    fig_trend = go.Figure()
                    for i, skill in enumerate(top_skills_names):
                        skill_data = melted[melted['skill'] == skill]
                        fig_trend.add_trace(go.Scatter(
                            x=skill_data['date'],
                            y=skill_data['percentage'],
                            mode='lines+markers',
                            name=skill,
                            line=dict(width=3, color=chart_colors[i % len(chart_colors)]),
                            marker=dict(size=7, line=dict(width=1, color=bg_color)),
                            hovertemplate=f'<b>{skill}</b><br>%{{x|%b %d}}<br>%{{y:.1f}}%<extra></extra>',
                            fill='tonexty' if i > 0 else None,
                            fillcolor=f'rgba({int(chart_colors[i % len(chart_colors)][1:3], 16)}, {int(chart_colors[i % len(chart_colors)][3:5], 16)}, {int(chart_colors[i % len(chart_colors)][5:7], 16)}, 0.1)'
                        ))
                    
                    fig_trend = style_chart(fig_trend, 'Top Trending Skills - % of Jobs (Last 30 Days)')
                    fig_trend.update_layout(
                        height=450,
                        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
                    )
                    fig_trend.update_xaxes(range=[datetime(2025, 10, 1), datetime.now()])
                    fig_trend.update_yaxes(ticksuffix='%')
                    st.plotly_chart(fig_trend, width='stretch', config=PLOTLY_CONFIG)
                else:
                    st.info("Not enough recent data to identify trending skills.")
            else:
                st.info("No skill columns found in the recent data window.")
        else:
            st.info("No valid dates available to compute trends.")
    else:
        st.info("Dataset lacks 'date_posted' column; cannot compute trends.")

    # Falling skills: skills declining in frequency the most
    st.markdown("### 4) Cooling Skills (Declining Fast)")
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

                    # Visualize top 5 declining skills over time
                    top_declining_names = declining_df['Skill'].head(5).tolist()
                    plot_cols_decline = [f'skill_{n}' for n in top_declining_names]
                    melted_decline = daily_pct[plot_cols_decline].reset_index().melt(id_vars='date', var_name='skill', value_name='percentage')
                    melted_decline['skill'] = melted_decline['skill'].str.replace('skill_', '', regex=False)
                    
                    # Create figure with go.Scatter
                    fig_decline = go.Figure()
                    declining_skills = melted_decline['skill'].unique()
                    for i, skill in enumerate(declining_skills):
                        skill_data = melted_decline[melted_decline['skill'] == skill]
                        fig_decline.add_trace(go.Scatter(
                            x=skill_data['date'],
                            y=skill_data['percentage'],
                            mode='lines+markers',
                            name=skill,
                            line=dict(width=3, color=chart_colors[i % len(chart_colors)]),
                            marker=dict(size=7, line=dict(width=1, color=bg_color)),
                            hovertemplate=f'<b>{skill}</b><br>%{{x|%b %d}}<br>%{{y:.1f}}%<extra></extra>',
                            fill='tonexty' if i > 0 else None,
                            fillcolor=f'rgba({int(chart_colors[i % len(chart_colors)][1:3], 16)}, {int(chart_colors[i % len(chart_colors)][3:5], 16)}, {int(chart_colors[i % len(chart_colors)][5:7], 16)}, 0.1)'
                        ))
                    
                    fig_decline = style_chart(fig_decline, 'Top Declining Skills - % of Jobs (Last 30 Days)')
                    fig_decline.update_layout(
                        height=450,
                        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
                    )
                    fig_decline.update_xaxes(range=[datetime(2025, 10, 1), datetime.now()])
                    fig_decline.update_yaxes(ticksuffix='%')
                    st.plotly_chart(fig_decline, width='stretch', config=PLOTLY_CONFIG)
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
    
    # Resume upload and analysis section
    st.markdown("### Upload Your Resume")
    uploaded_resume = st.file_uploader(
        "Upload Your Resume (PDF)",
        type=['pdf'],
        help="Upload your resume to get personalized job match insights"
    )
    
    if uploaded_resume:
        # Save uploaded file
        with open('data/resume.pdf', 'wb') as f:
            f.write(uploaded_resume.getbuffer())
        st.success("Resume uploaded!")
    
    # API Key Management
    # Force user to provide their own key
    if 'user_gemini_key' not in st.session_state:
        st.session_state.user_gemini_key = ""
    
    api_key = st.text_input(
        "Enter your Gemini API Key", 
        type="password",
        value=st.session_state.user_gemini_key,
        help="Your key is used only for this session and not stored permanently. Get one at aistudio.google.com"
    )

    with st.expander("🔑 How to get a Gemini API Key?"):
        st.markdown("""
        1. Select or create a **Google Cloud Project** in the [Google Cloud Console](https://console.cloud.google.com/)
        2. Make sure **Billing** is enabled for your project (required for usage beyond free quotas)
        3. Go to [Google AI Studio](https://aistudio.google.com/app/apikey) to create an API key for your project
        4. Copy the key and paste it above
        5. *Recommended*: Restrict your key in [Google Cloud Console > Credentials](https://console.cloud.google.com/apis/credentials) for security
        
        *Note: The API has a free usage tier, but billing may be required for high volume usage. Check current quotas on the pricing page.*
        """)
    
    if api_key:
        st.session_state.user_gemini_key = api_key
    else:
        st.warning("Please enter a Gemini API Key to use the AI features.")

    if st.button("Analyze Resume Fit", type="primary", disabled=not api_key):
        if not st.session_state.data_loaded:
            st.error("No job data loaded! Please load or scrape job data first.")
        elif not os.path.exists('data/resume.pdf'):
            st.error("Please upload a resume first!")
            st.info("Use the file uploader above to upload your resume (PDF format)")
        elif filtered_df.empty:
            st.error("No jobs match your current filters! Please adjust filters to include some jobs.")
        else:
            st.info(f"Analyzing 10 jobs from your filtered selection ({len(filtered_df)} total). This will take ~5-10 seconds.")
            with st.spinner("Analyzing your fit for jobs using AI..."):
                try:
                    from resume_analyzer import ResumeAnalyzer
                    analyzer = ResumeAnalyzer('data/resume.pdf', api_key=api_key)
                    matches = analyzer.analyze_job_market_fit(
                        filtered_df,
                        sample_size=10  # Reduced to 10 to stay within free tier token limits (1M TPM)
                    )
                    
                    # Check if analysis actually produced results
                    if matches is None or matches.empty:
                        st.error("Analysis failed - no matches generated")
                        st.info("This could be due to: API rate limits, invalid API key, or malformed AI responses. Check the console logs for details.")
                    elif 'match_percentage' not in matches.columns:
                        st.error("Analysis produced invalid data format")
                        st.info("The AI responses are not in the expected format. Try again or check your GEMINI_API_KEY.")
                    else:
                        st.session_state.matches_df = matches
                        matches.to_csv('data/job_matches.csv', index=False)
                        st.success(f"Analysis complete! Matched {len(matches)} jobs.")
                        
                except ValueError as e:
                    st.error(f"Configuration error: {e}")
                    st.info("Make sure GEMINI_API_KEY is set in your .env file")
                except Exception as e:
                    st.error(f"Resume analyzer error: {e}")
                    st.info("Check that all dependencies are installed (PyMuPDF, google-generativeai)")
    
    st.markdown("---")
    
    if st.session_state.matches_df is not None:
        matches_df = st.session_state.matches_df
        
        if matches_df.empty or 'match_percentage' not in matches_df.columns:
            st.warning("No valid match data available. The resume analysis may have failed.")
            st.info("Try re-uploading your resume and running the analysis again.")
        else:
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
            fig = go.Figure(data=[go.Histogram(
                x=matches_df['match_percentage'],
                nbinsx=20,
                marker=dict(
                    color=matches_df['match_percentage'],
                    colorscale=[[0, '#1A1A1A'], [0.5, '#808080'], [1, '#FFFFFF']],
                    line=dict(width=0.5, color=border_color)
                ),
                hovertemplate='%{x:.0f}%<br>Jobs: %{y}<extra></extra>'
            )])
            
            fig = style_chart(fig, "Distribution of Match Scores", show_grid=False)
            fig.update_xaxes(ticksuffix='%')
            fig.update_layout(
                xaxis_title='Match Percentage',
                yaxis_title='Number of Jobs',
                bargap=0.05
            )
            st.plotly_chart(fig, width='stretch', config=PLOTLY_CONFIG)
            
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
                
                # Calculate text colors based on frequency
                missing_vals = list(top_missing.values())
                max_missing = max(missing_vals)
                text_colors_missing = ['#1A1A1A' if val > max_missing * 0.5 else '#FFFFFF' for val in missing_vals]
                
                fig = go.Figure(data=[go.Bar(
                    x=missing_vals,
                    y=list(top_missing.keys()),
                    orientation='h',
                    marker=dict(
                        color=missing_vals,
                        colorscale=[[0, '#1A1A1A'], [0.5, '#808080'], [1, '#FFFFFF']],
                        line=dict(width=0),
                        cornerradius=5,
                        showscale=False
                    ),
                    hovertemplate='<b>%{y}</b><br>Frequency: %{x}<extra></extra>',
                    text=[f'{val}' for val in missing_vals],
                    textposition='inside',
                    textfont=dict(size=11, family='Inter', color=text_colors_missing)
                )])
                
                fig = style_chart(fig, "Skills You Should Develop", show_grid=False)
                fig.update_layout(
                    yaxis=dict(categoryorder='total ascending'),
                    showlegend=False,
                    xaxis=dict(showgrid=False)
                )
                st.plotly_chart(fig, width='stretch', config=PLOTLY_CONFIG)
            
            # Generate personalized insights
            if st.button("Generate Personalized Career Advice", disabled=not api_key):
                with st.spinner("Generating insights with Gemini AI..."):
                    try:
                        from resume_analyzer import ResumeAnalyzer
                        analyzer = ResumeAnalyzer('data/resume.pdf', api_key=api_key)
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
    st.subheader("Role Analytics & Predictions")
    
    # Ensure we have date data
    if 'date_posted' not in filtered_df.columns:
        st.warning("No date information available in the dataset. Cannot generate time-based analytics.")
        st.info("Make sure your scraped data includes 'date_posted' field.")
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
        st.markdown("### 1. Role Distribution (Last 30 Days)")
        
        if len(df_last_30) > 0:
            role_counts = df_last_30['title'].value_counts().head(15)
            total_jobs = len(df_last_30)
            role_pct = (role_counts / total_jobs * 100).round(2)
            
            # Calculate text colors based on role percentage
            text_colors_role = ['#1A1A1A' if val > role_pct.max() * 0.5 else '#FFFFFF' for val in role_pct.values]
            
            fig1 = go.Figure(data=[go.Bar(
                x=role_pct.values,
                y=role_pct.index,
                orientation='h',
                marker=dict(
                    color=role_pct.values,
                    colorscale=[[0, '#1A1A1A'], [0.5, '#808080'], [1, '#FFFFFF']],
                    line=dict(width=0),
                    cornerradius=5,
                    showscale=False
                ),
                hovertemplate='<b>%{y}</b><br>%{x:.1f}%<extra></extra>',
                text=[f'{val:.1f}%' for val in role_pct.values],
                textposition='inside',
                textfont=dict(size=11, family='Inter', color=text_colors_role)
            )])
            
            title_text = f"Top 15 Role Titles (Last 30 Days)"
            fig1 = style_chart(fig1, title_text, show_grid=False)
            fig1.update_layout(
                height=500,
                showlegend=False,
                yaxis=dict(categoryorder='total ascending'),
                xaxis=dict(ticksuffix='%', showgrid=False)
            )
            st.plotly_chart(fig1, width='stretch', config=PLOTLY_CONFIG)
            
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
        st.markdown("### 2. Daily Role Growth Rate")
        
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
            
            # Create figure with go.Scatter
            fig2 = go.Figure()
            for i, role in enumerate(top_roles):
                role_data = daily_role_counts[daily_role_counts['title'] == role]
                fig2.add_trace(go.Scatter(
                    x=role_data['date'],
                    y=role_data['percentage'],
                    mode='lines+markers',
                    name=role,
                    line=dict(width=3, color=chart_colors[i % len(chart_colors)]),
                    marker=dict(size=7, line=dict(width=1, color=bg_color)),
                    hovertemplate=f'<b>{role}</b><br>%{{x|%b %d}}<br>%{{y:.1f}}%<extra></extra>'
                ))
            
            fig2 = style_chart(fig2, "Daily Role Distribution - % of Jobs (Top 8 Roles)")
            fig2.update_layout(
                height=500,
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
            )
            fig2.update_xaxes(range=[datetime(2025, 10, 1), datetime.now()])
            fig2.update_yaxes(ticksuffix='%', autorange=True)
            st.plotly_chart(fig2, width='stretch', config=PLOTLY_CONFIG)
            
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
                st.dataframe(growth_df, width='stretch')
        else:
            st.warning("No date data available for growth analysis")
        
        # Chart 3: Time-series prediction
        st.markdown("### 3. Future Role Demand Prediction")
        
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
                        
                        # Plot with go.Scatter
                        fig3 = go.Figure()
                        
                        # Historical data
                        hist_data = combined_df[combined_df['type'] == 'Historical']
                        fig3.add_trace(go.Scatter(
                            x=hist_data['date'],
                            y=hist_data['cumulative'],
                            mode='lines+markers',
                            name='Historical',
                            line=dict(width=3, color=accent_color),
                            marker=dict(size=7, line=dict(width=1, color=bg_color)),
                            hovertemplate='<b>Historical</b><br>%{x|%b %d}<br>Jobs: %{y:.0f}<extra></extra>'
                        ))
                        
                        # Predicted data
                        pred_data = combined_df[combined_df['type'] == 'Predicted']
                        fig3.add_trace(go.Scatter(
                            x=pred_data['date'],
                            y=pred_data['cumulative'],
                            mode='lines+markers',
                            name='Predicted',
                            line=dict(width=3, color=border_color, dash='dash'),
                            marker=dict(size=7, line=dict(width=1, color=bg_color)),
                            hovertemplate='<b>Predicted</b><br>%{x|%b %d}<br>Jobs: %{y:.0f}<extra></extra>'
                        ))
                        
                        # Add confidence interval (simple approach)
                        bound_color = 'rgba(128,128,128,0.3)'
                        fig3.add_trace(go.Scatter(
                            x=predictions_df['date'],
                            y=future_predictions * 1.1,
                            mode='lines',
                            line=dict(dash='dot', color=bound_color, width=2),
                            name='Upper Bound (+10%)',
                            showlegend=True,
                            hovertemplate='%{x|%b %d}<br>Jobs: %{y:.0f}<extra></extra>'
                        ))
                        
                        fig3.add_trace(go.Scatter(
                            x=predictions_df['date'],
                            y=future_predictions * 0.9,
                            mode='lines',
                            line=dict(dash='dot', color=bound_color, width=2),
                            name='Lower Bound (-10%)',
                            fill='tonexty',
                            fillcolor=bound_color,
                            showlegend=True,
                            hovertemplate='%{x|%b %d}<br>Jobs: %{y:.0f}<extra></extra>'
                        ))
                        
                        title_text = f"Cumulative Job Postings Forecast: {selected_role}"
                        fig3 = style_chart(fig3, title_text)
                        fig3.update_layout(
                            height=500,
                            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
                        )
                        fig3.update_xaxes(range=[datetime(2025, 10, 1), datetime.now() + timedelta(days=30)])
                        
                        st.plotly_chart(fig3, width='stretch', config=PLOTLY_CONFIG)
                        
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
                        
                        st.info(f"**Model Accuracy (R²):** {model.score(X, y):.3f} | Based on {len(daily_counts)} days of data")
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

# Tab 6: Compare Filters
with tab6:
    st.header("Compare Filter Sets")
    st.markdown("Compare job market metrics between two different filter configurations")
    
    col_a, col_b = st.columns(2)
    
    # Helper function to apply filters
    def apply_comparison_filters(base_df, location_val, metro_val, job_type_val, remote_val, exp_val, role_val, edu_val):
        temp_df = base_df.copy()
        
        # Location filter
        if location_val and location_val != 'All':
            temp_df = temp_df[temp_df['location'] == location_val]
        elif metro_val and metro_val != 'Individual City':
            metro_cities = METRO_AREAS[metro_val]
            temp_df = temp_df[temp_df['location'].isin(metro_cities)]
        
        # Job type filter
        if job_type_val != 'All':
            temp_df = temp_df[temp_df.get('job_type_norm').fillna('') == job_type_val]
        
        # Remote filter
        if remote_val:
            temp_df = temp_df[temp_df['is_remote'] == True]
        
        # Education filter
        if edu_val != 'All':
            temp_df = temp_df[temp_df.get('education_level').fillna('') == edu_val]
        
        # Experience filter
        if 'years_experience_required' in temp_df.columns and exp_val != 'All':
            years = pd.to_numeric(temp_df['years_experience_required'], errors='coerce')
            if exp_val == 'New Grad (0 yrs)':
                temp_df = temp_df[years == 0]
            elif exp_val == 'Entry Level (0-2 yrs)':
                temp_df = temp_df[years <= 2]
            elif exp_val == 'Mid-Level (3-5 yrs)':
                temp_df = temp_df[(years >= 3) & (years <= 5)]
        
        # Role filter
        if role_val != 'All':
            temp_df = temp_df[temp_df['role_category'] == role_val]
        
        return temp_df
    
    # Filter Set A
    with col_a:
        st.subheader("Filter Set A")
        
        loc_type_a = st.selectbox("Location Type (A)", metro_options, index=0, key='loc_type_a')
        if loc_type_a == 'Individual City':
            loc_a = st.selectbox("Location (A)", locations, key='loc_a')
            metro_a = None
        else:
            loc_a = None
            metro_a = loc_type_a
        
        job_type_a = st.selectbox("Job Type (A)", job_types, index=0, key='job_type_a')
        remote_a = st.checkbox("Remote Only (A)", key='remote_a')
        exp_a = st.selectbox("Experience Level (A)", exp_levels, key='exp_a')
        role_a = st.selectbox("Job Title (A)", roles, key='role_a')
        edu_a = st.selectbox("Education (A)", edu_levels, index=0, key='edu_a')
    
    # Filter Set B
    with col_b:
        st.subheader("Filter Set B")
        
        loc_type_b = st.selectbox("Location Type (B)", metro_options, index=0, key='loc_type_b')
        if loc_type_b == 'Individual City':
            loc_b = st.selectbox("Location (B)", locations, key='loc_b')
            metro_b = None
        else:
            loc_b = None
            metro_b = loc_type_b
        
        job_type_b = st.selectbox("Job Type (B)", job_types, index=0, key='job_type_b')
        remote_b = st.checkbox("Remote Only (B)", key='remote_b')
        exp_b = st.selectbox("Experience Level (B)", exp_levels, key='exp_b')
        role_b = st.selectbox("Job Title (B)", roles, key='role_b')
        edu_b = st.selectbox("Education (B)", edu_levels, index=0, key='edu_b')
    
    # Apply filters
    df_a = apply_comparison_filters(df, loc_a, metro_a, job_type_a, remote_a, exp_a, role_a, edu_a)
    df_b = apply_comparison_filters(df, loc_b, metro_b, job_type_b, remote_b, exp_b, role_b, edu_b)
    
    st.markdown("---")
    st.subheader("Comparison Metrics")
    
    # Key metrics comparison
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**Total Jobs**")
        st.metric("Set A", len(df_a), delta=None)
        st.metric("Set B", len(df_b), delta=int(len(df_b) - len(df_a)))
    
    with col2:
        st.markdown("**Unique Companies**")
        st.metric("Set A", df_a['company'].nunique())
        st.metric("Set B", df_b['company'].nunique(), delta=int(df_b['company'].nunique() - df_a['company'].nunique()))
    
    with col3:
        st.markdown("**Avg Salary**")
        avg_sal_a = df_a['average_salary'].mean()
        avg_sal_b = df_b['average_salary'].mean()
        st.metric("Set A", f"${avg_sal_a:,.0f}" if pd.notna(avg_sal_a) else "N/A")
        if pd.notna(avg_sal_a) and pd.notna(avg_sal_b):
            st.metric("Set B", f"${avg_sal_b:,.0f}", delta=f"${avg_sal_b - avg_sal_a:,.0f}")
        else:
            st.metric("Set B", f"${avg_sal_b:,.0f}" if pd.notna(avg_sal_b) else "N/A")
    
    with col4:
        st.markdown("**Remote %**")
        remote_pct_a = (df_a['is_remote'].sum() / len(df_a) * 100) if len(df_a) > 0 else 0
        remote_pct_b = (df_b['is_remote'].sum() / len(df_b) * 100) if len(df_b) > 0 else 0
        st.metric("Set A", f"{remote_pct_a:.1f}%")
        st.metric("Set B", f"{remote_pct_b:.1f}%", delta=f"{remote_pct_b - remote_pct_a:.1f}%")
    
    # Side-by-side comparisons
    st.markdown("---")
    st.subheader("Visual Comparisons")
    
    # Salary distribution comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Set A: Salary Distribution**")
        if len(df_a) > 0 and df_a['average_salary'].notna().sum() > 0:
            salary_data_a = df_a['average_salary'].dropna()
            fig_a = go.Figure(data=[go.Histogram(
                x=salary_data_a,
                nbinsx=20,
                marker=dict(
                    color=salary_data_a,
                    colorscale=[[0, '#FFFFFF'], [0.5, '#808080'], [1, '#1A1A1A']],
                    line=dict(width=0.5, color=border_color)
                ),
                hovertemplate='$%{x:,.0f}<br>Count: %{y}<extra></extra>'
            )])
            fig_a = style_chart(fig_a, "", show_grid=False)
            fig_a.update_xaxes(tickprefix='$', tickformat=',.0f')
            fig_a.update_layout(height=300, bargap=0.05)
            st.plotly_chart(fig_a, width='stretch', config=PLOTLY_CONFIG, key='salary_hist_a')
        else:
            st.info("No salary data available for Set A")
    
    with col2:
        st.markdown("**Set B: Salary Distribution**")
        if len(df_b) > 0 and df_b['average_salary'].notna().sum() > 0:
            salary_data_b = df_b['average_salary'].dropna()
            fig_b = go.Figure(data=[go.Histogram(
                x=salary_data_b,
                nbinsx=20,
                marker=dict(
                    color=salary_data_b,
                    colorscale=[[0, '#FFFFFF'], [0.5, '#808080'], [1, '#1A1A1A']],
                    line=dict(width=0.5, color=border_color)
                ),
                hovertemplate='$%{x:,.0f}<br>Count: %{y}<extra></extra>'
            )])
            fig_b = style_chart(fig_b, "", show_grid=False)
            fig_b.update_xaxes(tickprefix='$', tickformat=',.0f')
            fig_b.update_layout(height=300, bargap=0.05)
            st.plotly_chart(fig_b, width='stretch', config=PLOTLY_CONFIG, key='salary_hist_b')
        else:
            st.info("No salary data available for Set B")
    
    # Top companies comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Set A: Top 5 Companies**")
        if len(df_a) > 0:
            top_companies_a = df_a['company'].value_counts().head(5)
            st.dataframe(top_companies_a.reset_index().rename(columns={'index': 'Company', 'company': 'Count'}), width='stretch')
        else:
            st.info("No data for Set A")
    
    with col2:
        st.markdown("**Set B: Top 5 Companies**")
        if len(df_b) > 0:
            top_companies_b = df_b['company'].value_counts().head(5)
            st.dataframe(top_companies_b.reset_index().rename(columns={'index': 'Company', 'company': 'Count'}), width='stretch')
        else:
            st.info("No data for Set B")
    
    # Top skills comparison
    skill_cols = [col for col in df.columns if col.startswith('skill_')]
    if skill_cols:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Set A: Top 10 Skills**")
            if len(df_a) > 0:
                skill_demand_a = {}
                total_jobs_a = len(df_a)
                for col in skill_cols:
                    skill_name = col.replace('skill_', '')
                    skill_demand_a[skill_name] = (df_a[col].sum() / total_jobs_a * 100)
                top_skills_a = dict(sorted(skill_demand_a.items(), key=lambda x: x[1], reverse=True)[:10])
                
                fig_a = go.Figure(data=[go.Bar(
                    x=list(top_skills_a.values()),
                    y=list(top_skills_a.keys()),
                    orientation='h',
                    marker=dict(
                        color=list(top_skills_a.values()),
                        colorscale=[[0, '#1A1A1A'], [0.5, '#808080'], [1, '#FFFFFF']],
                        line=dict(width=0),
                        cornerradius=5,
                        showscale=False
                    ),
                    hovertemplate='<b>%{y}</b><br>%{x:.1f}%<extra></extra>'
                )])
                fig_a = style_chart(fig_a, "", show_grid=False)
                fig_a.update_layout(
                    height=350,
                    showlegend=False,
                    yaxis=dict(categoryorder='total ascending'),
                    xaxis=dict(ticksuffix='%', showgrid=False)
                )
                st.plotly_chart(fig_a, width='stretch', config=PLOTLY_CONFIG, key='skills_a')
            else:
                st.info("No data for Set A")
        
        with col2:
            st.markdown("**Set B: Top 10 Skills**")
            if len(df_b) > 0:
                skill_demand_b = {}
                total_jobs_b = len(df_b)
                for col in skill_cols:
                    skill_name = col.replace('skill_', '')
                    skill_demand_b[skill_name] = (df_b[col].sum() / total_jobs_b * 100)
                top_skills_b = dict(sorted(skill_demand_b.items(), key=lambda x: x[1], reverse=True)[:10])
                
                fig_b = go.Figure(data=[go.Bar(
                    x=list(top_skills_b.values()),
                    y=list(top_skills_b.keys()),
                    orientation='h',
                    marker=dict(
                        color=list(top_skills_b.values()),
                        colorscale=[[0, '#1A1A1A'], [0.5, '#808080'], [1, '#FFFFFF']],
                        line=dict(width=0),
                        cornerradius=5,
                        showscale=False
                    ),
                    hovertemplate='<b>%{y}</b><br>%{x:.1f}%<extra></extra>'
                )])
                fig_b = style_chart(fig_b, "", show_grid=False)
                fig_b.update_layout(
                    height=350,
                    showlegend=False,
                    yaxis=dict(categoryorder='total ascending'),
                    xaxis=dict(ticksuffix='%', showgrid=False)
                )
                st.plotly_chart(fig_b, width='stretch', config=PLOTLY_CONFIG, key='skills_b')
            else:
                st.info("No data for Set B")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Built by @MShiyaji**")
st.sidebar.markdown(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
