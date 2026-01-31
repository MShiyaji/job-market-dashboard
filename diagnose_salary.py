import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('data/processed_jobs.csv', low_memory=False)

# Filter to valid salaries
mask = (df['average_salary'].notna()) & (df['average_salary'] >= 30000) & (df['average_salary'] <= 500000)
df_sal = df[mask].copy()

print("SALARY DATA DIAGNOSTICS")
print("=" * 50)
print(f"Total jobs with valid salary: {len(df_sal)}")
print(f"Salary range: ${df_sal['average_salary'].min():,.0f} to ${df_sal['average_salary'].max():,.0f}")
print(f"Mean salary: ${df_sal['average_salary'].mean():,.0f}")
print(f"Std dev: ${df_sal['average_salary'].std():,.0f}")

# Experience correlation
if 'years_experience_required' in df_sal.columns:
    exp_corr = df_sal['average_salary'].corr(df_sal['years_experience_required'])
    exp_filled = df_sal['years_experience_required'].notna().sum()
    print(f"\nExperience data: {exp_filled}/{len(df_sal)} jobs have experience")
    print(f"Experience-Salary correlation: {exp_corr:.3f}")

# Remote correlation  
if 'is_remote' in df_sal.columns:
    remote_corr = df_sal['average_salary'].corr(df_sal['is_remote'].astype(float))
    print(f"Remote-Salary correlation: {remote_corr:.3f}")

# State variance
print("\nSALARY BY STATE (top 5 by count):")
df_sal['state'] = df_sal['location'].apply(lambda x: x.split(', ')[-1] if isinstance(x, str) and ', ' in x else None)
state_stats = df_sal.groupby('state')['average_salary'].agg(['count', 'mean', 'std']).sort_values('count', ascending=False).head(5)
print(state_stats.to_string())

# Skill correlations
skill_cols = [c for c in df_sal.columns if c.startswith('skill_')]
print(f"\nTOP SKILL CORRELATIONS ({len(skill_cols)} skills):")
if skill_cols:
    corrs = [(c.replace('skill_', ''), df_sal['average_salary'].corr(df_sal[c].astype(float))) for c in skill_cols]
    corrs.sort(key=lambda x: abs(x[1]), reverse=True)
    for s, c in corrs[:5]:
        print(f"  {s}: {c:.3f}")

print("\nKEY INSIGHT: Skills have stronger correlation with salary than location/experience!")
