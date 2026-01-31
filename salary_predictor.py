"""
Salary Prediction Module

Trains per-role ML models to predict salary based on:
- Remote status
- Education level
- Years of experience
- Top skills for the role
"""

import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# Directory for saved models
MODELS_DIR = Path('data/models')
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class SalaryPredictor:
    """
    Per-role salary prediction using ML models.
    
    For each role category:
    1. Selects top 5 predictive skills via RF feature importance
    2. Trains best model (Linear, RF, or GradientBoosting)
    3. Saves model for reuse
    """
    
    def __init__(self):
        self.models: Dict[str, dict] = {}  # role -> {model, encoder, features, metrics, top_skills}
        self.education_encoder = LabelEncoder()
        self.all_skill_cols: List[str] = []  # All available skill columns
        self.is_trained = False
    

    
    @staticmethod
    def _extract_education_level(text: str) -> Optional[str]:
        """Extract highest education requirement from text."""
        import re
        if not isinstance(text, str):
            return None
        t = text.lower()
        # Order matters: PhD > Master's > Bachelor's
        if re.search(r"\b(ph\.?d|doctorate|doctoral)\b", t):
            return "PhD"
        if re.search(r"\b(postgraduate|advanced degree|advanced education)\b", t):
            return "Master's"
        if re.search(r"\b(master[''`s]*|masters|m\.sc\.|msc\b|m\.s\.|ms\b)\b", t):
            return "Master's"
        # if re.search(r"\b(bachelor[''`s]*|b\.sc\.|bsc\b|b\.s\.|bs\b)\b", t):
        #     return "Bachelor's"
        return "Bachelor's"
    
    @staticmethod
    def _categorize_job_title(title: str) -> str:
        """Categorize job title into role category."""
        if not isinstance(title, str):
            return "Other"
        t = title.lower().strip()
        if 'data scientist' in t or 'research scientist' in t:
            return 'Data Scientist'
        elif 'machine learning' in t or 'ml engineer' in t or 'applied scientist' in t or 'deep learning' in t:
            return 'Machine Learning Engineer'
        elif 'ai engineer' in t or 'artificial intelligence' in t:
            return 'AI Engineer'
        elif 'data engineer' in t or 'etl' in t or 'analytics engineer' in t or 'analytics specialist' in t:
            return 'Data Engineer'
        elif 'data analyst' in t or 'analyst' in t or 'analytics' in t or 'business intelligence' in t:
            return 'Data Analyst'
        elif 'software engineer' in t or 'developer' in t:
            return 'Software Engineer'
        else:
            return 'Other'

    @staticmethod
    def _classify_col(location: str) -> int:
        """
        Classify location into Cost of Living tiers.
        2 = HCOL (High), 1 = MCOL (Medium), 0 = LCOL (Low/Unknown)
        """
        if not isinstance(location, str):
            return 0
        
        loc = location.lower()
        
        # HCOL Keywords (Tier 2)
        hcol_keywords = [
            'san francisco', 'bay area', 'new york', 'nyc', 'seattle', 'boston',
            'los angeles', 'washington dc', 'd.c.', 'silicon valley', 'california', 
            ' ca', ' ny', ' wa', ' ma'
        ]
        if any(k in loc for k in hcol_keywords):
            return 2
            
        # MCOL Keywords (Tier 1)
        mcol_keywords = [
            'austin', 'chicago', 'denver', 'atlanta', 'philadelphia', 'phoenix',
            'dallas', 'houston', 'miami', 'texas', ' tx', 'colorado', ' co',
            'illinois', ' il', 'georgia', ' ga', 'florida', ' fl'
        ]
        if any(k in loc for k in mcol_keywords):
            return 1
            
        return 0
        
    @staticmethod
    def _extract_experience_from_title(title: str) -> Optional[float]:
        """
        Extract approximate years of experience from job title keywords.
        Returns estimated years or None if no keywords match.
        """
        if not isinstance(title, str):
            return None
        
        t = title.lower()
        
        # Executive / C-Suite (15+)
        if any(x in t for x in ['chief', 'cto', 'cio', 'cdo', 'ciso', 'svp', 'evp']):
             return 15.0
             
        # Director/VP (10+)
        if any(x in t for x in ['vice president', ' vp', 'vp ', 'director', 'head of']):
            return 10.0
        
        # Principal/Staff/Architect (8+)
        if any(x in t for x in ['principal', 'staff', 'distinguished', 'architect', 'fellow']):
            return 8.0
            
        # Manager/Lead (6+)
        if any(x in t for x in ['manager', 'lead', 'supervisor']):
            return 6.0
            
        # Senior (5+)
        if 'senior' in t or 'expert' in t or 'advanced' in t:
            return 5.0
        if 'sr.' in t or 'sr ' in t or t.endswith(' sr') or '(sr)' in t:
            return 5.0
            
        # Levels (IV, V, VI are usually Senior+)
        if any(x in t for x in [' iv', ' v ', ' vi ']):
            return 5.0
            
        # Level III (Often Senior, sometimes Mid-Senior)
        if ' iii' in t:
            return 5.0
            
        # Mid-level (3+)
        if any(x in t for x in ['intermediate', 'mid-level', 'mid level', 'middle']):
            return 3.0
        if ' ii' in t:
            return 3.0
            
        # Junior (1-2)
        if any(x in t for x in ['junior', 'jr.', 'jr ', 'associate']):
            return 1.0
        if t.endswith(' jr') or '(jr)' in t:
            return 1.0
        if ' i ' in t or t.endswith(' i'):
             return 1.0

        # Entry / Intern (0)
        if any(x in t for x in ['entry', 'intern', 'trainee', 'apprentice', 'graduate', 'fresh', 'new grad']):
            return 0.0
            
        return None

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare dataset for training.
        Derives missing columns and filters to jobs with valid salary data.
        """
        df_clean = df.copy()
        
        # Derive role_category if not present
        if 'role_category' not in df_clean.columns:
            if 'title' in df_clean.columns:
                df_clean['role_category'] = df_clean['title'].apply(self._categorize_job_title)
            else:
                df_clean['role_category'] = 'Other'
        
        # Derive education_level if not present
        if 'education_level' not in df_clean.columns:
            edu_source = None
            for c in ['requirements', 'description', 'job_description', 'full_description', 'posting_text']:
                if c in df_clean.columns:
                    edu_source = c
                    break
            if edu_source:
                df_clean['education_level'] = df_clean[edu_source].apply(self._extract_education_level)
            else:
                df_clean['education_level'] = None

        # Derive Cost of Living tier if not present
        if 'col_tier' not in df_clean.columns:
            if 'location' in df_clean.columns:
                df_clean['col_tier'] = df_clean['location'].apply(self._classify_col)
            else:
                df_clean['col_tier'] = 0
        
        # Filter to rows with salary
        df_clean = df_clean[df_clean['average_salary'].notna()]
        
        # Remove outliers (salaries < 30k or > 500k are likely errors)
        df_clean = df_clean[
            (df_clean['average_salary'] >= 30000) & 
            (df_clean['average_salary'] <= 500000)
        ]
        
        # Fill missing values
        df_clean['is_remote'] = df_clean['is_remote'].fillna(False).astype(int)
        
        # Fill missing experience from title first, then median
        if 'years_experience_required' in df_clean.columns:
            # 1. Extract from title for ALL rows
            title_exp = df_clean['title'].apply(self._extract_experience_from_title)
            
            # 2. Use title experience where available, otherwise use existing data
            df_clean['years_experience_required'] = title_exp.fillna(df_clean['years_experience_required'])
            
            # 3. Fill remaining with median (or default 2.0)
            median_exp = df_clean['years_experience_required'].median()
            if pd.isna(median_exp):
                median_exp = 2.0
            df_clean['years_experience_required'] = df_clean['years_experience_required'].fillna(median_exp)
        else:
            # If column doesn't exist, try to create it entirely from title
            title_exp = df_clean['title'].apply(self._extract_experience_from_title)
            df_clean['years_experience_required'] = title_exp.fillna(2.0)
        df_clean['education_level'] = df_clean['education_level'].fillna('No education')
        df_clean['role_category'] = df_clean['role_category'].fillna('Other')
        
        return df_clean

    
    
    def _encode_education(self, df: pd.DataFrame, fit: bool = True) -> pd.Series:
        """Encode education level."""
        edu_order = ['No education', "Bachelor's", "Master's", "PhD"]
        
        if fit:
            self.education_encoder.fit(edu_order)
        
        # Map to known categories
        education = df['education_level'].apply(
            lambda x: x if x in edu_order else 'No education'
        )
        
        return pd.Series(
            self.education_encoder.transform(education),
            index=df.index
        )
    
    def _build_features(self, df: pd.DataFrame, 
                        fit_encoders: bool = True,
                        skill_cols: List[str] = None) -> pd.DataFrame:
        """Build feature matrix for a role (remote, education, experience, skills)."""
        features = pd.DataFrame(index=df.index)
        
        # Remote flag
        features['is_remote'] = df['is_remote'].astype(int)
        
        # Education (encoded)
        features['education_encoded'] = self._encode_education(df, fit=fit_encoders)
        
        # Cost of Living Tier
        if 'col_tier' in df.columns:
            features['col_tier'] = df['col_tier'].fillna(0).astype(int)
        else:
            features['col_tier'] = 0
        
        # Experience
        features['years_experience'] = df['years_experience_required'].fillna(0)
        
        # Skills (binary features)
        if skill_cols:
            for col in skill_cols:
                if col in df.columns:
                    features[col] = df[col].fillna(0).astype(int)
                else:
                    features[col] = 0
        
        return features
    
    def _select_top_skills(self, df: pd.DataFrame, y: np.ndarray, n_skills: int = 10) -> List[str]:
        """
        Select top N most predictive skills for a role using Random Forest feature importance.
        """
        # Get all skill columns
        skill_cols = [c for c in df.columns if c.startswith('skill_')]
        if not skill_cols:
            return []
        
        # Build skill-only feature matrix
        X_skills = df[skill_cols].fillna(0).astype(int).values
        
        # Quick RF to get feature importance
        rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X_skills, y)
        
        # Get top N skills by importance
        importance = pd.DataFrame({
            'skill': skill_cols,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        top_skills = importance.head(n_skills)['skill'].tolist()
        return top_skills
    
    def _train_best_model(self, X: np.ndarray, y: np.ndarray) -> Tuple[object, dict]:
        """
        Train multiple models and return the best one based on CV R² score.
        """
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(
                n_estimators=100, max_depth=15, min_samples_split=5,
                random_state=42, n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                random_state=42
            )
        }
        
        best_model = None
        best_score = -np.inf
        best_name = ""
        all_scores = {}
        
        for name, model in models.items():
            try:
                # 5-fold CV
                scores = cross_val_score(model, X, y, cv=5, scoring='r2')
                mean_score = scores.mean()
                all_scores[name] = {
                    'cv_r2_mean': mean_score,
                    'cv_r2_std': scores.std()
                }
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_model = model
                    best_name = name
            except Exception as e:
                print(f"Error training {name}: {e}")
                continue
        
        if best_model is None:
            raise ValueError("No model could be trained successfully")
        
        # Train best model on full data
        best_model.fit(X, y)
        
        # Calculate final metrics
        y_pred = best_model.predict(X)
        metrics = {
            'model_name': best_name,
            'r2': r2_score(y, y_pred),
            'mae': mean_absolute_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'cv_scores': all_scores
        }
        
        return best_model, metrics
    
    def train(self, df: pd.DataFrame) -> Dict[str, dict]:
        """
        Train a separate model for each role category.
        
        Returns dict of {role: metrics} for each trained model.
        """
        df_clean = self.prepare_data(df)
        
        # Store all skill columns for reference
        self.all_skill_cols = [c for c in df_clean.columns if c.startswith('skill_')]
        
        if len(df_clean) < 20:
            raise ValueError(f"Insufficient data: only {len(df_clean)} jobs with valid salary")
        
        roles = df_clean['role_category'].unique()
        results = {}
        
        for role in roles:
            # Skip "Other" category - too heterogeneous for meaningful predictions
            if role == 'Other':
                print(f"Skipping {role}: catch-all category excluded from training")
                continue
            
            role_df = df_clean[df_clean['role_category'] == role]
            
            # Need minimum samples to train
            if len(role_df) < 15:
                print(f"Skipping {role}: only {len(role_df)} samples (need 15+)")
                continue
            
            print(f"Training model for {role} ({len(role_df)} samples)...")
            
            try:
                y = role_df['average_salary'].values
                
                # Select top 10 most predictive skills for this role
                top_skills = self._select_top_skills(role_df, y, n_skills=10)
                print(f"  Top skills: {[s.replace('skill_', '') for s in top_skills[:5]]}...")
                
                # Build features including skills
                X_df = self._build_features(role_df, fit_encoders=True, skill_cols=top_skills)
                X = X_df.values
                
                # Train best model
                model, metrics = self._train_best_model(X, y)
                
                # Store model info including top skills
                self.models[role] = {
                    'model': model,
                    'feature_names': X_df.columns.tolist(),
                    'metrics': metrics,
                    'education_encoder': self.education_encoder,
                    'top_skills': top_skills,  # Store top skills for this role
                    'all_skill_cols': self.all_skill_cols  # Store all skills for reference
                }
                
                results[role] = metrics
                
                # Save model
                model_path = MODELS_DIR / f"{role.replace(' ', '_')}_model.joblib"
                joblib.dump(self.models[role], model_path)
                print(f"  R²: {metrics['r2']:.3f}, MAE: ${metrics['mae']:,.0f}")
                print(f"  Saved to {model_path}")
                
            except Exception as e:
                print(f"Error training {role}: {e}")
                continue
        
        self.is_trained = len(self.models) > 0
        return results
    
    def load_models(self) -> bool:
        """Load all saved models from disk."""
        if not MODELS_DIR.exists():
            return False
        
        model_files = list(MODELS_DIR.glob("*_model.joblib"))
        if not model_files:
            return False
        
        for model_path in model_files:
            try:
                model_data = joblib.load(model_path)
                role = model_path.stem.replace('_model', '').replace('_', ' ')
                self.models[role] = model_data
                print(f"Loaded model for {role}")
            except Exception as e:
                print(f"Error loading {model_path}: {e}")
                continue
        
        self.is_trained = len(self.models) > 0
        return self.is_trained
    
    def predict(self, role_category: str, is_remote: bool,
                education_level: str, years_experience: float,
                location: str = None,
                skills: List[str] = None) -> Optional[Dict]:
        """
        Predict salary for given job characteristics.
        
        Args:
            role_category: Job role (e.g., 'Data Scientist')
            is_remote: Whether the position is remote
            education_level: Education level (Bachelor's, Master's, PhD)
            years_experience: Years of experience
            skills: List of skill names (e.g., ['Python', 'SQL', 'Machine Learning'])
        
        Returns dict with predicted_salary, confidence_interval, and model_info.
        """
        if role_category not in self.models:
            return None
        
        model_info = self.models[role_category]
        model = model_info['model']
        feature_names = model_info['feature_names']
        top_skills = model_info.get('top_skills', [])
        
        # Restore encoders
        self.education_encoder = model_info['education_encoder']
        
        # Build feature vector
        features = {}
        
        # Remote
        features['is_remote'] = int(is_remote)
        
        # Education
        edu_order = ['No education', "Bachelor's", "Master's", "PhD"]
        edu_mapped = education_level if education_level in edu_order else 'No education'
        try:
            features['education_encoded'] = self.education_encoder.transform([edu_mapped])[0]
        except:
            features['education_encoded'] = 0
            
        # Cost of Living
        features['col_tier'] = self._classify_col(location) if location else 0
        
        # Experience
        features['years_experience'] = years_experience
        
        # Skills - convert skill names to skill_* column format and set binary flags
        skills_set = set(skills) if skills else set()
        for skill_col in top_skills:
            skill_name = skill_col.replace('skill_', '')
            features[skill_col] = 1 if skill_name in skills_set else 0
        
        # Create feature array in correct order
        X = np.array([[features.get(f, 0) for f in feature_names]])
        
        # Predict
        predicted_salary = model.predict(X)[0]
        
        # Estimate confidence interval (rough approximation)
        mae = model_info['metrics']['mae']
        
        return {
            'predicted_salary': predicted_salary,
            'lower_bound': predicted_salary - mae,
            'upper_bound': predicted_salary + mae,
            'model_name': model_info['metrics']['model_name'],
            'r2': model_info['metrics']['r2']
        }
    
    def get_top_skills(self, role_category: str, n: int = 5) -> List[str]:
        """
        Get the top N most important skills for a role.
        Returns skill names (without 'skill_' prefix) for UI display.
        """
        if role_category not in self.models:
            return []
        
        top_skills = self.models[role_category].get('top_skills', [])
        # Return skill names without prefix, limited to n
        return [s.replace('skill_', '') for s in top_skills[:n]]
    
    def get_all_skills(self, role_category: str = None) -> List[str]:
        """
        Get all available skill names for selection.
        Returns skill names (without 'skill_' prefix) for UI display.
        """
        if role_category and role_category in self.models:
            all_skills = self.models[role_category].get('all_skill_cols', [])
        else:
            # Fallback to any available model's skill list
            for model_info in self.models.values():
                all_skills = model_info.get('all_skill_cols', [])
                if all_skills:
                    break
            else:
                all_skills = []
        
        return [s.replace('skill_', '') for s in all_skills]
    
    def get_feature_importance(self, role_category: str) -> Optional[pd.DataFrame]:
        """
        Get feature importance for a role's model.
        Only works for tree-based models.
        """
        if role_category not in self.models:
            return None
        
        model_info = self.models[role_category]
        model = model_info['model']
        feature_names = model_info['feature_names']
        
        if hasattr(model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance
        elif hasattr(model, 'coef_'):
            # For linear model, use absolute coefficients
            importance = pd.DataFrame({
                'feature': feature_names,
                'importance': np.abs(model.coef_)
            }).sort_values('importance', ascending=False)
            return importance
        
        return None
    
    def get_available_roles(self) -> List[str]:
        """Get list of roles with trained models."""
        return list(self.models.keys())
    
    def get_model_metrics(self) -> Dict[str, dict]:
        """Get metrics for all trained models."""
        return {role: info['metrics'] for role, info in self.models.items()}


# Example usage
if __name__ == "__main__":
    # Load sample data
    try:
        df = pd.read_csv('data/processed_jobs.csv', low_memory=False)
        
        predictor = SalaryPredictor()
        results = predictor.train(df)
        
        print("\n=== Training Results ===")
        for role, metrics in results.items():
            print(f"\n{role}:")
            print(f"  Best Model: {metrics['model_name']}")
            print(f"  R²: {metrics['r2']:.3f}")
            print(f"  MAE: ${metrics['mae']:,.0f}")
            print(f"  RMSE: ${metrics['rmse']:,.0f}")
        
        # Test prediction
        if predictor.is_trained:
            roles = predictor.get_available_roles()
            if roles:
                test_role = roles[0]
                result = predictor.predict(
                    role_category=test_role,
                    is_remote=True,
                    education_level="Master's",
                    years_experience=3
                )
                if result:
                    print(f"\n=== Sample Prediction for {test_role} ===")
                    print(f"Predicted Salary: ${result['predicted_salary']:,.0f}")
                    print(f"Range: ${result['lower_bound']:,.0f} - ${result['upper_bound']:,.0f}")
                    
    except FileNotFoundError:
        print("No data file found. Run from project root with data/processed_jobs.csv")
