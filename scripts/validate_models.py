#!/usr/bin/env python
"""
Validate trained salary prediction models.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
from pathlib import Path

MODELS_DIR = Path('data/models')

def validate_models():
    print('=== Model Validation Report ===\n')
    
    for model_path in sorted(MODELS_DIR.glob('*_model.joblib')):
        role = model_path.stem.replace('_model', '').replace('_', ' ')
        try:
            data = joblib.load(model_path)
            model = data["model"]
            metrics = data["metrics"]
            
            print(f'{role}:')
            print(f'  Model type: {type(model).__name__}')
            print(f'  Best model name: {metrics["model_name"]}')
            print(f'  R2 score: {metrics["r2"]:.3f}')
            print(f'  MAE: ${metrics["mae"]:,.0f}')
            print(f'  RMSE: ${metrics["rmse"]:,.0f}')
            print(f'  Features ({len(data["feature_names"])}): {data["feature_names"][:5]}...')
            
            top_skills = data.get("top_skills", [])
            print(f'  Top skills ({len(top_skills)}): {top_skills[:3]}...')
            
            # Check if CV scores exist and show comparison
            cv_scores = metrics.get('cv_scores', {})
            if cv_scores:
                print(f'  CV Comparison:')
                for name, scores in cv_scores.items():
                    print(f'    {name}: R2={scores["cv_r2_mean"]:.3f} (+/-{scores["cv_r2_std"]:.3f})')
            print()
        except Exception as e:
            print(f'{role}: ERROR - {e}\n')

if __name__ == "__main__":
    validate_models()
