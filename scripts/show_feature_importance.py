#!/usr/bin/env python
"""
Show Feature Importance for Salary Models

This script loads the pre-trained models and displays the feature importance
ranking for each role.
"""

import sys
import os
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from salary_predictor import SalaryPredictor

def main():
    print("=" * 60)
    print("Feature Importance Ranking by Role")
    print("=" * 60)
    
    predictor = SalaryPredictor()
    
    # Load trained models
    if not predictor.load_models():
        print("No trained models found. Please run scripts/train_salary_models.py first.")
        return

    roles = predictor.get_available_roles()
    
    for role in roles:
        print(f"\nRole: {role}")
        print("-" * 30)
        
        importance_df = predictor.get_feature_importance(role)
        
        if importance_df is not None and not importance_df.empty:
            # Format importance for display
            print(importance_df.to_string(index=False, float_format=lambda x: "{:.4f}".format(x)))
        else:
            print("Feature importance not available for this model.")
            
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
