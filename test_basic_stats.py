"""
Test basic statistics function
"""
import pandas as pd
import os
import json
from datetime import datetime

def basic_statistics(df):
    """
    Calculate basic statistics about the dataset.
    """
    stats = {
        'n_patients': df['patient_id'].nunique(),
        'n_visits': len(df),
        'visits_per_patient': len(df) / df['patient_id'].nunique(),
        'adherence_rate': df['adherent'].mean(),
        'discontinuation_rate': df['discontinued'].mean(),
        'treatment_distribution': df['treatment'].value_counts().to_dict(),
        'gender_distribution': df['gender'].value_counts().to_dict(),
        'ethnicity_distribution': df['ethnicity'].value_counts().to_dict(),
        'age_stats': {
            'mean': df['age'].mean(),
            'std': df['age'].std(),
            'min': df['age'].min(),
            'max': df['age'].max()
        }
    }
    return stats

print("Testing basic statistics...")
try:
    df = pd.read_csv("data/processed_data/patient_data_processed.csv")
    df['visit_date'] = pd.to_datetime(df['visit_date'])
    stats = basic_statistics(df)
    print("[OK] Basic statistics calculated successfully")
    print(json.dumps(stats, indent=2, default=str))
except Exception as e:
    print(f"[FAIL] Failed to calculate basic statistics: {e}")
    import traceback
    traceback.print_exc()

print("Test complete.")