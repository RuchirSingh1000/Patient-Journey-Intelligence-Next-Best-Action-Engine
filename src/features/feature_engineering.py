"""
Feature engineering for PharmaPulse project.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json

def load_data(data_path: str = "data/processed_data/patient_data_processed.csv") -> pd.DataFrame:
    """
    Load the processed patient data.

    Args:
        data_path: Path to the processed data CSV file

    Returns:
        Loaded DataFrame
    """
    df = pd.read_csv(data_path)
    df['visit_date'] = pd.to_datetime(df['visit_date'])
    return df

def create_patient_level_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create patient-level features from longitudinal data.

    Args:
        df: Input DataFrame with longitudinal data

    Returns:
        DataFrame with one row per patient and engineered features
    """
    # Sort by patient and visit date
    df_sorted = df.sort_values(['patient_id', 'visit_date']).copy()

    # Group by patient
    grouped = df_sorted.groupby('patient_id')

    # Initialize list to hold patient-level records
    patient_records = []

    for patient_id, group in grouped:
        # Basic demographics (take from first visit)
        first_visit = group.iloc[0]
        patient_record = {
            'patient_id': patient_id,
            'age': first_visit['age'],
            'gender': first_visit['gender'],
            'ethnicity': first_visit['ethnicity'],
            'diabetes': first_visit['diabetes'],
            'hypertension': first_visit['hypertension'],
            'depression': first_visit['depression'],
            'asthma': first_visit['asthma'],
            'obesity': first_visit['obesity'],
        }

        # Treatment exposure: we'll consider the most frequent treatment or the last treatment
        # For simplicity, we'll use the treatment from the last visit
        last_visit = group.iloc[-1]
        patient_record['treatment'] = last_visit['treatment']

        # Longitudinal features
        n_visits = len(group)
        patient_record['n_visits'] = n_visits

        # Adherence features
        adherence_series = group['adherent']
        patient_record['adherence_rate'] = adherence_series.mean()
        patient_record['n_adherent_visits'] = adherence_series.sum()
        patient_record['n_nonadherent_visits'] = n_visits - adherence_series.sum()
        # Streak of adherent visits at the end
        adherent_streak = 0
        for val in reversed(adherence_series.values):
            if val == 1:
                adherent_streak += 1
            else:
                break
        patient_record['current_adherent_streak'] = adherent_streak

        # Discontinuation: we define discontinuation as having a discontinuation event at any visit
        # Note: in our data, discontinuation is only recorded at the last visit for simplicity
        # But we'll check if any visit has discontinuation=1
        patient_record['discontinued'] = group['discontinued'].max()  # 1 if any visit discontinued

        # Time to discontinuation (if discontinued, else censored at last visit)
        if patient_record['discontinued'] == 1:
            # Find the first visit where discontinued=1 (in our data, it's only at the last visit)
            disc_visit = group[group['discontinued'] == 1].iloc[0]
            # Time from first visit to discontinuation visit
            time_to_event = (disc_visit['visit_date'] - first_visit['visit_date']).days
        else:
            # Censored at last visit
            time_to_event = (last_visit['visit_date'] - first_visit['visit_date']).days
        patient_record['time_to_discontinuation'] = time_to_event

        # Clinical features: trends and variability
        # Systolic BP: mean, std, trend (slope over time)
        if n_visits > 1:
            x = np.arange(n_visits)
            sbp_vals = group['systolic_bp'].values
            hba1c_vals = group['hba1c'].values
            # Calculate slopes using linear regression
            sbp_slope = np.polyfit(x, sbp_vals, 1)[0] if len(x) > 1 else 0
            hba1c_slope = np.polyfit(x, hba1c_vals, 1)[0] if len(x) > 1 else 0
        else:
            sbp_slope = 0
            hba1c_slope = 0

        patient_record['systolic_bp_mean'] = group['systolic_bp'].mean()
        patient_record['systolic_bp_std'] = group['systolic_bp'].std()
        patient_record['systolic_bp_slope'] = sbp_slope
        patient_record['hba1c_mean'] = group['hba1c'].mean()
        patient_record['hba1c_std'] = group['hba1c'].std()
        patient_record['hba1c_slope'] = hba1c_slope

        # Visit frequency: average time between visits
        if n_visits > 1:
            visit_dates = group['visit_date'].sort_values()
            time_between_visits = (visit_dates.diff().dt.days).dropna()
            patient_record['mean_time_between_visits'] = time_between_visits.mean()
            patient_record['std_time_between_visits'] = time_between_visits.std()
        else:
            patient_record['mean_time_between_visits'] = np.nan
            patient_record['std_time_between_visits'] = np.nan

        # Comorbidity count
        comorbidity_cols = ['diabetes', 'hypertension', 'depression', 'asthma', 'obesity']
        patient_record['comorbidity_count'] = group[comorbidity_cols].iloc[0].sum()  # same for all visits

        patient_records.append(patient_record)

    # Convert to DataFrame
    patient_df = pd.DataFrame(patient_records)

    # Handle missing values for visit time features (if only one visit)
    patient_df['mean_time_between_visits'].fillna(0, inplace=True)
    patient_df['std_time_between_visits'].fillna(0, inplace=True)

    return patient_df

def save_patient_features(df: pd.DataFrame,
                          output_path: str = "data/processed_data/patient_features.csv"):
    """
    Save patient-level features to CSV.

    Args:
        df: DataFrame with patient-level features
        output_path: Path to save the CSV file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Patient-level features saved to {output_path}")
    print(f"Shape: {df.shape}")

if __name__ == "__main__":
    # Load data
    df = load_data()

    # Create patient-level features
    patient_df = create_patient_level_features(df)

    # Save features
    save_patient_features(patient_df)

    # Print summary
    print("\n=== Patient-level Features Summary ===")
    print(f"Number of patients: {len(patient_df)}")
    print(f"Features: {patient_df.columns.tolist()}")
    print("\nFirst few rows:")
    print(patient_df.head())