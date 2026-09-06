"""
Exploratory Data Analysis for PharmaPulse project.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set backend to avoid opening windows
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import json

# Set style for plots
# plt.style.use('seaborn-v0_8')  # This style may not be available, using seaborn defaults
sns.set_palette("husl")

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

def basic_statistics(df: pd.DataFrame) -> dict:
    """
    Calculate basic statistics about the dataset.

    Args:
        df: Input DataFrame

    Returns:
        Dictionary containing basic statistics
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

def plot_adherence_by_treatment(df: pd.DataFrame, save_path: str = None):
    """
    Plot adherence rates by treatment group.

    Args:
        df: Input DataFrame
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(10, 6))
    adherence_by_treatment = df.groupby('treatment')['adherent'].mean()
    adherence_by_treatment.plot(kind='bar')
    plt.title('Adherence Rate by Treatment Group')
    plt.xlabel('Treatment Group (0=Control, 1=Treatment A, 2=Treatment B)')
    plt.ylabel('Adherence Rate')
    plt.xticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_discontinuation_over_time(df: pd.DataFrame, save_path: str = None):
    """
    Plot discontinuation rates over time.

    Args:
        df: Input DataFrame
        save_path: Path to save the plot (optional)
    """
    # Get last visit for each patient
    last_visits = df.sort_values('visit_date').groupby('patient_id').tail(1)

    plt.figure(figsize=(12, 6))
    # Convert visit_date to datetime if not already
    last_visits['visit_date'] = pd.to_datetime(last_visits['visit_date'])
    last_visits['month'] = last_visits['visit_date'].dt.to_period('M')
    disc_over_time = last_visits.groupby('month')['discontinued'].mean()

    disc_over_time.plot(kind='line', marker='o')
    plt.title('Discontinuation Rate Over Time')
    plt.xlabel('Month')
    plt.ylabel('Discontinuation Rate')
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_clinical_features_distribution(df: pd.DataFrame, save_path: str = None):
    """
    Plot distribution of key clinical features.

    Args:
        df: Input DataFrame
        save_path: Path to save the plot (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Systolic BP distribution
    axes[0, 0].hist(df['systolic_bp'], bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Distribution of Systolic Blood Pressure')
    axes[0, 0].set_xlabel('Systolic BP (mmHg)')
    axes[0, 0].set_ylabel('Frequency')

    # HbA1c distribution
    axes[0, 1].hist(df['hba1c'], bins=30, edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('Distribution of HbA1c')
    axes[0, 1].set_xlabel('HbA1c (%)')
    axes[0, 1].set_ylabel('Frequency')

    # BP vs Adherence
    adherence_bp = df.groupby('adherent')['systolic_bp'].mean()
    axes[1, 0].bar(['Non-Adherent', 'Adherent'], adherence_bp.values)
    axes[1, 0].set_title('Average Systolic BP by Adherence Status')
    axes[1, 0].set_ylabel('Average Systolic BP (mmHg)')

    # HbA1c vs Adherence
    adherence_hba1c = df.groupby('adherent')['hba1c'].mean()
    axes[1, 1].bar(['Non-Adherent', 'Adherent'], adherence_hba1c.values)
    axes[1, 1].set_title('Average HbA1c by Adherence Status')
    axes[1, 1].set_ylabel('Average HbA1c (%)')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def correlation_analysis(df: pd.DataFrame, save_path: str = None):
    """
    Perform correlation analysis and generate heatmap.

    Args:
        df: Input DataFrame
        save_path: Path to save the plot (optional)
    """
    # Select numeric columns for correlation
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude ID-like columns if any
    numeric_cols = [col for col in numeric_cols if 'id' not in col.lower() and 'visit' not in col.lower()]

    corr_matrix = df[numeric_cols].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix of Numeric Features')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return corr_matrix

def generate_eda_report(df: pd.DataFrame, output_dir: str = "reports/"):
    """
    Generate a comprehensive EDA report.

    Args:
        df: Input DataFrame
        output_dir: Directory to save reports and plots
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)

    # Calculate basic statistics
    stats = basic_statistics(df)

    # Save statistics to JSON
    with open(os.path.join(output_dir, "basic_statistics.json"), 'w') as f:
        json.dump(stats, f, indent=2, default=str)

    # Generate plots
    plot_adherence_by_treatment(df,
                               save_path=os.path.join(output_dir, "figures", "adherence_by_treatment.png"))

    plot_discontinuation_over_time(df,
                                  save_path=os.path.join(output_dir, "figures", "discontinuation_over_time.png"))

    plot_clinical_features_distribution(df,
                                       save_path=os.path.join(output_dir, "figures", "clinical_features_distribution.png"))

    corr_matrix = correlation_analysis(df,
                                      save_path=os.path.join(output_dir, "figures", "correlation_heatmap.png"))

    # Save correlation matrix
    corr_matrix.to_csv(os.path.join(output_dir, "correlation_matrix.csv"))

    # Create a summary report
    report = f"""
# PharmaPulse - Exploratory Data Analysis Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Overview
- Number of Patients: {stats['n_patients']}
- Number of Visits: {stats['n_visits']}
- Average Visits per Patient: {stats['visits_per_patient']:.2f}

## Outcome Rates
- Adherence Rate: {stats['adherence_rate']:.2%}
- Discontinuation Rate: {stats['discontinuation_rate']:.2%}

## Demographics
- Age: Mean={stats['age_stats']['mean']:.1f}±{stats['age_stats']['std']:.1f} years
- Gender Distribution: {stats['gender_distribution']}
- Ethnicity Distribution: {stats['ethnicity_distribution']}

## Treatment Distribution
{stats['treatment_distribution']}

## Key Findings
1. Adherence rates vary by treatment group (see adherence_by_treatment.png)
2. Discontinuation patterns over time (see discontinuation_over_time.png)
3. Clinical features show expected relationships with adherence (see clinical_features_distribution.png)
4. Feature correlations available in correlation_matrix.csv and correlation_heatmap.png

Reports and figures saved to: {output_dir}
"""

    with open(os.path.join(output_dir, "eda_report.md"), 'w') as f:
        f.write(report)

    print(f"EDA report generated and saved to {output_dir}")
    return stats

if __name__ == "__main__":
    # Load data
    df = load_data()

    # Generate EDA report
    stats = generate_eda_report(df)

    # Print summary to console
    print("\n=== EDA Summary ===")
    print(f"Patients: {stats['n_patients']}")
    print(f"Visits: {stats['n_visits']}")
    print(f"Adherence Rate: {stats['adherence_rate']:.2%}")
    print(f"Discontinuation Rate: {stats['discontinuation_rate']:.2%}")