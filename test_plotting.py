"""
Test plotting functions
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os

def plot_adherence_by_treatment(df, save_path=None):
    """
    Plot adherence rates by treatment group.
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
    plt.close()  # Close the figure to free memory
    print(f"[OK] Plot saved to {save_path}" if save_path else "[OK] Plot generated")

print("Testing plotting...")
try:
    df = pd.read_csv("data/processed_data/patient_data_processed.csv")

    # Create reports directory if it doesn't exist
    os.makedirs("reports/figures", exist_ok=True)

    # Test the plotting function
    plot_adherence_by_treatment(df, save_path="reports/figures/test_adherence_by_treatment.png")

except Exception as e:
    print(f"[FAIL] Failed to generate plot: {e}")
    import traceback
    traceback.print_exc()

print("Test complete.")