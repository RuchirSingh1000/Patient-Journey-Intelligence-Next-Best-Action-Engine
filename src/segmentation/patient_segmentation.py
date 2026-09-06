"""
Patient segmentation using clustering and dimensionality reduction.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import os
import json

def load_patient_features(feature_path: str = "data/processed_data/patient_features.csv") -> pd.DataFrame:
    """
    Load patient-level features.

    Args:
        feature_path: Path to the patient features CSV file

    Returns:
        Loaded DataFrame
    """
    df = pd.read_csv(feature_path)
    return df

def prepare_features_for_clustering(df: pd.DataFrame) -> tuple:
    """
    Prepare features for clustering by selecting numeric columns and scaling.

    Args:
        df: Input DataFrame with patient features

    Returns:
        Tuple of (scaled_features, feature_names, scaler)
    """
    # Select numeric columns for clustering
    # Exclude ID and categorical columns that need encoding
    exclude_cols = ['patient_id', 'gender', 'ethnicity']  # We'll encode these separately
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    # Remove any remaining non-numeric or ID-like columns
    feature_cols = [col for col in numeric_features if col not in exclude_cols]

    # For simplicity, we'll also exclude the target variables if we plan to use them later
    # But for segmentation, we can include all features except ID
    # Let's use all numeric columns except patient_id
    feature_cols = [col for col in df.columns if col != 'patient_id' and df[col].dtype in [np.float64, np.int64]]

    X = df[feature_cols].copy()

    # Handle missing values (if any)
    X = X.fillna(X.mean())

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, feature_cols, scaler

def find_optimal_clusters(X_scaled, max_clusters: int = 10) -> dict:
    """
    Find optimal number of clusters using elbow method and silhouette score.

    Args:
        X_scaled: Scaled feature matrix
        max_clusters: Maximum number of clusters to consider

    Returns:
        Dictionary with scores for different numbers of clusters
    """
    inertia = []
    silhouette_scores = []
    ch_scores = []  # Calinski-Harabasz
    db_scores = []  # Davies-Bouldin

    K_range = range(2, max_clusters + 1)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)

        inertia.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, cluster_labels))
        ch_scores.append(calinski_harabasz_score(X_scaled, cluster_labels))
        db_scores.append(davies_bouldin_score(X_scaled, cluster_labels))

    results = {
        'K': list(K_range),
        'inertia': inertia,
        'silhouette_score': silhouette_scores,
        'calinski_harabasz_score': ch_scores,
        'davies_bouldin_score': db_scores
    }

    return results

def perform_clustering(X_scaled, n_clusters: int = 3) -> tuple:
    """
    Perform KMeans clustering.

    Args:
        X_scaled: Scaled feature matrix
        n_clusters: Number of clusters

    Returns:
        Tuple of (cluster_labels, kmeans_model)
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    return cluster_labels, kmeans

def perform_pca(X_scaled, n_components: int = 2) -> tuple:
    """
    Perform PCA for dimensionality reduction.

    Args:
        X_scaled: Scaled feature matrix
        n_components: Number of principal components

    Returns:
        Tuple of (X_pca, pca_model)
    """
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca, pca

def plot_elbow_method(results: dict, save_path: str = None):
    """
    Plot elbow method and silhouette scores.

    Args:
        results: Dictionary from find_optimal_clusters
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 5))

    # Inertia (Elbow)
    plt.subplot(1, 2, 1)
    plt.plot(results['K'], results['inertia'], 'bo-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)

    # Silhouette score
    plt.subplot(1, 2, 2)
    plt.plot(results['K'], results['silhouette_score'], 'go-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Optimal k')
    plt.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_clusters_pca(X_pca: np.ndarray, cluster_labels: np.ndarray, save_path: str = None):
    """
    Plot clusters in PCA space.

    Args:
        X_pca: PCA-transformed data
        cluster_labels: Cluster labels
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6)
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('Patient Segments (PCA)')
    plt.colorbar(scatter, label='Cluster')
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_cluster_results(df: pd.DataFrame, cluster_labels: np.ndarray,
                         pca_model, X_pca: np.ndarray,
                         output_dir: str = "data/processed_data/"):
    """
    Save clustering results.

    Args:
        df: Original patient features DataFrame
        cluster_labels: Cluster labels from clustering
        pca_model: Fitted PCA model
        X_pca: PCA-transformed data
        output_dir: Directory to save results
    """
    # Add cluster labels to original dataframe
    df_with_clusters = df.copy()
    df_with_clusters['cluster'] = cluster_labels

    # Add PCA components
    df_with_clusters['pca_1'] = X_pca[:, 0]
    df_with_clusters['pca_2'] = X_pca[:, 1] if X_pca.shape[1] > 1 else 0

    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "patient_segments.csv")
    df_with_clusters.to_csv(csv_path, index=False)

    # Save clustering model and PCA model? For simplicity, we'll just save the data
    # In a real deployment, we would save the models using joblib or pickle

    # Create a summary report
    cluster_summary = df_with_clusters.groupby('cluster').agg({
        'patient_id': 'count',
        'age': 'mean',
        'adherence_rate': 'mean',
        'discontinued': 'mean',
        'comorbidity_count': 'mean'
    }).rename(columns={
        'patient_id': 'n_patients',
        'age': 'avg_age',
        'adherence_rate': 'avg_adherence_rate',
        'discontinued': 'discontinuation_rate',
        'comorbidity_count': 'avg_comorbidity_count'
    })

    # Convert to dictionary for JSON serialization
    summary_dict = cluster_summary.reset_index().to_dict(orient='records')

    report = {
        'n_clusters': len(np.unique(cluster_labels)),
        'cluster_summary': summary_dict,
        'pca_explained_variance_ratio': pca_model.explained_variance_ratio_.tolist()
    }

    json_path = os.path.join(output_dir, "clustering_summary.json")
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"Clustering results saved to {output_dir}")
    print(f"  - Patient segments: {csv_path}")
    print(f"  - Clustering summary: {json_path}")

def main():
    """
    Main function to run patient segmentation.
    """
    print("Loading patient features...")
    df = load_patient_features()

    print("Preparing features for clustering...")
    X_scaled, feature_names, scaler = prepare_features_for_clustering(df)
    print(f"  - Number of features: {len(feature_names)}")
    print(f"  - Feature names: {feature_names}")

    print("\nFinding optimal number of clusters...")
    results = find_optimal_clusters(X_scaled, max_clusters=10)

    # Plot elbow method
    os.makedirs("reports/figures", exist_ok=True)
    plot_elbow_method(results, save_path="reports/figures/elbow_method.png")
    print("  - Elbow method plot saved to reports/figures/elbow_method.png")

    # Based on the results, we'll choose a number of clusters (for now, let's use 3)
    # In practice, we would look at the elbow and silhouette scores
    n_clusters = 3
    print(f"\nPerforming clustering with {n_clusters} clusters...")
    cluster_labels, kmeans_model = perform_clustering(X_scaled, n_clusters=n_clusters)

    # Evaluate clustering
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    print(f"  - Silhouette Score: {silhouette_avg:.3f}")

    print("\nPerforming PCA for visualization...")
    X_pca, pca_model = perform_pca(X_scaled, n_components=2)

    # Plot clusters
    plot_clusters_pca(X_pca, cluster_labels, save_path="reports/figures/clusters_pca.png")
    print("  - PCA cluster plot saved to reports/figures/clusters_pca.png")

    print("\nSaving clustering results...")
    save_cluster_results(df, cluster_labels, pca_model, X_pca, output_dir="data/processed_data/")

    print("\n=== Segmentation Complete ===")

if __name__ == "__main__":
    main()