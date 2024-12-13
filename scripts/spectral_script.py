# spectral_script.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler

def perform_spectral_clustering(filepath, n_clusters=5, n_neighbors=44):
    # Load data
    df_transformed = pd.read_csv(filepath)

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_transformed)

    # Initialize Spectral Clustering
    spectral = SpectralClustering(
        n_clusters=n_clusters,
        affinity='nearest_neighbors',
        n_neighbors=n_neighbors,
        assign_labels='kmeans',
        random_state=42
    )
    
    # Fit the model and predict cluster labels
    cluster_labels = spectral.fit_predict(X_scaled)

    # Add the cluster labels to the DataFrame
    df_transformed['Cluster'] = cluster_labels
    
    return df_transformed
