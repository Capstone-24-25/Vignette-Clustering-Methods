# spectral script
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score

df_transformed = pd.read_csv('data/transformed_user_behavior_dataset.csv')

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_transformed)

n_clusters = 5

spectral = SpectralClustering(
    n_clusters=n_clusters,
    affinity='nearest_neighbors',
    n_neighbors=44, 
    assign_labels='kmeans',
    random_state=42
)
# Fit the model and predict cluster labels
cluster_labels = spectral.fit_predict(X_scaled)

# Add the cluster labels to the DataFrame
df_transformed['Cluster'] = cluster_labels


silhouette_avg = silhouette_score(X_scaled, cluster_labels)
print(f"Silhouette score for Spectral model: {silhouette_avg}")

print(df_transformed.head(30))

