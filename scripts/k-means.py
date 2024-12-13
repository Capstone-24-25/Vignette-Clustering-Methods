# Import relevant packages
from sklearn.cluster import KMeans # type: ignore
from sklearn.metrics import silhouette_score # type: ignore
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np # type: ignore
import pandas as pd # type: ignore
import matplotlib.pyplot as plt

# Read in data
df_transformed = pd.read_csv('./data/transformed_user_behavior_dataset.csv')
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_transformed)

# PCA 
pca_2 = PCA(n_components=2)
pca_2_result = pca_2.fit_transform(X_scaled)
print(pca_2_result)
print(pca_2.explained_variance_ratio_)


#df_numeric = df_transformed[['App Usage Time (min/day)', 'Screen On Time (hours/day)', 'Battery Drain (mAh/day)', 'Number of Apps Installed', 'Data Usage (MB/day)']]

# Use elbow method to find optimal number of clusters
wcss = [] # average within cluster sum of squares
silhouette_scores = []
models = []

for i in range(1, 21):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0) # n_init: Number of times the k-means algorithm is run with different centroid seeds
    kmeans.fit(pca_2_result)
    wcss.append(kmeans.inertia_) # inertial is the wcss for a given number of clusters
    models.append(kmeans)

    if i > 1:
        labels_i = kmeans.labels_
        silhouette_avg_i = silhouette_score(pca_2_result, labels_i)
        silhouette_scores.append(silhouette_avg_i)
                             
plt.plot(range(1, 21), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.savefig('elbow.png')
plt.show()

print(silhouette_scores)
print(silhouette_scores.index(max(silhouette_scores)) + 1)

# Applying K-Means++ initialization
n_clusters = 8

print(f"Generating K-Means model with {n_clusters} clusters")
kmeans = KMeans(n_clusters=n_clusters, init='k-means++').fit(X_scaled)
labels = kmeans.labels_

# Visualization using PCA 
pca_2 = PCA(n_components=2)
pca_2_result = pca_2.fit_transform(X_scaled)

centroids = kmeans.cluster_centers_
centroids_2 = pca_2.fit_transform(centroids)

viz = pd.DataFrame(pca_2_result)
viz['cluster_id'] = labels
viz = viz.rename(columns={0: "PC1", 1:"PC2"})

viz.plot.scatter('PC1', 'PC2', c=2, colormap='viridis')

plt.scatter(centroids_2[:, 0], centroids_2[:, 1], marker='X', s=200, linewidths=1.5,
                color='red', edgecolors="black")
plt.title('Clustering of Principal Components')
plt.savefig('k-means-clusters-PCA.png')
plt.show()

# Silhouette score... Necessary???
silhouette_avg = silhouette_score(X_scaled, labels)
print(f"Silhouette score for K-means model: {silhouette_avg}")