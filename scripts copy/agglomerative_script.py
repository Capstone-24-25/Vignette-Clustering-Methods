# agglomerative script
from sklearn.metrics import silhouette_score # type: ignore
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering 

import numpy as np # type: ignore
import pandas as pd # type: ignore
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
# wqre

# Read in data
df_transformed = pd.read_csv('./data/transformed_user_behavior_dataset.csv')
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_transformed)

# Select number of clusters and dendrogram
dend = shc.dendrogram(shc.linkage(X_scaled, method='ward')) # why use ward? 
plt.title("Dendrogram")
plt.xlabel("Data Points")
plt.show()

n_clusters = 6 
agglom = AgglomerativeClustering(n_clusters=n_clusters).fit(X_scaled) 
labels = agglom.labels_

# Visualization using PCA 
pca_2 = PCA(n_components=2)
pca_2_result = pca_2.fit_transform(X_scaled)

# centroids = agglom.cluster_centers_
# centroids_2 = pca_2.fit_transform(centroids)

viz = pd.DataFrame(pca_2_result)
viz['cluster_id'] = labels
viz = viz.rename(columns={0: "PC1", 1:"PC2"})

viz.plot.scatter('PC1', 'PC2', c=2, colormap='viridis')

#plt.scatter(centroids_2[:, 0], centroids_2[:, 1], marker='X', s=200, linewidths=1.5,
#                color='red', edgecolors="black")
plt.title('Clustering of Principal Components')
plt.savefig('agglom-PCA.png')
plt.show()
