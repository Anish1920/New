import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# Load dataset
file_path = "Warehouse_and_Retail_Sales.csv"
df = pd.read_csv(file_path)

# Handle missing values (fill with zero for sales data, mode for categorical)
df['SUPPLIER'].fillna(df['SUPPLIER'].mode()[0], inplace=True)
df['ITEM TYPE'].fillna(df['ITEM TYPE'].mode()[0], inplace=True)
df.fillna(0, inplace=True)

# Aggregate data by ITEM CODE (sum up sales data)
df_grouped = df.groupby(['ITEM CODE', 'ITEM DESCRIPTION', 'ITEM TYPE']).agg(
    {'RETAIL SALES': 'sum', 'RETAIL TRANSFERS': 'sum', 'WAREHOUSE SALES': 'sum'}).reset_index()

# Feature selection
features = ['RETAIL SALES', 'RETAIL TRANSFERS', 'WAREHOUSE SALES']
X = df_grouped[features]

# Normalize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means Clustering
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df_grouped['KMeans_Cluster'] = kmeans.fit_predict(X_scaled)

# Hierarchical Clustering
linkage_matrix = linkage(X_scaled, method='ward')
df_grouped['Hierarchical_Cluster'] = fcluster(linkage_matrix, 5, criterion='maxclust')

# DBSCAN Clustering
dbscan = DBSCAN(eps=1.5, min_samples=5)
df_grouped['DBSCAN_Cluster'] = dbscan.fit_predict(X_scaled)

# Save results
df_grouped.to_csv("Clustered_Products.csv", index=False)

# Plot K-Means Clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df_grouped['RETAIL SALES'], y=df_grouped['WAREHOUSE SALES'], hue=df_grouped['KMeans_Cluster'], palette='viridis')
plt.title("K-Means Clustering (Retail Sales vs Warehouse Sales)")
plt.xlabel("Retail Sales")
plt.ylabel("Warehouse Sales")
plt.legend(title="Cluster")
plt.show()

# Dendrogram for Hierarchical Clustering
plt.figure(figsize=(12, 6))
dendrogram(linkage_matrix, truncate_mode='level', p=10)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Products")
plt.ylabel("Distance")
plt.show()

print("Clustering complete. Results saved in 'Clustered_Products.csv'.")
