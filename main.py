# Importing necessary libraries
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the dataset
mall_customers_df = pd.read_csv("Mall_Customers.csv")

# Preprocessing
# Drop irrelevant columns
mall_customers_df.drop(columns=['CustomerID', 'Gender'], inplace=True)

# Feature scaling
scaler = StandardScaler()
mall_customers_scaled = scaler.fit_transform(mall_customers_df)

# Determine the optimal number of clusters (K)
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(mall_customers_scaled)
    inertia.append(kmeans.inertia_)

# Plot the elbow method graph
plt.plot(range(1, 11), inertia)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

# Based on the elbow method, choose the optimal number of clusters (K)
optimal_k = 5

# Apply KMeans clustering
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(mall_customers_scaled)

# Add cluster labels to the original dataset
mall_customers_df['Cluster'] = kmeans.labels_

# Interpretation
# Analyze the characteristics of customers in each cluster
cluster_means = mall_customers_df.groupby('Cluster').mean()
print(cluster_means)