import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os

print("=" * 50)
print("PHASE 3 — K-MEANS CLUSTERING")
print("=" * 50)

# ── Load cleaned data ─────────────────────────────
customers = pd.read_csv('data/cleaned.csv')
print(f"\nLoaded {customers.shape[0]} customers")

features = ['total_spend', 'num_orders', 'avg_order_value']
X = customers[features]

# ── Scale ─────────────────────────────────────────
# K-Means uses distance — unscaled features let high-value
# columns dominate. Scaling puts everything on equal footing.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("\nFeatures scaled with StandardScaler")

# ── Elbow method ──────────────────────────────────
inertias = []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(list(K_range), inertias, 'bo-')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method — Finding Optimal K')
plt.tight_layout()
os.makedirs('outputs', exist_ok=True)
plt.savefig('outputs/elbow_plot.png', dpi=150)
plt.close()
print("\nElbow plot saved to outputs/elbow_plot.png")

# ── Fit final model (K=4 is typical for this dataset) ──
K = 4
km = KMeans(n_clusters=K, random_state=42, n_init=10)
customers['cluster'] = km.fit_predict(X_scaled)
print(f"\nFitted K-Means with K={K}")
print(f"\nCluster sizes:\n{customers['cluster'].value_counts().sort_index()}")

# ── Profile clusters ──────────────────────────────
profile = customers.groupby('cluster')[features].mean().round(2)
print(f"\nCluster profiles:\n{profile}")

# Name clusters based on their profile
cluster_names = {}
for c in range(K):
    row = profile.loc[c]
    if row['total_spend'] > profile['total_spend'].median() and row['num_orders'] > profile['num_orders'].median():
        cluster_names[c] = 'High-value loyal'
    elif row['total_spend'] > profile['total_spend'].median():
        cluster_names[c] = 'High spend, low frequency'
    elif row['num_orders'] > profile['num_orders'].median():
        cluster_names[c] = 'Frequent budget buyer'
    else:
        cluster_names[c] = 'Occasional low spender'

customers['segment'] = customers['cluster'].map(cluster_names)
print(f"\nSegment names:\n{customers['segment'].value_counts()}")

# ── PCA visualization ─────────────────────────────
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(9, 6))
colors = ['#1D9E75', '#7F77DD', '#D85A30', '#BA7517']
for c in range(K):
    mask = customers['cluster'] == c
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                label=cluster_names[c], alpha=0.5,
                s=20, color=colors[c])
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('Customer Segments — PCA View')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/clusters_pca.png', dpi=150)
plt.close()
print("PCA cluster plot saved to outputs/clusters_pca.png")

# ── Save ──────────────────────────────────────────
customers.to_csv('data/clustered.csv', index=False)
print("\nSaved to data/clustered.csv")
