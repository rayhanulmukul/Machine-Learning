import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

sns.scatterplot(x=df[feature_names[0]], y=df[feature_names[1]], hue=df['target'], palette='Set1')
plt.show()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(2)
X_pca = pca.fit_transform(X_scaled)
print(pca.explained_variance_)

sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y, palette='viridis')
plt.show()

pca_full = PCA().fit(X_scaled)
plt.plot(np.cumsum(pca_full.explained_variance_ratio_), marker='o')
plt.show()

pca = PCA(3)
X_scaled = pca.fit_transform(X_scaled)

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(projection='3d')
scatter = ax.scatter(X_scaled[:,0], X_scaled[:,1], X_scaled[:,2], c=y, cmap='Set1')
plt.show()
