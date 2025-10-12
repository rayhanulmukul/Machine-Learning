import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv('blob.csv')
display(df.head())

for col in df.columns:
    if df[col].dtype == object:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

if 'TrueCluster' in df.columns:
    X = df.drop('TrueCluster', axis=1).values
    y_true = df['TrueCluster'].values
else:
    X = df.values
    y_true = None

if X.shape[1] >= 2:
    plt.scatter(X[:,0], X[:,1], c=y_true if y_true is not None else 'gray', cmap='viridis')
    plt.show()

Kmeans = KMeans(3, random_state=42)
label_Kmeans = Kmeans.fit_predict(X)

plt.scatter(X[:,0], X[:,1], c=label_Kmeans, cmap='viridis')
plt.scatter(Kmeans.cluster_centers_[:,0], Kmeans.cluster_centers_[:,1], c='red', marker='X', label='Centers')
plt.legend()
plt.show()

Gaussianmixture = GaussianMixture(3, random_state=42)
label_GaussianMixture = Gaussianmixture.fit_predict(X)

plt.scatter(X[:,0], X[:,1], c=label_GaussianMixture, cmap='viridis')
plt.scatter(Gaussianmixture.means_[:,0], Gaussianmixture.means_[:,1], c='red', marker='X', label='Centers')
plt.legend()
plt.show()


features = df.columns.drop('TrueCluster', errors='ignore')

Kmeans_imp = np.std(Kmeans.cluster_centers_, axis=0)
Gaussianmixture_imp = np.std(Gaussianmixture.means_, axis=0)

imp = pd.DataFrame({'Feature': features, 'Importance': Kmeans_imp}).sort_values('Importance')

imp.plot.barh(x='Feature', y='Importance', legend=True)
plt.show()

imp = pd.DataFrame({'Feature': features, 'Importance': Gaussianmixture_imp}).sort_values('Importance')

imp.plot.barh(x='Feature', y='Importance', legend=True)
plt.show()