import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, pdist
from sklearn.metrics import silhouette_score

# Import data
df = pd.read_csv('./resources/Diabetes.csv')

# Normalize data using min-max feature scaling
for column in df:
    df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())

# View normalized data
print(df)

x = df.iloc[:, :8]
y = df['class']

sc = StandardScaler()
sc.fit(x)
x = sc.transform(x)
df['class'] = df['class'].astype(np.int64)

# Initialize k-means
model = KMeans(n_clusters=2, random_state=11)
model.fit(x)
print(model.labels_)

# Predict class
df['predict_class'] = np.choose(model.labels_, [0, 1]).astype(np.int64)

# Evaluate model performance
print("Accuracy: ", metrics.accuracy_score(df['class'], df['predict_class']))
print("Classification Report: ", metrics.classification_report(df['class'], df['predict_class']))

# Determine ideal number of clusters using elbow method
K = range(1, 10)
KM = [KMeans(n_clusters=k).fit(x) for k in K]
centroids = [k.cluster_centers_ for k in KM]

D_k = [cdist(x, cent, 'euclidean') for cent in centroids]
cIdx = [np.argmin(D, axis=1) for D in D_k]
dist = [np.min(D, axis=1) for D in D_k]
avgWithinSS = [sum(d)/x.shape[0] for d in dist]

wcss = [sum(d**2) for d in dist]
tss = sum(pdist(x)**2)/x.shape[0]
bss = tss-wcss
varExplained = bss/tss*100
kIdx = 2

# Plot results
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(K, avgWithinSS, 'b*-')
plt.plot(K[kIdx], avgWithinSS[kIdx], marker='o', markersize=12, markeredgewidth=2, markeredgecolor='r',
         markerfacecolor='None')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Average within-cluster sum of squares')
plt.title('Elbow for k-means clustering')

plt.tight_layout()
plt.savefig('./resources/kmeans_elbow.png')
plt.show()

# Determine ideal number of clusters using silhouette method
score = []
for n_clusters in range(2, 10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=11)
    kmeans.fit(x)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    score.append(silhouette_score(x, labels, metric='euclidean'))

# Plot results
plt.figure(figsize=(10, 10))
plt.plot(score)
plt.grid(True)
plt.ylabel('Silhouette Score')
plt.xlabel('k')
plt.title('Silhouette for k-means')
plt.savefig('./resources/kmeans_silhouette.png')
plt.show()

# Create covariance matrix
cov_mat = np.cov(x.T)
print('Covariance Matrix \n%s' % cov_mat)

eig_vals, eig_vecs = np.linalg.eig(cov_mat)
print('Eigenvectors \n%s' % eig_vecs)
print('Eigenvalues \n%s' % eig_vals)

tot = sum(eig_vals)
var_exp = [(i/tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
print('Cumulative Variance Explained\n', cum_var_exp)

# Plot results
plt.figure(figsize=(6, 8))
plt.bar(range(8), var_exp, alpha=0.5, align='center', label='Individual Explained Variance')
plt.step(range(8), cum_var_exp, where='mid', label='Cumulative Explained Variance')
plt.ylabel('Explained Variance Ratio')
plt.xlabel('Principal Components')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('./resources/eigen.png')
plt.show()