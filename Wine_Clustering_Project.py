# Wine Dataset Clustering Project
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.datasets import load_wine
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = load_wine()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows',None)
print(df.head()) #starting 5 rows
print(df.shape)#shape

print(df.info())#info

print(df.isna().sum()) #null values

print(df.describe().T)#5 point summary


#visualizing using pairplot using 4 random columns
plt.figure(figsize=(20,12))
sns.pairplot(df[['alcalinity_of_ash','alcohol',"hue","total_phenols"]])
plt.show()

#dropping target column

df.drop(columns="target",inplace=True)

#scaling

from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

x=scaler.fit_transform(df)
x


#PCA - to reduce the dimentionality

from sklearn.decomposition import PCA

pca=PCA(n_components=3)
x_pca=pca.fit_transform(x)

x_pca#now we have only 3 columns


#kmeans with elbow method and Silhouette  score

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
wcss=[]#
for i in range(1,11):
  kmeans=KMeans(n_clusters=i)
  kmeans.fit(x_pca)
  kmeans_labels=kmeans.fit_predict(x_pca)
  wcss.append(kmeans.inertia_)

#elbow 

plt.figure(figsize=(10,5))
plt.plot(range(1,11),wcss,marker="o")
plt.xlabel("Number of clsuters")
plt.ylabel("WCSS")
plt.title("Kmeans cluster elbow method")
plt.grid(True)
plt.show()

##as per elbow method we can try n cluster for 3 or 6

kmeans3=KMeans(n_clusters=3)
kmeans_labels3=kmeans3.fit_predict(x_pca)
print("Silhouette score for 3 :",silhouette_score(x_pca,kmeans_labels3))

kmeans6=KMeans(n_clusters=6)
kmeans_labels6=kmeans6.fit_predict(x_pca)
print("Silhouette score for 6 :",silhouette_score(x_pca,kmeans_labels6))

#visualising the clusters

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title("Kmeans with 3 clusters")
sns.scatterplot(x=x_pca[:,0],y=x_pca[:,1],hue=kmeans_labels3)
sns.scatterplot(x=kmeans3.cluster_centers_[:,0],y=kmeans3.cluster_centers_[:,1],marker="X")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.subplot(1,2,2)
plt.title("Kmeans with 6 clusters")
sns.scatterplot(x=x_pca[:,0],y=x_pca[:,1],hue=kmeans_labels6)
sns.scatterplot(x=kmeans6.cluster_centers_[:,0],y=kmeans6.cluster_centers_[:,1],marker="X")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

#agglomarative clustering

from scipy.cluster.hierarchy import dendrogram,linkage

plt.figure(figsize=(10,8))
dendrogram(linkage(x_pca,method="ward"))
plt.show()


from sklearn.cluster import AgglomerativeClustering

ac=AgglomerativeClustering(n_clusters=3)
ac_labels=ac.fit_predict(x_pca)
print("silhouette_score :",silhouette_score(x_pca,ac_labels))

#visualizing the clusters

plt.figure(figsize=(10,5))
sns.scatterplot(x=x_pca[:,0],y=x_pca[:,1],hue=ac_labels)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

#DBSCAN

from sklearn.cluster import DBSCAN

dbscan=DBSCAN(eps=0.5,min_samples=3)
labels=dbscan.fit_predict(x_pca)

plt.scatter(x_pca[:,0],x_pca[:,1],c=labels)
plt.show()
print(list(labels).count(-1))

print(silhouette_score(x_pca,labels))

chart=pd.DataFrame(columns=["model","silhouette_score"])

kmean=pd.DataFrame({"model":"KMeans","silhouette_score":silhouette_score(x_pca,kmeans_labels3)},index=[0])
agglo=pd.DataFrame({"model":"Agglomerative","silhouette_score":silhouette_score(x_pca,ac_labels)},index=[0])
dbscan=pd.DataFrame({"model":"DBSCAN","silhouette_score":silhouette_score(x_pca,labels)},index=[0])
chart=pd.concat([chart,kmean,agglo,dbscan])
chart

plt.plot(chart.iloc[:,0],chart.iloc[:,1])
plt.show()
