# Clustering
🍷 Wine Dataset Clustering Project
This project applies unsupervised machine learning techniques to the Wine dataset from sklearn.datasets. It demonstrates dimensionality reduction using PCA and clustering using KMeans, Agglomerative Clustering, and DBSCAN. The results are evaluated using the Silhouette Score.

📂 Dataset
Source: sklearn.datasets.load_wine()

Description: Contains the results of a chemical analysis of wines grown in the same region in Italy. The analysis determined the quantities of 13 constituents found in each of the three types of wines.

🛠️ Technologies Used
Python

Scikit-learn

Pandas

NumPy

Matplotlib

Seaborn

📊 Methods Applied
1. Data Preprocessing
Checked for missing values

Standardized features using StandardScaler

Applied PCA to reduce dimensionality to 3 principal components

2. Clustering Algorithms
Algorithm	Evaluation Metric
KMeans (with Elbow + Silhouette)	✅
Agglomerative Clustering (Dendrogram)	✅
DBSCAN (density-based)	✅

3. Evaluation
Used Silhouette Score to evaluate clustering performance

Plotted clusters and dendrograms for visual inspection

📈 Visualizations
Elbow plot to determine optimal k for KMeans

Silhouette scores comparison across models

2D scatter plots of PCA components colored by cluster

Dendrogram for hierarchical clustering

📁 Files
Wine_Clustering_Project.py – Core implementation script

Wine_Clustering_Scores.csv – Comparison of silhouette scores

README.md – Project documentation

📌 Results
Clustering Algorithm	Silhouette Score
KMeans	~0.28 (for 3 clusters)
Agglomerative	~0.27
DBSCAN	Varies (depends on eps and noise points)

🚀 How to Run
bash
Copy
Edit
pip install -r requirements.txt
python Wine_Clustering_Project.py
✅ Future Improvements
Tune DBSCAN parameters (eps, min_samples)

Try Gaussian Mixture Models

Deploy as an interactive Streamlit app
