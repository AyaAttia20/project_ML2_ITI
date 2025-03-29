from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from evaluation import evaluate_clustering
from sklearn.manifold import TSNE



# def pca_fun(data,num_component):
#     pca = PCA(n_components=num_component)
#     pca_features = pca.fit_transform(data)
#     per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)

#     # plt.figure(figsize=(10, 6))
#     # plt.plot(range(1, len(per_var) + 1), per_var.cumsum(), marker="o", linestyle="--")
#     # plt.grid()
#     # plt.ylabel("Cumulative Percentage of Explained Variance")
#     # plt.xlabel("Number of Components")
#     # plt.title("Explained Variance by Component")
#     # plt.show()
    
#     return pca_features, pca, per_var


def pca_fun_cluster(data, num_component): 
        
    print("Dimensionality Reduction For Clustering")
    print("--------------------------------------------------------------------------------------")

    pca = PCA(n_components=num_component)
    pca_features = pca.fit_transform(data)
    per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
    
    return pca_features, pca, per_var

  


def anything():
    print("This is a placeholder function in the clustering module.")
    # You can add any functionality you want here.
    
def optimal_clusters(data):
    inertias = []

    for i in range(1,11):
        kmeans = KMeans(n_clusters=i)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)


    return inertias  



def kmeans_clustering(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters=kmeans.fit_predict(data)
    return clusters



def calculate_silhouette(pca_features):
    silhouette_scores = []
    k_range = range(2, 11)  

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(pca_features)
        silho_score,_,_,_,_=evaluate_clustering(pca_features, k)
        silhouette_scores.append(silho_score)

    return silhouette_scores    



def tsne_fun(pca_features, n_components):
    
    tsne = TSNE(n_components, perplexity=30, n_iter=1000, random_state=42)
    X_tsne = tsne.fit_transform(pca_features)
    
    return X_tsne