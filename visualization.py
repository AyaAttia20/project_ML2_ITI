
import matplotlib.pyplot as plt
import  numpy as np


def plot_pca(per_var):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(per_var) + 1), per_var.cumsum(), marker="o", linestyle="--")
    plt.grid()
    plt.ylabel("Cumulative Percentage of Explained Variance")
    plt.xlabel("Number of Components")
    plt.title("Explained Variance by Component")    
    plt.show()



def plot_elbow(inertias):
    # plt.plot((1,11), inertias, marker='o')
    plt.plot(range(1, len(inertias) + 1), inertias, marker='o')
    plt.title('Elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.grid()
    plt.show()



def plot_clusters(pca_features, clusters):
        
    plt.scatter(pca_features[:, 0], pca_features[:, 1], c=clusters, cmap="plasma")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("K-Means Clustering using PCA")
    plt.colorbar(label="Cluster")
    plt.show()    


def plot_solutt(k_values, silhouette_scores):
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, silhouette_scores)
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score vs. Number of Clusters")
    plt.show()


def plot_tsne(X_tsne, clusters):

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=clusters.astype(int), cmap='tab10', s=1)
    plt.legend(*scatter.legend_elements(), title="Tsnee Clusters")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.colorbar(scatter, label="Cluster")
    plt.title('t-SNE ')
    plt.show()



