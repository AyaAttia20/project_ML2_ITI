
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score,mutual_info_score, adjusted_rand_score



def evaluate_clustering(X, n_clusters):
    
       
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(X)

        silhouette = silhouette_score(X, kmeans.labels_)
        db_index = davies_bouldin_score(X, kmeans.labels_)
        ch_index = calinski_harabasz_score(X, kmeans.labels_)
        mutual_info = mutual_info_score(kmeans.labels_, kmeans.labels_)
        adjusted_rand = adjusted_rand_score(kmeans.labels_, kmeans.labels_)
        
        return silhouette,db_index,ch_index,mutual_info,adjusted_rand
   