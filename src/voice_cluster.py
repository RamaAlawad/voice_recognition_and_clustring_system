"""
Voice Clustering Module
Handles speaker clustering using extracted features
"""

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
import pickle

class VoiceClusterer:
    def __init__(self, method='agglomerative'):
        """
        Initialize voice clusterer
        
        Args:
            method: Clustering method ('agglomerative' or 'kmeans')
        """
        self.method = method
        self.clusterer = None
        self.cluster_labels = None
        self.features = None
        
    def fit_predict(self, features, n_clusters=None):
        """
        Perform clustering on voice features
        
        Args:
            features: Array of voice features
            n_clusters: Number of clusters (if None, will estimate)
            
        Returns:
            numpy.ndarray: Cluster labels
        """
        self.features = np.array(features)
        
        # Estimate number of clusters if not provided
        if n_clusters is None:
            n_clusters = self._estimate_clusters()
        
        # Choose clustering algorithm
        if self.method == 'agglomerative':
            self.clusterer = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage='ward'
            )
        elif self.method == 'kmeans':
            self.clusterer = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=10
            )
        else:
            raise ValueError("Method must be 'agglomerative' or 'kmeans'")
        
        # Fit and predict
        self.cluster_labels = self.clusterer.fit_predict(self.features)
        
        return self.cluster_labels
    
    def _estimate_clusters(self):
        """
        Estimate optimal number of clusters using silhouette score
        
        Returns:
            int: Estimated number of clusters
        """
        if len(self.features) < 4:
            return 2
        
        max_clusters = min(8, len(self.features) // 2)
        best_score = -1
        best_k = 2
        
        for k in range(2, max_clusters + 1):
            if self.method == 'agglomerative':
                clusterer = AgglomerativeClustering(n_clusters=k, linkage='ward')
            else:
                clusterer = KMeans(n_clusters=k, random_state=42, n_init=10)
            
            labels = clusterer.fit_predict(self.features)
            score = silhouette_score(self.features, labels)
            
            if score > best_score:
                best_score = score
                best_k = k
        
        print(f"Estimated {best_k} clusters (silhouette score: {best_score:.3f})")
        return best_k
    
    def get_cluster_info(self):
        """
        Get information about clusters
        
        Returns:
            dict: Cluster information
        """
        if self.cluster_labels is None:
            return None
        
        unique_labels = np.unique(self.cluster_labels)
        cluster_info = {}
        
        for label in unique_labels:
            mask = self.cluster_labels == label
            cluster_info[f"Cluster_{label}"] = {
                'count': np.sum(mask),
                'indices': np.where(mask)[0].tolist()
            }
        
        return cluster_info
    
    def save_model(self, filepath):
        """Save the trained clusterer"""
        model_data = {
            'clusterer': self.clusterer,
            'cluster_labels': self.cluster_labels,
            'features': self.features,
            'method': self.method
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained clusterer"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.clusterer = model_data['clusterer']
        self.cluster_labels = model_data['cluster_labels']
        self.features = model_data['features']
        self.method = model_data['method']
        
        print(f"Model loaded from {filepath}")

