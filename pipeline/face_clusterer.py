import logging
from typing import Dict, List
import numpy as np

logger = logging.getLogger(__name__)

class FaceClusterer:
    """Face clustering using HDBSCAN."""
    def __init__(self, min_cluster_size: int = 3, min_samples: int = 2, metric: str = 'euclidean'):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.metric = metric
        logger.info(f"Initialized FaceClusterer with min_cluster_size={min_cluster_size}")

    def cluster(self, embeddings: np.ndarray) -> np.ndarray:
        """Clusters face embeddings. Returns labels array (-1 for noise)."""
        try:
            import hdbscan
            if len(embeddings) == 0:
                return np.array([])
            if len(embeddings) < self.min_cluster_size:
                return np.array([-1] * len(embeddings))
                
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                metric=self.metric,
                cluster_selection_method='eom'
            )
            labels = clusterer.fit_predict(embeddings)
            return labels
        except ImportError:
            logger.error("hdbscan not installed.")
            return np.array([-1] * len(embeddings))
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            return np.array([-1] * len(embeddings))

    def get_cluster_centers(self, embeddings: np.ndarray, labels: np.ndarray) -> Dict[int, np.ndarray]:
        """Calculates centroids for each cluster."""
        centers = {}
        unique_labels = set(labels)
        for label in unique_labels:
            if label == -1:
                continue
            mask = (labels == label)
            cluster_embeds = embeddings[mask]
            centers[label] = np.mean(cluster_embeds, axis=0)
        return centers

    def merge_clusters(self, embeddings: np.ndarray, labels: np.ndarray, threshold: float = 0.45) -> np.ndarray:
        """Merges similar clusters post-HDBSCAN based on cosine distance."""
        centers = self.get_cluster_centers(embeddings, labels)
        if not centers:
            return labels
            
        new_labels = labels.copy()
        unique_labels = list(centers.keys())
        
        merges = {l: l for l in unique_labels}
        
        for i in range(len(unique_labels)):
            for j in range(i + 1, len(unique_labels)):
                l1 = unique_labels[i]
                l2 = unique_labels[j]
                
                c1 = centers[l1]
                c2 = centers[l2]
                
                c1_norm = c1 / (np.linalg.norm(c1) + 1e-10)
                c2_norm = c2 / (np.linalg.norm(c2) + 1e-10)
                
                similarity = np.dot(c1_norm, c2_norm)
                distance = 1.0 - similarity
                
                if distance < threshold:
                    root1 = l1
                    while merges[root1] != root1:
                        root1 = merges[root1]
                    root2 = l2
                    while merges[root2] != root2:
                        root2 = merges[root2]
                        
                    if root1 != root2:
                        merges[root2] = root1
                        
        for i in range(len(new_labels)):
            if new_labels[i] != -1:
                l = new_labels[i]
                root = l
                while merges[root] != root:
                    root = merges[root]
                new_labels[i] = root
                
        return new_labels
