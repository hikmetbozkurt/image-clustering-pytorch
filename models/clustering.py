import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import pickle
import logging
from typing import Dict, Tuple, Optional
from config import Config

logger = logging.getLogger(__name__)


class ProductClusteringModel:
    def __init__(self, n_clusters: int = 5, use_pca: bool = True, use_anomaly_detection: bool = True):
        self.n_clusters = n_clusters
        self.use_pca = use_pca
        self.use_anomaly_detection = use_anomaly_detection
        
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=Config.PCA_COMPONENTS) if use_pca else None
        self.isolation_forest = IsolationForest(
            contamination=Config.CONTAMINATION,
            random_state=Config.RANDOM_STATE
        ) if use_anomaly_detection else None
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=Config.RANDOM_STATE,
            n_init=10
        )
        
        self.features_scaled: Optional[np.ndarray] = None
        self.features_reduced: Optional[np.ndarray] = None
        self.anomaly_mask: Optional[np.ndarray] = None
        self.cluster_labels: Optional[np.ndarray] = None
        self.metrics: Dict[str, float] = {}
        
    def preprocess(self, features: np.ndarray) -> np.ndarray:
        logger.info("="*50)
        logger.info("PREPROCESSING")
        logger.info("="*50)
        logger.info(f"Original shape: {features.shape}")
        
        self.features_scaled = self.scaler.fit_transform(features)
        logger.info("Features scaled using StandardScaler")
        
        if self.use_pca:
            self.features_reduced = self.pca.fit_transform(self.features_scaled)
            explained_var = np.sum(self.pca.explained_variance_ratio_)
            logger.info(f"PCA reduced dimensions: {features.shape[1]} → {Config.PCA_COMPONENTS}")
            logger.info(f"Explained variance: {explained_var:.2%}")
        else:
            self.features_reduced = self.features_scaled
            logger.info("PCA skipped")
        
        return self.features_reduced
    
    def detect_anomalies(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.use_anomaly_detection:
            logger.info("Anomaly detection skipped")
            return features, np.ones(len(features), dtype=bool)
        
        logger.info("="*50)
        logger.info("ANOMALY DETECTION")
        logger.info("="*50)
        
        predictions = self.isolation_forest.fit_predict(features)
        self.anomaly_mask = predictions == 1
        
        n_anomalies = np.sum(~self.anomaly_mask)
        logger.info(f"Detected {n_anomalies} anomalies ({n_anomalies/len(features)*100:.1f}%)")
        logger.info(f"Clean samples: {np.sum(self.anomaly_mask)}")
        
        return features[self.anomaly_mask], self.anomaly_mask
    
    def cluster(self, features: np.ndarray) -> np.ndarray:
        logger.info("="*50)
        logger.info("CLUSTERING")
        logger.info("="*50)
        logger.info(f"Running K-means with {self.n_clusters} clusters...")
        
        self.cluster_labels = self.kmeans.fit_predict(features)
        
        unique, counts = np.unique(self.cluster_labels, return_counts=True)
        logger.info("Clustering complete")
        logger.info("Cluster distribution:")
        for cluster_id, count in zip(unique, counts):
            logger.info(f"  Cluster {cluster_id}: {count} samples ({count/len(features)*100:.1f}%)")
        
        return self.cluster_labels
    
    def evaluate(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        logger.info("="*50)
        logger.info("EVALUATION")
        logger.info("="*50)
        
        silhouette = silhouette_score(features, labels)
        self.metrics['silhouette_score'] = silhouette
        
        davies_bouldin = davies_bouldin_score(features, labels)
        self.metrics['davies_bouldin_score'] = davies_bouldin
        
        self.metrics['inertia'] = self.kmeans.inertia_
        
        logger.info(f"Silhouette Score: {silhouette:.4f} (higher is better)")
        logger.info(f"Davies-Bouldin Index: {davies_bouldin:.4f} (lower is better)")
        logger.info(f"Inertia: {self.kmeans.inertia_:.2f}")
        
        return self.metrics
    
    def fit(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        processed_features = self.preprocess(features)
        clean_features, mask = self.detect_anomalies(processed_features)
        labels = self.cluster(clean_features)
        self.evaluate(clean_features, labels)
        
        return labels, mask
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        features_scaled = self.scaler.transform(features)
        
        if self.use_pca:
            features_reduced = self.pca.transform(features_scaled)
        else:
            features_reduced = features_scaled
        
        labels = self.kmeans.predict(features_reduced)
        return labels
    
    def save_model(self, filepath: str) -> None:
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Model saved to {filepath}")
    
    @staticmethod
    def load_model(filepath: str) -> 'ProductClusteringModel':
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from {filepath}")
        return model
    
    def get_cluster_centers(self) -> np.ndarray:
        if self.use_pca:
            centers_scaled = self.pca.inverse_transform(self.kmeans.cluster_centers_)
            centers = self.scaler.inverse_transform(centers_scaled)
        else:
            centers = self.scaler.inverse_transform(self.kmeans.cluster_centers_)
        return centers


if __name__ == "__main__":
    import os
    
    logger.info("Testing ProductClusteringModel...")
    
    features = np.load(Config.FEATURES_FILE)
    logger.info(f"Loaded features: {features.shape}")
    
    model = ProductClusteringModel(
        n_clusters=Config.N_CLUSTERS,
        use_pca=Config.USE_PCA,
        use_anomaly_detection=Config.USE_ANOMALY_DETECTION
    )
    
    labels, mask = model.fit(features)
    
    logger.info("="*50)
    logger.info("RESULTS SUMMARY")
    logger.info("="*50)
    logger.info(f"Total samples: {len(features)}")
    logger.info(f"Clean samples: {np.sum(mask)}")
    logger.info(f"Anomalies: {np.sum(~mask)}")
    logger.info(f"Silhouette Score: {model.metrics['silhouette_score']:.4f}")
    
    os.makedirs(Config.RESULTS_PATH, exist_ok=True)
    model.save_model(f"{Config.RESULTS_PATH}/clustering_model.pkl")