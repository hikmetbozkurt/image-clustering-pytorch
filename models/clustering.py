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

        self.requested_pca_components = Config.PCA_COMPONENTS
        self.pca: Optional[PCA] = None

        self.isolation_forest: Optional[IsolationForest] = IsolationForest(
            contamination=Config.CONTAMINATION,
            random_state=Config.RANDOM_STATE
        ) if use_anomaly_detection else None

        self.kmeans: Optional[KMeans] = None

        self.features_scaled: Optional[np.ndarray] = None
        self.features_reduced: Optional[np.ndarray] = None
        self.anomaly_mask: Optional[np.ndarray] = None
        self.cluster_labels: Optional[np.ndarray] = None
        self.metrics: Dict[str, float] = {}

    @staticmethod
    def _safe_unique_count(labels: np.ndarray) -> int:
        return len(np.unique(labels)) if labels is not None and len(labels) > 0 else 0

    def preprocess(self, features: np.ndarray) -> np.ndarray:
        logger.info("=" * 50)
        logger.info("PREPROCESSING")
        logger.info("=" * 50)
        logger.info(f"Original shape: {features.shape}")

        # Scale
        self.features_scaled = self.scaler.fit_transform(features)
        logger.info("Features scaled using StandardScaler")

        # PCA (robust)
        if self.use_pca:
            n_samples, n_features = self.features_scaled.shape
            # Max allowed components for sklearn PCA with svd_solver='full':
            # 0 < n_components <= min(n_samples, n_features)
            # Empirically, use n_samples-1 to avoid edge SVD issues on tiny sets.
            max_allowed = max(0, min(n_samples - 1, n_features))
            n_components = min(self.requested_pca_components, max_allowed)

            if n_components >= 2:
                self.pca = PCA(n_components=n_components, svd_solver="full")
                self.features_reduced = self.pca.fit_transform(self.features_scaled)
                explained_var = float(np.sum(self.pca.explained_variance_ratio_))
                logger.info(f"PCA reduced dimensions: {n_features} → {n_components}")
                logger.info(f"Explained variance: {explained_var:.2%}")
            else:
                # Not enough samples/features to do meaningful PCA → skip
                self.pca = None
                self.features_reduced = self.features_scaled
                if max_allowed == 0:
                    logger.info("PCA skipped (dataset too small).")
                else:
                    logger.info(f"PCA skipped (requested={self.requested_pca_components}, "
                                f"max_allowed={max_allowed}).")
        else:
            self.pca = None
            self.features_reduced = self.features_scaled
            logger.info("PCA skipped")

        return self.features_reduced

    def detect_anomalies(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.use_pca:
            pass

        if not self.use_anomaly_detection or self.isolation_forest is None:
            logger.info("Anomaly detection skipped")
            return features, np.ones(len(features), dtype=bool)

        logger.info("=" * 50)
        logger.info("ANOMALY DETECTION")
        logger.info("=" * 50)

        n_samples = len(features)

        if n_samples < 20:
            logger.info(f"Anomaly detection skipped (too few samples: {n_samples} < 20)")
            mask = np.ones(n_samples, dtype=bool)
            return features, mask

        predictions = self.isolation_forest.fit_predict(features)
        self.anomaly_mask = (predictions == 1)

        n_anomalies = int(np.sum(~self.anomaly_mask))
        logger.info(f"Detected {n_anomalies} anomalies ({n_anomalies / max(1, n_samples) * 100:.1f}%)")
        logger.info(f"Clean samples: {int(np.sum(self.anomaly_mask))}")

        if np.sum(self.anomaly_mask) == 0:
            logger.info("All samples flagged as anomalies; disabling anomaly filtering.")
            self.anomaly_mask = np.ones(n_samples, dtype=bool)

        return features[self.anomaly_mask], self.anomaly_mask

    def cluster(self, features: np.ndarray) -> np.ndarray:
        logger.info("=" * 50)
        logger.info("CLUSTERING")
        logger.info("=" * 50)

        n_samples = len(features)
        if n_samples == 0:
            logger.info("No samples available to cluster. Returning empty labels.")
            self.cluster_labels = np.array([], dtype=int)
            return self.cluster_labels

        effective_k = min(self.n_clusters, n_samples)

        if effective_k < 2:
            logger.info(f"Too few samples to form multiple clusters (n={n_samples}). "
                        f"Assigning all to cluster 0.")
            self.kmeans = KMeans(n_clusters=1, random_state=Config.RANDOM_STATE, n_init=10)
            self.cluster_labels = self.kmeans.fit_predict(features)
        else:
            logger.info(f"Running K-means with {effective_k} clusters...")
            self.kmeans = KMeans(n_clusters=effective_k, random_state=Config.RANDOM_STATE, n_init=10)
            self.cluster_labels = self.kmeans.fit_predict(features)

        unique, counts = np.unique(self.cluster_labels, return_counts=True)
        logger.info("Clustering complete")
        logger.info("Cluster distribution:")
        for cluster_id, count in zip(unique, counts):
            logger.info(f"  Cluster {cluster_id}: {count} samples ({count / n_samples * 100:.1f}%)")

        return self.cluster_labels

    def evaluate(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        logger.info("=" * 50)
        logger.info("EVALUATION")
        logger.info("=" * 50)

        self.metrics.clear()
        n = len(features)
        n_labels = self._safe_unique_count(labels)

        # Silhouette: needs at least 2 clusters and n > 1, and preferably non-singleton clusters
        if n > 1 and n_labels >= 2:
            try:
                self.metrics['silhouette_score'] = float(silhouette_score(features, labels))
                logger.info(f"Silhouette Score: {self.metrics['silhouette_score']:.4f} (higher is better)")
            except Exception as e:
                logger.info(f"Silhouette Score unavailable: {e}")
        else:
            logger.info("Silhouette Score skipped (needs at least 2 clusters and >1 sample).")

        if n_labels >= 2:
            try:
                self.metrics['davies_bouldin_score'] = float(davies_bouldin_score(features, labels))
                logger.info(f"Davies-Bouldin Index: {self.metrics['davies_bouldin_score']:.4f} "
                            f"(lower is better)")
            except Exception as e:
                logger.info(f"Davies-Bouldin Index unavailable: {e}")
        else:
            logger.info("Davies-Bouldin Index skipped (needs at least 2 clusters).")

        if self.kmeans is not None and hasattr(self.kmeans, "inertia_"):
            self.metrics['inertia'] = float(self.kmeans.inertia_)
            logger.info(f"Inertia: {self.metrics['inertia']:.2f}")

        return self.metrics

    def fit(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        processed_features = self.preprocess(features)
        clean_features, mask = self.detect_anomalies(processed_features)
        labels = self.cluster(clean_features)
        self.evaluate(clean_features, labels)
        return labels, mask

    def predict(self, features: np.ndarray) -> np.ndarray:
        features_scaled = self.scaler.transform(features)
        if self.use_pca and self.pca is not None:
            features_reduced = self.pca.transform(features_scaled)
        else:
            features_reduced = features_scaled

        if self.kmeans is None:
            raise RuntimeError("Model is not fitted. Call fit() before predict().")

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
        if self.kmeans is None:
            raise RuntimeError("Model is not fitted. No cluster centers available.")
        if self.use_pca and self.pca is not None:
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

    logger.info("=" * 50)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 50)
    logger.info(f"Total samples: {len(features)}")
    logger.info(f"Clean samples: {int(np.sum(mask))}")
    logger.info(f"Anomalies: {int(np.sum(~mask))}")
    if 'silhouette_score' in model.metrics:
        logger.info(f"Silhouette Score: {model.metrics['silhouette_score']:.4f}")

    os.makedirs(Config.RESULTS_PATH, exist_ok=True)
    model.save_model(f"{Config.RESULTS_PATH}/clustering_model.pkl")