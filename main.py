import os
import numpy as np
import argparse
import logging
from typing import Tuple, List
from config import Config
from models.feature_extractor import FeatureExtractor, get_image_paths
from models.clustering import ProductClusteringModel
from utils.visualization import create_all_visualizations, plot_elbow_curve, plot_silhouette_scores

logger = logging.getLogger(__name__)


def setup_directories() -> None:
    os.makedirs(Config.RAW_IMAGES_PATH, exist_ok=True)
    os.makedirs(Config.PROCESSED_DATA_PATH, exist_ok=True)
    os.makedirs(Config.RESULTS_PATH, exist_ok=True)
    os.makedirs(f"{Config.RESULTS_PATH}/plots", exist_ok=True)


def extract_features(force_recompute: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    if not force_recompute and os.path.exists(Config.FEATURES_FILE):
        logger.info("="*50)
        logger.info("LOADING EXISTING FEATURES")
        logger.info("="*50)
        features = np.load(Config.FEATURES_FILE)
        image_paths = np.load(Config.LABELS_FILE, allow_pickle=True)
        logger.info(f"Loaded features: {features.shape}")
        logger.info(f"Loaded {len(image_paths)} image paths")
        return features, image_paths
    
    logger.info("="*50)
    logger.info("FEATURE EXTRACTION")
    logger.info("="*50)
    
    image_paths = get_image_paths(Config.RAW_IMAGES_PATH)
    
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {Config.RAW_IMAGES_PATH}")
    
    logger.info(f"Found {len(image_paths)} images")
    
    extractor = FeatureExtractor(
        model_name=Config.PRETRAINED_MODEL,
        device=Config.DEVICE
    )
    
    features, valid_paths = extractor.extract_features(
        image_paths,
        batch_size=Config.BATCH_SIZE
    )
    
    np.save(Config.FEATURES_FILE, features)
    np.save(Config.LABELS_FILE, np.array(valid_paths))
    
    logger.info(f"Features saved to {Config.FEATURES_FILE}")
    logger.info(f"Image paths saved to {Config.LABELS_FILE}")
    
    return features, valid_paths


def train_clustering_model(features: np.ndarray) -> Tuple[ProductClusteringModel, np.ndarray, np.ndarray]:
    logger.info("="*50)
    logger.info("TRAINING CLUSTERING MODEL")
    logger.info("="*50)
    
    model = ProductClusteringModel(
        n_clusters=Config.N_CLUSTERS,
        use_pca=Config.USE_PCA,
        use_anomaly_detection=Config.USE_ANOMALY_DETECTION
    )
    
    labels, mask = model.fit(features)
    
    model_path = f"{Config.RESULTS_PATH}/clustering_model.pkl"
    model.save_model(model_path)
    
    return model, labels, mask


def analyze_optimal_k(features: np.ndarray, max_clusters: int = 10) -> None:
    logger.info("="*50)
    logger.info("ANALYZING OPTIMAL K")
    logger.info("="*50)
    
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    if Config.USE_PCA:
        pca = PCA(n_components=Config.PCA_COMPONENTS)
        features_reduced = pca.fit_transform(features_scaled)
    else:
        features_reduced = features_scaled
    
    plot_elbow_curve(
        features_reduced,
        max_clusters=max_clusters,
        save_path=f"{Config.RESULTS_PATH}/plots/elbow_curve.png"
    )
    
    plot_silhouette_scores(
        features_reduced,
        max_clusters=max_clusters,
        save_path=f"{Config.RESULTS_PATH}/plots/silhouette_scores.png"
    )


def create_visualizations(model: ProductClusteringModel, features: np.ndarray, image_paths: np.ndarray) -> None:
    logger.info("="*50)
    logger.info("CREATING VISUALIZATIONS")
    logger.info("="*50)
    
    if not isinstance(image_paths, np.ndarray) or image_paths.ndim == 0:
        image_paths = np.array(image_paths.tolist() if hasattr(image_paths, 'tolist') else image_paths)
    
    if model.use_anomaly_detection:
        features_clean = features[model.anomaly_mask]
        image_paths_clean = image_paths[model.anomaly_mask]
    else:
        features_clean = features
        image_paths_clean = image_paths
    
    if model.use_pca:
        features_viz = model.features_reduced[model.anomaly_mask] if model.use_anomaly_detection else model.features_reduced
    else:
        features_viz = model.features_scaled[model.anomaly_mask] if model.use_anomaly_detection else model.features_scaled
    
    create_all_visualizations(
        features_viz,
        model.cluster_labels,
        image_paths_clean,
        f"{Config.RESULTS_PATH}/plots"
    )


def save_results(model: ProductClusteringModel, image_paths: np.ndarray) -> None:
    import pandas as pd
    
    logger.info("="*50)
    logger.info("SAVING RESULTS")
    logger.info("="*50)
    
    results = []
    
    for i, path in enumerate(image_paths):
        if model.use_anomaly_detection:
            is_anomaly = not model.anomaly_mask[i]
            cluster = model.cluster_labels[np.sum(model.anomaly_mask[:i])] if not is_anomaly else -1
        else:
            is_anomaly = False
            cluster = model.cluster_labels[i]
        
        results.append({
            'image_path': path,
            'cluster': cluster,
            'is_anomaly': is_anomaly
        })
    
    df = pd.DataFrame(results)
    
    output_path = f"{Config.RESULTS_PATH}/clustering_results.csv"
    df.to_csv(output_path, index=False)
    
    logger.info(f"Results saved to {output_path}")
    logger.info(f"Total images: {len(df)}")
    logger.info(f"Anomalies: {df['is_anomaly'].sum()}")
    logger.info(f"Clustered images: {(df['cluster'] != -1).sum()}")
    logger.info("Cluster distribution:")
    logger.info(f"\n{df[df['cluster'] != -1]['cluster'].value_counts().sort_index()}")


def main(args) -> None:
    Config.print_config()
    setup_directories()
    
    features, image_paths = extract_features(force_recompute=args.recompute_features)
    
    if args.analyze_k:
        analyze_optimal_k(features, max_clusters=args.max_k)
    
    model, labels, mask = train_clustering_model(features)
    
    if args.visualize:
        create_visualizations(model, features, image_paths)
    
    save_results(model, image_paths)
    
    logger.info("="*50)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("="*50)
    logger.info(f"Results saved in: {Config.RESULTS_PATH}")
    logger.info("Files created:")
    logger.info("  - clustering_model.pkl")
    logger.info("  - clustering_results.csv")
    logger.info("  - plots/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E-Commerce Product Clustering")
    
    parser.add_argument(
        '--recompute-features',
        action='store_true',
        help='Recompute features even if they exist'
    )
    
    parser.add_argument(
        '--analyze-k',
        action='store_true',
        help='Analyze optimal number of clusters'
    )
    
    parser.add_argument(
        '--max-k',
        type=int,
        default=10,
        help='Maximum number of clusters to analyze'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        default=True,
        help='Create visualizations'
    )
    
    args = parser.parse_args()
    
    main(args)