import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from PIL import Image
import os
import logging
from typing import List, Optional
from config import Config

logger = logging.getLogger(__name__)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def plot_clusters_2d(features: np.ndarray, labels: np.ndarray, method: str = 'tsne', save_path: Optional[str] = None) -> None:
    logger.info(f"Visualizing clusters using {method.upper()}...")
    
    # Reduce to 2D
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=Config.RANDOM_STATE, perplexity=30)
        features_2d = reducer.fit_transform(features)
    else:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Plot each cluster with different color
    unique_labels = np.unique(labels)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=[colors[i]],
            label=f'Cluster {label}',
            alpha=0.6,
            s=50,
            edgecolors='black',
            linewidth=0.5
        )
    
    plt.title(f'Product Clusters Visualization ({method.upper()})', fontsize=16, fontweight='bold')
    plt.xlabel('Component 1', fontsize=12)
    plt.ylabel('Component 2', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    plt.show()


def plot_cluster_distribution(labels: np.ndarray, save_path: Optional[str] = None) -> None:
    unique, counts = np.unique(labels, return_counts=True)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(unique, counts, color=plt.cm.Set3(np.linspace(0, 1, len(unique))), 
                   edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.title('Cluster Size Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Cluster ID', fontsize=12)
    plt.ylabel('Number of Products', fontsize=12)
    plt.xticks(unique)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    plt.show()


def plot_sample_images_per_cluster(image_paths: List[str], labels: np.ndarray, n_samples: int = 5, save_path: Optional[str] = None) -> None:
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    fig, axes = plt.subplots(n_clusters, n_samples, figsize=(15, 3*n_clusters))
    
    if n_clusters == 1:
        axes = axes.reshape(1, -1)
    
    for i, label in enumerate(unique_labels):
        # Get images from this cluster
        cluster_indices = np.where(labels == label)[0]
        
        # Randomly sample images
        sample_indices = np.random.choice(
            cluster_indices, 
            size=min(n_samples, len(cluster_indices)), 
            replace=False
        )
        
        for j, idx in enumerate(sample_indices):
            try:
                img = Image.open(image_paths[idx]).convert('RGB')
                axes[i, j].imshow(img)
                axes[i, j].axis('off')
                
                if j == 0:
                    axes[i, j].set_title(f'Cluster {label}', fontsize=12, fontweight='bold')
            except:
                axes[i, j].axis('off')
        
        # Hide unused subplots
        for j in range(len(sample_indices), n_samples):
            axes[i, j].axis('off')
    
    plt.suptitle('Sample Products per Cluster', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    plt.show()


def plot_elbow_curve(features: np.ndarray, max_clusters: int = 10, save_path: Optional[str] = None) -> None:
    from sklearn.cluster import KMeans
    
    inertias = []
    K_range = range(2, max_clusters + 1)
    
    logger.info("Computing elbow curve...")
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=Config.RANDOM_STATE, n_init=10)
        kmeans.fit(features)
        inertias.append(kmeans.inertia_)
    
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Inertia', fontsize=12)
    plt.title('Elbow Curve', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    plt.show()


def plot_silhouette_scores(features: np.ndarray, max_clusters: int = 10, save_path: Optional[str] = None) -> None:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    
    silhouette_scores = []
    K_range = range(2, max_clusters + 1)
    
    logger.info("Computing silhouette scores...")
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=Config.RANDOM_STATE, n_init=10)
        labels = kmeans.fit_predict(features)
        score = silhouette_score(features, labels)
        silhouette_scores.append(score)
    
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Silhouette Score', fontsize=12)
    plt.title('Silhouette Scores', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    plt.show()


def create_all_visualizations(features: np.ndarray, labels: np.ndarray, image_paths: List[str], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("="*50)
    logger.info("CREATING VISUALIZATIONS")
    logger.info("="*50)
    
    plot_clusters_2d(
        features, labels, method='tsne',
        save_path=f"{output_dir}/clusters_tsne.png"
    )
    
    plot_clusters_2d(
        features, labels, method='pca',
        save_path=f"{output_dir}/clusters_pca.png"
    )
    
    plot_cluster_distribution(
        labels,
        save_path=f"{output_dir}/cluster_distribution.png"
    )
    
    plot_sample_images_per_cluster(
        image_paths, labels, n_samples=5,
        save_path=f"{output_dir}/sample_products.png"
    )
    
    logger.info("All visualizations created successfully!")


if __name__ == "__main__":
    logger.info("Testing visualization utilities...")
    
    # Load data
    features = np.load(Config.FEATURES_FILE)
    image_paths = np.load(Config.LABELS_FILE, allow_pickle=True)
    
    # Load model
    from models.clustering import ProductClusteringModel
    model = ProductClusteringModel.load_model(f"{Config.RESULTS_PATH}/clustering_model.pkl")
    
    # Get clean features and labels
    features_clean = features[model.anomaly_mask]
    labels = model.cluster_labels
    image_paths_clean = image_paths[model.anomaly_mask]
    
    # Create visualizations
    create_all_visualizations(
        features_clean, 
        labels, 
        image_paths_clean,
        f"{Config.RESULTS_PATH}/plots"
    )