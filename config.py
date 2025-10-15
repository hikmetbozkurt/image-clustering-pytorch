import torch
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

class Config:
    # Paths
    RAW_IMAGES_PATH = "./data/raw_images"
    PROCESSED_DATA_PATH = "./data/processed"
    FEATURES_FILE = "./data/processed/features.npy"
    LABELS_FILE = "./data/processed/labels.npy"
    RESULTS_PATH = "./results"
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 32
    
    PRETRAINED_MODEL = "resnet50"
    FEATURE_DIM = 2048
    
    USE_PCA = True
    PCA_COMPONENTS = 512
    
    USE_ANOMALY_DETECTION = True
    CONTAMINATION = 0.1
    
    N_CLUSTERS = 5
    RANDOM_STATE = 42
    
    TSNE_COMPONENTS = 2
    TSNE_PERPLEXITY = 30
    
    N_WORKERS = 4
    
    @classmethod
    def print_config(cls) -> None:
        logger = logging.getLogger(__name__)
        logger.info("="*50)
        logger.info("CONFIGURATION")
        logger.info("="*50)
        logger.info(f"Device: {cls.DEVICE}")
        logger.info(f"Image Size: {cls.IMAGE_SIZE}")
        logger.info(f"Batch Size: {cls.BATCH_SIZE}")
        logger.info(f"Model: {cls.PRETRAINED_MODEL}")
        logger.info(f"PCA Components: {cls.PCA_COMPONENTS if cls.USE_PCA else 'Disabled'}")
        logger.info(f"Contamination: {cls.CONTAMINATION if cls.USE_ANOMALY_DETECTION else 'Disabled'}")
        logger.info(f"Number of Clusters: {cls.N_CLUSTERS}")
        logger.info("="*50)