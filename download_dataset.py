import os
import shutil
import logging
from pathlib import Path
from typing import Optional, Dict, List
from tqdm import tqdm

logger = logging.getLogger(__name__)


def get_free_disk_space(path: str = ".") -> int:
    """Get free disk space in bytes"""
    import shutil
    stat = shutil.disk_usage(path)
    return stat.free


def format_size(bytes: int) -> str:
    """Format bytes to human-readable size"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes < 1024.0:
            return f"{bytes:.1f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.1f} PB"


def check_disk_space(required_gb: float, path: str = ".") -> bool:
    """Check if enough disk space is available"""
    required_bytes = required_gb * 1024 * 1024 * 1024
    free_space = get_free_disk_space(path)
    
    if free_space < required_bytes:
        logger.error(f"Insufficient disk space!")
        logger.error(f"Required: {format_size(required_bytes)}")
        logger.error(f"Available: {format_size(free_space)}")
        return False
    
    logger.info(f"Disk space check passed: {format_size(free_space)} available")
    return True


def setup_kaggle() -> bool:
    """Check if Kaggle API is configured"""
    try:
        import kaggle
        logger.info("Kaggle API configured")
        return True
    except OSError:
        logger.error("Kaggle API not configured")
        logger.info("\nSetup instructions:")
        logger.info("1. Go to https://www.kaggle.com/account")
        logger.info("2. Click 'Create New API Token'")
        logger.info("3. Save kaggle.json to:")
        logger.info("   - Windows: C:\\Users\\<username>\\.kaggle\\kaggle.json")
        logger.info("   - Linux/Mac: ~/.kaggle/kaggle.json")
        return False
    except ImportError:
        logger.error("Kaggle package not installed")
        logger.info("Install with: pip install kaggle")
        return False


def download_fashion_product_images(target_dir: str = "./data/raw_images") -> bool:
    """
    Download Fashion Product Images Dataset from Kaggle
    Dataset: ~44K images, ~15GB
    """
    logger.info("="*60)
    logger.info("DOWNLOADING FASHION PRODUCT IMAGES DATASET")
    logger.info("="*60)
    logger.warning("Dataset size: ~15 GB | Images: ~44,000")
    
    # Check disk space (need ~20GB for download + extraction)
    if not check_disk_space(required_gb=20.0):
        logger.error("Please free up disk space before continuing")
        return False
    
    # Confirm download
    logger.info("\nThis will download ~15GB of data. Continue? (y/n)")
    try:
        confirmation = input().strip().lower()
        if confirmation != 'y':
            logger.info("Download cancelled")
            return False
    except (EOFError, KeyboardInterrupt):
        logger.info("\nDownload cancelled")
        return False
    
    if not setup_kaggle():
        return False
    
    import kaggle
    
    temp_dir = "./temp_dataset"
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        logger.info("\nDownloading dataset (this may take 10-30 minutes)...")
        kaggle.api.dataset_download_files(
            'paramaggarwal/fashion-product-images-dataset',
            path=temp_dir,
            unzip=True
        )
        
        # Find images directory
        images_source = None
        for root, dirs, files in os.walk(temp_dir):
            if 'images' in dirs:
                images_source = os.path.join(root, 'images')
                break
            if len([f for f in files if f.endswith(('.jpg', '.png'))]) > 100:
                images_source = root
                break
        
        if not images_source:
            logger.error("Could not find images in downloaded dataset")
            return False
        
        # Count total images first
        all_images = []
        for root, _, files in os.walk(images_source):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    all_images.append(os.path.join(root, file))
        
        logger.info(f"Found {len(all_images)} images. Copying to {target_dir}...")
        os.makedirs(target_dir, exist_ok=True)
        
        # Copy with progress bar
        for src in tqdm(all_images, desc="Copying images", unit="img"):
            dst = os.path.join(target_dir, os.path.basename(src))
            shutil.copy2(src, dst)
        
        logger.info(f"Successfully copied {len(all_images)} images")
        
        # Cleanup
        logger.info("Cleaning up temporary files...")
        shutil.rmtree(temp_dir)
        
        return True
        
    except KeyboardInterrupt:
        logger.warning("\nDownload interrupted by user")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return False
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return False


def download_sample_images(target_dir: str = "./data/raw_images", n_samples: int = 100) -> bool:
    """
    Download a small sample of images for testing
    Uses Fashion MNIST dataset (~50MB download)
    """
    logger.info("="*60)
    logger.info(f"DOWNLOADING {n_samples} SAMPLE IMAGES FOR TESTING")
    logger.info("="*60)
    logger.info("Dataset: Fashion MNIST | Size: ~50 MB")
    
    # Check disk space
    if not check_disk_space(required_gb=0.5):
        logger.error("Please free up disk space")
        return False
    
    os.makedirs(target_dir, exist_ok=True)
    
    try:
        from torchvision.datasets import FashionMNIST
        from PIL import Image
        import numpy as np
        
        logger.info("\nDownloading Fashion MNIST dataset...")
        dataset = FashionMNIST(
            root='./data/fashion_mnist',
            train=True,
            download=True
        )
        
        n_to_save = min(n_samples, len(dataset))
        logger.info(f"Converting {n_to_save} images to RGB format...")
        
        for i in tqdm(range(n_to_save), desc="Processing images", unit="img"):
            img, _ = dataset[i]
            img_rgb = Image.fromarray(np.array(img)).convert('RGB')
            img_rgb = img_rgb.resize((224, 224))
            img_rgb.save(f"{target_dir}/sample_{i:04d}.png")
        
        logger.info(f"Successfully saved {n_to_save} sample images")
        return True
        
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.info("Install with: pip install torchvision pillow")
        return False
    except Exception as e:
        logger.error(f"Error: {e}")
        return False


def list_available_datasets() -> None:
    """List popular e-commerce datasets available on Kaggle"""
    datasets: List[Dict[str, str]] = [
        {
            "name": "Fashion Product Images",
            "kaggle_id": "paramaggarwal/fashion-product-images-dataset",
            "size": "~15 GB",
            "images": "44,000+",
            "description": "Fashion e-commerce products with metadata"
        },
        {
            "name": "Amazon Fashion Products",
            "kaggle_id": "PromptCloudHQ/all-jc-penny-products",
            "size": "~2 GB",
            "images": "10,000+",
            "description": "JCPenney fashion products"
        },
        {
            "name": "Flipkart Products",
            "kaggle_id": "PromptCloudHQ/flipkart-products",
            "size": "~500 MB",
            "images": "20,000+",
            "description": "Flipkart product catalog"
        }
    ]
    
    logger.info("="*60)
    logger.info("AVAILABLE E-COMMERCE DATASETS")
    logger.info("="*60)
    
    for i, ds in enumerate(datasets, 1):
        logger.info(f"\n{i}. {ds['name']}")
        logger.info(f"   Kaggle ID: {ds['kaggle_id']}")
        logger.info(f"   Size: {ds['size']}")
        logger.info(f"   Images: {ds['images']}")
        logger.info(f"   Description: {ds['description']}")


def main() -> None:
    """Main menu for dataset download"""
    # Setup logging for command-line usage
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    logger.info("="*60)
    logger.info("E-COMMERCE DATASET DOWNLOADER")
    logger.info("="*60)
    
    logger.info("\nOptions:")
    logger.info("1. Download Fashion Product Images (Full - ~44K images, ~15GB)")
    logger.info("2. Download sample images for testing (100 images, ~50MB)")
    logger.info("3. List available datasets")
    logger.info("4. Exit")
    
    try:
        choice = input("\nEnter your choice (1-4): ").strip()
    except (EOFError, KeyboardInterrupt):
        logger.info("\nExiting...")
        return
    
    if choice == "1":
        success = download_fashion_product_images()
        if success:
            logger.info("="*60)
            logger.info("READY TO USE!")
            logger.info("="*60)
            logger.info("Next step: python main.py --recompute-features")
    
    elif choice == "2":
        success = download_sample_images(n_samples=100)
        if success:
            logger.info("="*60)
            logger.info("READY TO USE!")
            logger.info("="*60)
            logger.info("Next step: python main.py")
    
    elif choice == "3":
        list_available_datasets()
        logger.info("\nManual download instructions:")
        logger.info("1. Visit the Kaggle dataset page")
        logger.info("2. Download the ZIP file")
        logger.info("3. Extract images to ./data/raw_images/")
        logger.info("4. Run: python main.py --recompute-features")
    
    elif choice == "4":
        logger.info("Exiting...")
    
    else:
        logger.warning("Invalid choice! Please enter 1-4")


if __name__ == "__main__":
    main()
