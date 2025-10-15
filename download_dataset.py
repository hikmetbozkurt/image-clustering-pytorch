"""
Download and prepare e-commerce datasets for the project
"""

import os
import zipfile
import shutil
from pathlib import Path


def setup_kaggle():
    """Check if Kaggle API is configured"""
    try:
        import kaggle
        print("✓ Kaggle API configured")
        return True
    except OSError:
        print("❌ Kaggle API not configured")
        print("\nSetup instructions:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Click 'Create New API Token'")
        print("3. Save kaggle.json to:")
        print("   - Windows: C:\\Users\\<username>\\.kaggle\\kaggle.json")
        print("   - Linux/Mac: ~/.kaggle/kaggle.json")
        return False


def download_fashion_product_images(target_dir="./data/raw_images"):
    """
    Download Fashion Product Images Dataset from Kaggle
    
    Dataset: ~44K fashion product images
    Source: https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset
    """
    print("\n" + "="*60)
    print("DOWNLOADING FASHION PRODUCT IMAGES DATASET")
    print("="*60)
    
    if not setup_kaggle():
        return False
    
    import kaggle
    
    # Create temp directory
    temp_dir = "./temp_dataset"
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Download dataset
        print("\nDownloading dataset (this may take a few minutes)...")
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
            # Sometimes images are directly in the folder
            if len([f for f in files if f.endswith(('.jpg', '.png'))]) > 100:
                images_source = root
                break
        
        if not images_source:
            print("❌ Could not find images in downloaded dataset")
            return False
        
        # Copy images to target directory
        print(f"\nCopying images to {target_dir}...")
        os.makedirs(target_dir, exist_ok=True)
        
        image_count = 0
        for root, _, files in os.walk(images_source):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    src = os.path.join(root, file)
                    dst = os.path.join(target_dir, file)
                    shutil.copy2(src, dst)
                    image_count += 1
                    
                    if image_count % 1000 == 0:
                        print(f"  Copied {image_count} images...")
        
        print(f"\n✓ Successfully copied {image_count} images to {target_dir}")
        
        # Cleanup
        print("\nCleaning up temporary files...")
        shutil.rmtree(temp_dir)
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return False


def download_sample_images(target_dir="./data/raw_images", n_samples=100):
    """
    Download a small sample of images for testing
    Uses a public dataset or generates sample images
    """
    print("\n" + "="*60)
    print(f"DOWNLOADING {n_samples} SAMPLE IMAGES FOR TESTING")
    print("="*60)
    
    os.makedirs(target_dir, exist_ok=True)
    
    try:
        from torchvision.datasets import FashionMNIST
        from PIL import Image
        import numpy as np
        
        # Download Fashion MNIST as fallback
        print("\nDownloading Fashion MNIST for testing...")
        dataset = FashionMNIST(
            root='./data/fashion_mnist',
            train=True,
            download=True
        )
        
        # Convert and save samples
        print(f"Converting {n_samples} images...")
        for i in range(min(n_samples, len(dataset))):
            img, _ = dataset[i]
            # Convert grayscale to RGB
            img_rgb = Image.fromarray(np.array(img)).convert('RGB')
            # Resize to realistic size
            img_rgb = img_rgb.resize((224, 224))
            img_rgb.save(f"{target_dir}/sample_{i:04d}.png")
            
            if (i + 1) % 100 == 0:
                print(f"  Saved {i + 1} images...")
        
        print(f"\n✓ Successfully saved {min(n_samples, len(dataset))} sample images")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def list_available_datasets():
    """List popular e-commerce datasets available on Kaggle"""
    datasets = [
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
    
    print("\n" + "="*60)
    print("AVAILABLE E-COMMERCE DATASETS")
    print("="*60)
    
    for i, ds in enumerate(datasets, 1):
        print(f"\n{i}. {ds['name']}")
        print(f"   Kaggle ID: {ds['kaggle_id']}")
        print(f"   Size: {ds['size']}")
        print(f"   Images: {ds['images']}")
        print(f"   Description: {ds['description']}")


def main():
    """Main menu for dataset download"""
    print("\n" + "="*60)
    print("E-COMMERCE DATASET DOWNLOADER")
    print("="*60)
    
    print("\nOptions:")
    print("1. Download Fashion Product Images (Full - ~44K images)")
    print("2. Download sample images for testing (100 images)")
    print("3. List available datasets")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        success = download_fashion_product_images()
        if success:
            print("\n" + "="*60)
            print("READY TO USE!")
            print("="*60)
            print("Run: python main.py")
    
    elif choice == "2":
        success = download_sample_images(n_samples=100)
        if success:
            print("\n" + "="*60)
            print("READY TO USE!")
            print("="*60)
            print("Run: python main.py")
    
    elif choice == "3":
        list_available_datasets()
        print("\nTo download manually:")
        print("1. Visit the Kaggle dataset page")
        print("2. Download the ZIP file")
        print("3. Extract images to ./data/raw_images/")
    
    elif choice == "4":
        print("Exiting...")
    
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    main()

