import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import logging
from typing import List, Tuple, Optional
from config import Config

logger = logging.getLogger(__name__)


class ImageDataset(Dataset):
    def __init__(self, image_paths: List[str], transform: Optional[transforms.Compose] = None):
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, img_path
        except (IOError, OSError) as e:
            logger.warning(f"Failed to load {img_path}: {e}")
            blank = Image.new('RGB', Config.IMAGE_SIZE, (255, 255, 255))
            if self.transform:
                blank = self.transform(blank)
            return blank, img_path


class FeatureExtractor:
    def __init__(self, model_name: str = 'resnet50', device: Optional[torch.device] = None):
        self.device = device or Config.DEVICE
        self.model_name = model_name
        self.model = self._load_model()
        self.transform = self._get_transforms()
        
    def _load_model(self) -> nn.Module:
        logger.info(f"Loading {self.model_name} model...")
        
        if self.model_name == 'resnet50':
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            model = nn.Sequential(*list(model.children())[:-1])
        elif self.model_name == 'resnet101':
            model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
            model = nn.Sequential(*list(model.children())[:-1])
        elif self.model_name == 'vgg16':
            model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            model = model.features
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        model = model.to(self.device)
        model.eval()
        return model
    
    def _get_transforms(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def extract_features(self, image_paths: List[str], batch_size: int = 32) -> Tuple[np.ndarray, List[str]]:
        dataset = ImageDataset(image_paths, transform=self.transform)
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=False,
            num_workers=Config.N_WORKERS
        )
        
        features_list = []
        valid_paths = []
        
        logger.info(f"Extracting features from {len(image_paths)} images...")
        
        with torch.no_grad():
            for images, paths in tqdm(dataloader, desc="Processing batches"):
                images = images.to(self.device)
                outputs = self.model(images)
                outputs = outputs.view(outputs.size(0), -1)
                
                features_list.append(outputs.cpu().numpy())
                valid_paths.extend(paths)
        
        features = np.vstack(features_list)
        logger.info(f"Extracted features shape: {features.shape}")
        
        return features, valid_paths
    
    def extract_single_image(self, image_path: str) -> Optional[np.ndarray]:
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.model(image_tensor)
                features = features.view(features.size(0), -1)
            
            return features.cpu().numpy()
        except (IOError, OSError) as e:
            logger.error(f"Error processing {image_path}: {e}")
            return None


def get_image_paths(directory: str) -> List[str]:
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    image_paths = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if os.path.splitext(file)[1].lower() in valid_extensions:
                image_paths.append(os.path.join(root, file))
    
    return image_paths


if __name__ == "__main__":
    Config.print_config()
    
    extractor = FeatureExtractor(model_name=Config.PRETRAINED_MODEL)
    image_paths = get_image_paths(Config.RAW_IMAGES_PATH)
    logger.info(f"Found {len(image_paths)} images")
    
    if len(image_paths) > 0:
        features, valid_paths = extractor.extract_features(
            image_paths, 
            batch_size=Config.BATCH_SIZE
        )
        
        os.makedirs(Config.PROCESSED_DATA_PATH, exist_ok=True)
        np.save(Config.FEATURES_FILE, features)
        np.save(Config.LABELS_FILE, valid_paths)
        
        logger.info(f"Features saved to {Config.FEATURES_FILE}")
        logger.info(f"Image paths saved to {Config.LABELS_FILE}")