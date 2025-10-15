import pytest
import numpy as np
import torch
from models.feature_extractor import FeatureExtractor, get_image_paths
from PIL import Image
import tempfile
import os


def test_get_image_paths():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test images
        Image.new('RGB', (100, 100)).save(os.path.join(tmpdir, 'test1.jpg'))
        Image.new('RGB', (100, 100)).save(os.path.join(tmpdir, 'test2.png'))
        
        paths = get_image_paths(tmpdir)
        assert len(paths) == 2
        assert all(p.endswith(('.jpg', '.png')) for p in paths)


def test_feature_extractor_init():
    extractor = FeatureExtractor(model_name='resnet50')
    assert extractor.model is not None
    assert extractor.transform is not None
    assert extractor.device == torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_extract_single_image():
    extractor = FeatureExtractor(model_name='resnet50')
    
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = os.path.join(tmpdir, 'test.jpg')
        img = Image.new('RGB', (224, 224))
        img.save(img_path)
        
        features = extractor.extract_single_image(img_path)
        
        assert features is not None
        assert isinstance(features, np.ndarray)
        assert features.shape[1] == 2048


def test_extract_features_batch():
    extractor = FeatureExtractor(model_name='resnet50')
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test images
        for i in range(5):
            Image.new('RGB', (224, 224)).save(os.path.join(tmpdir, f'test{i}.jpg'))
        
        paths = get_image_paths(tmpdir)
        features, valid_paths = extractor.extract_features(paths, batch_size=2)
        
        assert features.shape[0] == 5
        assert features.shape[1] == 2048
        assert len(valid_paths) == 5

