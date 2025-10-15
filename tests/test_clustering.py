import pytest
import numpy as np
from models.clustering import ProductClusteringModel
import tempfile
import os


def test_clustering_init():
    model = ProductClusteringModel(n_clusters=3, use_pca=True, use_anomaly_detection=True)
    assert model.n_clusters == 3
    assert model.use_pca is True
    assert model.scaler is not None
    assert model.kmeans is not None


def test_preprocess():
    model = ProductClusteringModel(n_clusters=3, use_pca=False)
    features = np.random.rand(100, 50)
    
    processed = model.preprocess(features)
    
    assert processed.shape == features.shape
    assert model.features_scaled is not None


def test_preprocess_with_pca():
    from config import Config
    original_pca = Config.PCA_COMPONENTS
    Config.PCA_COMPONENTS = 50  # Adjust for test
    
    model = ProductClusteringModel(n_clusters=3, use_pca=True)
    features = np.random.rand(100, 2048)
    
    processed = model.preprocess(features)
    
    assert processed.shape[0] == 100
    assert processed.shape[1] == 50
    
    Config.PCA_COMPONENTS = original_pca  # Restore


def test_clustering_fit():
    model = ProductClusteringModel(n_clusters=3, use_pca=False, use_anomaly_detection=False)
    features = np.random.rand(50, 20)
    
    labels, mask = model.fit(features)
    
    assert len(labels) == 50
    assert len(mask) == 50
    assert len(np.unique(labels)) <= 3


def test_clustering_with_anomaly_detection():
    model = ProductClusteringModel(n_clusters=3, use_pca=False, use_anomaly_detection=True)
    features = np.random.rand(100, 20)
    
    labels, mask = model.fit(features)
    
    assert len(labels) < 100  # Some removed as anomalies
    assert len(mask) == 100
    assert mask.dtype == bool


def test_save_load_model():
    model = ProductClusteringModel(n_clusters=3, use_pca=False)
    features = np.random.rand(50, 20)
    model.fit(features)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, 'model.pkl')
        model.save_model(model_path)
        
        loaded_model = ProductClusteringModel.load_model(model_path)
        
        assert loaded_model.n_clusters == 3
        assert loaded_model.cluster_labels is not None


def test_predict():
    model = ProductClusteringModel(n_clusters=3, use_pca=False)
    train_features = np.random.rand(50, 20)
    model.fit(train_features)
    
    test_features = np.random.rand(10, 20)
    predictions = model.predict(test_features)
    
    assert len(predictions) == 10
    assert all(0 <= p < 3 for p in predictions)

