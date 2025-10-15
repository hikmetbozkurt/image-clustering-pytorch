# Image Clustering Project

A simple prototype project for learning PyTorch and unsupervised machine learning.

## About

This is a basic learning project that demonstrates image clustering using:
- **PyTorch** for feature extraction (ResNet50)
- **Scikit-learn** for K-means clustering
- **Fashion MNIST** dataset for training

**Note:** This is a prototype/learning project, not intended for production use.

## Project Structure

```
ImageProcess/
├── data/               # Dataset storage
├── models/            # Feature extractor and clustering models
├── utils/             # Visualization utilities
├── results/           # Output files and plots
├── main.py            # Main pipeline
└── config.py          # Configuration settings
```

## Installation

1. Install required packages:

```bash
pip install -r requirements.txt
```

2. Download dataset (choose one option):

**Option A: Automatic Download (Recommended)**
```bash
python download_dataset.py
# Select option 1 for full dataset or option 2 for sample images
```

**Option B: Manual Download**
- Download from [Kaggle Fashion Products Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset)
- Extract images to `data/raw_images/` folder

**Option C: Use Your Own Images**
- Simply place your images in `data/raw_images/` folder
- Supports: jpg, jpeg, png, bmp, gif

## Usage

Run the main script:

```bash
python main.py
```

Run tests:

```bash
pytest tests/
```

## Features

- Image feature extraction using pre-trained ResNet50
- K-means clustering for grouping similar images
- PCA dimensionality reduction
- Visualization of clustering results (t-SNE, PCA plots)

## Output

The project generates:
- Clustering results in CSV format
- Visualization plots (clusters, distributions)
- Saved model files

## Learning Goals

This project was created to practice:
- Working with PyTorch and pre-trained models
- Understanding unsupervised learning
- Image processing and feature extraction
- Data visualization techniques

## License

Educational/Learning purposes only.