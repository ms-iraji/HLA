
"""
data_loader.py
-------------------------------------------------------
Utility functions for loading and preprocessing public
gene-expression (microarray) datasets used in the paper:
    - Colon
    - CNS
    - GLI-85
    - SMK-CAN-187
    - Leukemia
Each dataset is referenced from open-access repositories (GEO or UCI).
Author: [Your Name]
-------------------------------------------------------
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

# -------------------------------------------------------
# Predefined dataset sources (publicly available)
# -------------------------------------------------------
DATASET_LINKS = {
    "colon": "https://jundongli.com/upload/dataset/Colon.csv",
    "cns": "https://jundongli.com/upload/dataset/CNS.csv",
    "gli85": "https://jundongli.com/upload/dataset/GLI-85.csv",
    "smkcan187": "https://jundongli.com/upload/dataset/SMK-CAN-187.csv",
    "leukemia": "https://jundongli.com/upload/dataset/Leukemia.csv"
}

def load_csv_dataset(path, label_col=0, normalize=True):
    """Load a CSV dataset from disk or URL."""
    df = pd.read_csv(path)
    y = df.iloc[:, label_col].values
    X = df.drop(df.columns[label_col], axis=1).values
    if normalize:
        X = StandardScaler().fit_transform(X)
    return X, y

def load_sample_dataset(name="colon", normalize=True):
    """Load a named benchmark microarray dataset."""
    name = name.lower()
    if name not in DATASET_LINKS:
        raise ValueError(f"Dataset {name} not found. Choose from {list(DATASET_LINKS.keys())}")
    url = DATASET_LINKS[name]
    print(f"[INFO] Downloading {name.upper()} dataset from: {url}")
    df = pd.read_csv(url)
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values
    if normalize:
        X = StandardScaler().fit_transform(X)
    return X, y

def load_synthetic_microarray(n_samples=200, n_features=100, n_informative=20, random_state=0):
    """Generate synthetic data for demonstration."""
    X, y = make_classification(n_samples=n_samples, n_features=n_features,
                               n_informative=n_informative, n_redundant=5,
                               random_state=random_state)
    X = StandardScaler().fit_transform(X)
    return X, y

if __name__ == "__main__":
    # Example usage
    X, y = load_sample_dataset("colon")
    print("Dataset shape:", X.shape, "Unique labels:", np.unique(y))
