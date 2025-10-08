"""
Data loading, preprocessing, and evaluation utilities for DML experiments.
"""

import numpy as np
import pandas as pd
import os
import warnings
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification

warnings.filterwarnings('ignore', category=FutureWarning)


class DataLoader:
    """Data loading utility for various datasets used in DML experiments."""
    
    def __init__(self, data_dir=None):
        if data_dir is None:
            # Try to find DataSets directory relative to current location
            current_dir = Path(__file__).parent.parent.parent
            potential_paths = [
                current_dir / 'DataSets',
                current_dir.parent / 'DataSets',
                Path('./DataSets'),
                Path('../DataSets'),
                Path('../../DataSets')
            ]
            
            self.data_dir = None
            for path in potential_paths:
                if path.exists() and path.is_dir():
                    self.data_dir = path
                    break
        else:
            self.data_dir = Path(data_dir)
            
        # List of available datasets
        self.available_datasets = [
            'Vehicle', 'Bupa', 'Glass', 'Ionosphere', 'Iris', 
            'Wine', 'WDBC', 'Pima', 'New-thyroid'
        ]
        
    def load_dataset(self, dataset_name):
        """Load a dataset by name."""
        if self.data_dir is None or not self.data_dir.exists():
            print(f"Warning: DataSets directory not found. Generating synthetic data for {dataset_name}")
            return self._generate_synthetic_data_for_dataset(dataset_name)
            
        dataset_path = self.data_dir / dataset_name
        
        if not dataset_path.exists():
            print(f"Warning: Dataset {dataset_name} not found. Generating synthetic data.")
            return self._generate_synthetic_data_for_dataset(dataset_name)
            
        try:
            # Try to load from Excel files (most common format)
            samples_file = dataset_path / 'Samples.xlsx'
            labels_file = dataset_path / 'Labels.xlsx'
            
            if samples_file.exists() and labels_file.exists():
                X = pd.read_excel(samples_file, header=None).values
                y = pd.read_excel(labels_file, header=None).values.ravel()
            else:
                # Try alternative formats
                csv_file = dataset_path / 'Data.csv'
                if csv_file.exists():
                    data = pd.read_csv(csv_file)
                    X = data.iloc[:, :-1].values
                    y = data.iloc[:, -1].values
                else:
                    raise FileNotFoundError("No suitable data files found")
                        
            print(f"Loaded {dataset_name}: {len(X)} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
            return X, y
                
        except Exception as e:
            print(f"Error loading {dataset_name}: {e}. Generating synthetic data.")
            return self._generate_synthetic_data_for_dataset(dataset_name)
            
    def _generate_synthetic_data_for_dataset(self, dataset_name):
        """Generate synthetic data based on typical characteristics of known datasets."""
        
        # Dataset characteristics (approximate)
        dataset_params = {
            'Vehicle': (846, 18, 4),
            'Bupa': (345, 6, 2),
            'Glass': (214, 9, 6),
            'Ionosphere': (351, 34, 2),
            'Iris': (150, 4, 3),
            'Wine': (178, 13, 3),
            'WDBC': (569, 30, 2),
            'Pima': (768, 8, 2),
            'New-thyroid': (215, 5, 3),
        }
        
        if dataset_name in dataset_params:
            n_samples, n_features, n_classes = dataset_params[dataset_name]
        else:
            # Default synthetic dataset
            n_samples, n_features, n_classes = 200, 10, 3
            
        return self._generate_synthetic_data(n_samples, n_features, n_classes, dataset_name)
                                           
    def _generate_synthetic_data(self, n_samples, n_features, n_classes, dataset_name="Synthetic"):
        """Generate synthetic classification data."""
        
        # Generate synthetic data with some complexity
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=max(2, n_features // 2),
            n_redundant=max(0, n_features // 4),
            n_classes=n_classes,
            n_clusters_per_class=1,
            random_state=42,
            class_sep=0.8
        )
        
        # Add some noise to make it more realistic
        noise = np.random.normal(0, 0.1, X.shape)
        X = X + noise
        
        print(f"Generated synthetic {dataset_name}: {n_samples} samples, {n_features} features, {n_classes} classes")
        return X, y


class CrossValidator:
    """Cross-validation utility for DML experiments."""
    
    def __init__(self, n_splits=10, random_state=42):
        self.n_splits = n_splits
        self.random_state = random_state
        self.cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
    def cross_validate(self, estimator, X, y):
        """Perform cross-validation on estimator."""
        test_scores = []
        
        for train_idx, test_idx in self.cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Fit and predict
            estimator.fit(X_train, y_train)
            y_pred = estimator.predict(X_test)
            
            # Calculate metrics
            test_score = accuracy_score(y_test, y_pred)
            test_scores.append(test_score)
                
        results = {
            'test_accuracy_mean': np.mean(test_scores),
            'test_accuracy_std': np.std(test_scores),
            'test_scores': test_scores
        }
            
        return results


class PerformanceEvaluator:
    """Performance evaluation utility."""
    
    def __init__(self):
        self.results = []


def preprocess_data(X, y, standardize=True, encode_labels=True):
    """Preprocess data for DML experiments."""
    X = np.array(X)
    y = np.array(y)
    
    preprocessors = {}
    
    # Handle missing values
    if np.any(np.isnan(X)):
        print("Warning: Missing values detected, filling with median")
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        X = imputer.fit_transform(X)
        preprocessors['imputer'] = imputer
        
    # Standardize features
    if standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        preprocessors['scaler'] = scaler
        
    # Encode labels
    if encode_labels:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        preprocessors['label_encoder'] = label_encoder
        
    return X, y, preprocessors
