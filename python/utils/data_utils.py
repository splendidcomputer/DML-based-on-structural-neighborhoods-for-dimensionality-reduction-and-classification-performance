"""
Utility functions for data loading, preprocessing, and evaluation.
"""

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import scipy.io
import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    """Class for loading and preprocessing datasets."""

    def __init__(self, data_dir="../DataSets"):
        """
        Initialize DataLoader.

        Parameters:
        -----------
        data_dir : str
            Path to the datasets directory
        """
        self.data_dir = data_dir
        self.supported_datasets = [
            'Vehicle', 'Bupa', 'Glass', 'Ionosphere', 'Monks',
            'New-thyroid', 'Pima', 'WDBC', 'Iris', 'Wine',
            'Wholesale', 'CRC', 'KDD'
        ]

    def load_dataset(self, dataset_name):
        """
        Load a specific dataset.

        Parameters:
        -----------
        dataset_name : str
            Name of the dataset to load

        Returns:
        --------
        tuple of (numpy.ndarray, numpy.ndarray)
            Features and labels
        """
        if dataset_name not in self.supported_datasets:
            raise ValueError(f"Dataset {dataset_name} not supported. "
                             f"Supported datasets: {self.supported_datasets}")

        if dataset_name == 'KDD':
            return self._load_kdd_dataset()
        else:
            return self._load_xlsx_dataset(dataset_name)

    def _load_kdd_dataset(self):
        """Load KDD dataset from .mat files."""
        try:
            data_path = os.path.join(self.data_dir, 'KDD', 'kddData.mat')
            label_path = os.path.join(self.data_dir, 'KDD', 'kddLabel.mat')

            data_mat = scipy.io.loadmat(data_path)
            label_mat = scipy.io.loadmat(label_path)

            # Extract data (key names might vary)
            data_keys = [k for k in data_mat.keys() if not k.startswith('__')]
            label_keys = [
                k for k in label_mat.keys() if not k.startswith('__')]

            X = data_mat[data_keys[0]]
            y = label_mat[label_keys[0]].flatten()

            return X, y

        except Exception as e:
            print(f"Error loading KDD dataset: {e}")
            # Generate synthetic KDD-like data as fallback
            return self._generate_synthetic_data(1000, 41, 2)

    def _load_xlsx_dataset(self, dataset_name):
        """Load dataset from Excel files."""
        try:
            samples_path = os.path.join(
                self.data_dir, dataset_name, 'Samples.xlsx')
            labels_path = os.path.join(
                self.data_dir, dataset_name, 'Labels.xlsx')

            X = pd.read_excel(samples_path, header=None).values
            y = pd.read_excel(labels_path, header=None).values.flatten()

            return X, y

        except Exception as e:
            print(f"Error loading {dataset_name} dataset: {e}")
            # Generate synthetic data as fallback
            return self._generate_synthetic_dataset(dataset_name)

    def _generate_synthetic_dataset(self, dataset_name):
        """Generate synthetic data for testing when real data is not available."""

        synthetic_configs = {
            'Vehicle': (846, 18, 4),
            'Bupa': (345, 6, 2),
            'Glass': (214, 9, 6),
            'Ionosphere': (351, 34, 2),
            'Monks': (432, 6, 2),
            'New-thyroid': (215, 5, 3),
            'Pima': (768, 8, 2),
            'WDBC': (569, 30, 2),
            'Iris': (150, 4, 3),
            'Wine': (178, 13, 3),
            'Wholesale': (440, 8, 2),
            'CRC': (1000, 50, 2)
        }

        if dataset_name in synthetic_configs:
            n_samples, n_features, n_classes = synthetic_configs[dataset_name]
        else:
            n_samples, n_features, n_classes = 100, 10, 2

        print(f"Generating synthetic {dataset_name} dataset with "
              f"{n_samples} samples, {n_features} features, {n_classes} classes")

        return self._generate_synthetic_data(n_samples, n_features, n_classes)

    def _generate_synthetic_data(self, n_samples, n_features, n_classes):
        """Generate synthetic classification data."""
        np.random.seed(42)

        # Generate class centers
        centers = np.random.randn(n_classes, n_features) * 3

        X = []
        y = []

        samples_per_class = n_samples // n_classes

        for i in range(n_classes):
            # Generate samples around each center
            class_samples = np.random.randn(
                samples_per_class, n_features) + centers[i]
            X.append(class_samples)
            y.append(np.full(samples_per_class, i))

        # Handle remaining samples
        remaining = n_samples - samples_per_class * n_classes
        if remaining > 0:
            extra_samples = np.random.randn(remaining, n_features) + centers[0]
            X.append(extra_samples)
            y.append(np.full(remaining, 0))

        X = np.vstack(X)
        y = np.hstack(y)

        return X, y


class CrossValidator:
    """Class for cross-validation and evaluation."""

    def __init__(self, n_folds=10, random_state=42):
        """
        Initialize CrossValidator.

        Parameters:
        -----------
        n_folds : int, default=10
            Number of folds for cross-validation
        random_state : int, default=42
            Random state for reproducibility
        """
        self.n_folds = n_folds
        self.random_state = random_state

    def create_cv_partition(self, X, y, reduction_perc=1.0):
        """
        Create cross-validation partitions.

        Parameters:
        -----------
        X : numpy.ndarray
            Features
        y : numpy.ndarray
            Labels
        reduction_perc : float, default=1.0
            Percentage of data to use (for large datasets)

        Returns:
        --------
        generator
            Generator yielding (train_indices, test_indices) for each fold
        """
        # Apply data reduction if specified
        if reduction_perc < 1.0:
            n_samples = int(len(X) * reduction_perc)
            indices = np.random.choice(len(X), size=n_samples, replace=False)
            X = X[indices]
            y = y[indices]

        # Use stratified K-fold to maintain class distribution
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True,
                              random_state=self.random_state)

        for train_idx, test_idx in skf.split(X, y):
            yield train_idx, test_idx


class PerformanceEvaluator:
    """Class for evaluating model performance."""

    def __init__(self):
        """Initialize PerformanceEvaluator."""
        pass

    def calculate_metrics(self, y_true, y_pred):
        """
        Calculate comprehensive performance metrics.

        Parameters:
        -----------
        y_true : numpy.ndarray
            True labels
        y_pred : numpy.ndarray
            Predicted labels

        Returns:
        --------
        dict
            Dictionary containing various performance metrics
        """
        # Handle binary and multiclass scenarios
        average = 'binary' if len(np.unique(y_true)) == 2 else 'macro'

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
            'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }

        # Calculate sensitivity and specificity for binary classification
        if len(np.unique(y_true)) == 2:
            cm = metrics['confusion_matrix']
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
                metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            else:
                metrics['sensitivity'] = metrics['recall']
                metrics['specificity'] = 0
        else:
            # For multiclass, use macro-averaged recall as sensitivity
            metrics['sensitivity'] = metrics['recall']
            metrics['specificity'] = 0

        return metrics

    def evaluate_classifier(self, classifier, X_train, y_train, X_test, y_test,
                            dr_method='PCA', target_dim=2):
        """
        Evaluate a classifier with timing information.

        Parameters:
        -----------
        classifier : object
            Classifier object with fit and predict methods
        X_train : numpy.ndarray
            Training features
        y_train : numpy.ndarray
            Training labels
        X_test : numpy.ndarray
            Test features
        y_test : numpy.ndarray
            Test labels
        dr_method : str, default='PCA'
            Dimensionality reduction method
        target_dim : int, default=2
            Target dimensionality

        Returns:
        --------
        dict
            Dictionary containing metrics and timing information
        """
        # Fit classifier
        start_time = time.time()
        classifier.fit(X_train, y_train, dr_method, target_dim)
        fit_time = time.time() - start_time

        # Make predictions
        start_time = time.time()
        y_pred = classifier.predict(X_test)
        pred_time = time.time() - start_time

        # Calculate metrics
        metrics = self.calculate_metrics(y_test, y_pred)

        # Add timing information
        metrics['fit_time'] = fit_time
        metrics['prediction_time'] = pred_time
        metrics['time_per_sample'] = pred_time / len(X_test)

        return metrics

    def cross_validate_model(self, model_class, X, y, cv_generator,
                             dr_method='PCA', target_dim=2, model_params=None):
        """
        Perform cross-validation evaluation.

        Parameters:
        -----------
        model_class : class
            Model class to instantiate
        X : numpy.ndarray
            Features
        y : numpy.ndarray
            Labels
        cv_generator : generator
            Cross-validation generator
        dr_method : str, default='PCA'
            Dimensionality reduction method
        target_dim : int, default=2
            Target dimensionality
        model_params : dict, optional
            Parameters for model instantiation

        Returns:
        --------
        dict
            Dictionary containing averaged metrics across folds
        """
        model_params = model_params or {}

        fold_metrics = []

        for fold, (train_idx, test_idx) in enumerate(cv_generator):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Instantiate model
            model = model_class(**model_params)

            # Evaluate model
            metrics = self.evaluate_classifier(model, X_train, y_train, X_test, y_test,
                                               dr_method, target_dim)
            fold_metrics.append(metrics)

        # Average metrics across folds
        avg_metrics = self._average_metrics(fold_metrics)

        return avg_metrics

    def _average_metrics(self, fold_metrics):
        """Average metrics across folds."""
        avg_metrics = {}

        # Metrics to average (excluding confusion matrices)
        scalar_metrics = ['accuracy', 'precision', 'recall', 'f1_score',
                          'sensitivity', 'specificity', 'fit_time',
                          'prediction_time', 'time_per_sample']

        for metric in scalar_metrics:
            values = [fm[metric] for fm in fold_metrics if metric in fm]
            if values:
                avg_metrics[f'avg_{metric}'] = np.mean(values)
                avg_metrics[f'std_{metric}'] = np.std(values)

        # Average confusion matrices
        if 'confusion_matrix' in fold_metrics[0]:
            cms = [fm['confusion_matrix'] for fm in fold_metrics]
            avg_metrics['avg_confusion_matrix'] = np.mean(cms, axis=0)

        return avg_metrics


def preprocess_data(X, y, normalize=True, encode_labels=True):
    """
    Preprocess data by normalizing features and encoding labels.

    Parameters:
    -----------
    X : numpy.ndarray
        Features
    y : numpy.ndarray
        Labels
    normalize : bool, default=True
        Whether to normalize features
    encode_labels : bool, default=True
        Whether to encode labels

    Returns:
    --------
    tuple of (numpy.ndarray, numpy.ndarray, dict)
        Processed features, labels, and preprocessing info
    """
    preprocessing_info = {}

    # Normalize features
    if normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        preprocessing_info['scaler'] = scaler

    # Encode labels
    if encode_labels:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        preprocessing_info['label_encoder'] = label_encoder

    return X, y, preprocessing_info
