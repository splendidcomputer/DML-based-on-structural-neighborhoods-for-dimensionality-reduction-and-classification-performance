"""
Distance Metric Learning Python Implementation

This package provides a Python implementation of the Distance Metric Learning
algorithm based on structural neighborhoods.
"""

__version__ = "1.0.0"
__author__ = "Python Implementation"

# Make key classes available at package level
try:
    from src.distance_metric_learning import DistanceMetricLearning, DMLClassifier
    from utils.data_utils import DataLoader, preprocess_data
    from utils.classifiers import MultiClassSVM, DistanceKNN, SimilarityKNN

    __all__ = [
        'DistanceMetricLearning',
        'DMLClassifier',
        'DataLoader',
        'preprocess_data',
        'MultiClassSVM',
        'DistanceKNN',
        'SimilarityKNN'
    ]
except ImportError:
    # Allow package to be imported even if dependencies are missing
    __all__ = []
