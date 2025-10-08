""""""

Distance Metric Learning Python ImplementationDistance Metric Learning Python Implementation



This package provides a Python implementation of the Distance Metric LearningThis package provides a Python implementation of the Distance Metric Learning

algorithm based on structural neighborhoods.algorithm based on structural neighborhoods.

""""""



__version__ = "1.0.0"__version__ = "1.0.0"

__author__ = "Python Implementation"__author__ = "Python Implementation"



# Make key classes available at package level# Make key classes available at package level

try:try:

    from src.distance_metric_learning import DistanceMetricLearning, DMLClassifier    from src.distance_metric_learning import DistanceMetricLearning, DMLClassifier

    from utils.data_utils import DataLoader, preprocess_data    from utils.data_utils import DataLoader, preprocess_data

    from utils.classifiers import MultiClassSVM, DistanceKNN, SimilarityKNN    from utils.classifiers import MultiClassSVM, DistanceKNN, SimilarityKNN



    __all__ = [    __all__ = [

        'DistanceMetricLearning',        'DistanceMetricLearning',

        'DMLClassifier',         'DMLClassifier',

        'DataLoader',        'DataLoader',

        'preprocess_data',        'preprocess_data',

        'MultiClassSVM',        'MultiClassSVM',

        'DistanceKNN',        'DistanceKNN',

        'SimilarityKNN'        'SimilarityKNN'

    ]    ]

except ImportError:except ImportError:

    # Allow package to be imported even if dependencies are missing    # Allow package to be imported even if dependencies are missing

    __all__ = []    __all__ = []
