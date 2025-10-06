"""
Additional classifier implementations used in the original MATLAB code.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time


def find_knn(test_point, training_data, k, p=1):
    """
    Find k nearest neighbors using L-p norm.

    Parameters:
    -----------
    test_point : numpy.ndarray
        Test point to find neighbors for
    training_data : numpy.ndarray
        Training data points
    k : int
        Number of neighbors to find
    p : int, default=1
        Norm to use (1 for L1, 2 for L2)

    Returns:
    --------
    numpy.ndarray
        Indices of k nearest neighbors
    """
    n_train = training_data.shape[0]
    distances = np.zeros(n_train)

    for i in range(n_train):
        distances[i] = np.linalg.norm(test_point - training_data[i], ord=p)

    # Sort and return k nearest neighbor indices
    nearest_indices = np.argsort(distances)
    return nearest_indices[:k]


class MultiClassSVM:
    """
    Multi-class SVM classifier (replicating MultiSVM.m functionality).
    """

    def __init__(self, kernel='rbf', C=1.0, gamma='scale', random_state=42):
        """
        Initialize Multi-class SVM.

        Parameters:
        -----------
        kernel : str, default='rbf'
            Kernel type
        C : float, default=1.0
            Regularization parameter
        gamma : str or float, default='scale'
            Kernel coefficient
        random_state : int, default=42
            Random state for reproducibility
        """
        self.svm = SVC(kernel=kernel, C=C, gamma=gamma,
                       random_state=random_state, decision_function_shape='ovr')

    def fit(self, X_train, y_train):
        """
        Fit the SVM classifier.

        Parameters:
        -----------
        X_train : numpy.ndarray
            Training features
        y_train : numpy.ndarray
            Training labels
        """
        self.svm.fit(X_train, y_train)
        return self

    def predict(self, X_test):
        """
        Predict using the trained SVM.

        Parameters:
        -----------
        X_test : numpy.ndarray
            Test features

        Returns:
        --------
        numpy.ndarray
            Predicted labels
        """
        return self.svm.predict(X_test)


class DistanceKNN:
    """
    Distance-based k-NN classifier (replicating the original k-NN implementation).
    """

    def __init__(self, n_neighbors=7, p=1):
        """
        Initialize Distance k-NN classifier.

        Parameters:
        -----------
        n_neighbors : int, default=7
            Number of neighbors to consider
        p : int, default=1
            Norm to use for distance calculation
        """
        self.n_neighbors = n_neighbors
        self.p = p
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """
        Fit the k-NN classifier.

        Parameters:
        -----------
        X_train : numpy.ndarray
            Training features
        y_train : numpy.ndarray
            Training labels
        """
        self.X_train = X_train
        self.y_train = y_train
        return self

    def predict(self, X_test):
        """
        Predict using k-NN.

        Parameters:
        -----------
        X_test : numpy.ndarray
            Test features

        Returns:
        --------
        numpy.ndarray
            Predicted labels
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Classifier must be fitted before prediction")

        n_test = X_test.shape[0]
        predictions = np.zeros(n_test)

        for i in range(n_test):
            # Find k nearest neighbors
            neighbor_indices = find_knn(
                X_test[i], self.X_train, self.n_neighbors, self.p)

            # Get neighbor labels and find mode (most common label)
            neighbor_labels = self.y_train[neighbor_indices]

            # Calculate mode
            unique_labels, counts = np.unique(
                neighbor_labels, return_counts=True)
            predictions[i] = unique_labels[np.argmax(counts)]

        return predictions


class SimilarityKNN:
    """
    Similarity-based k-NN classifier (replicating sim-k-NN implementation).
    """

    def __init__(self, n_neighbors=7):
        """
        Initialize Similarity k-NN classifier.

        Parameters:
        -----------
        n_neighbors : int, default=7
            Number of neighbors to consider
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """
        Fit the similarity k-NN classifier.

        Parameters:
        -----------
        X_train : numpy.ndarray
            Training features  
        y_train : numpy.ndarray
            Training labels
        """
        self.X_train = X_train
        self.y_train = y_train
        return self

    def predict(self, X_test):
        """
        Predict using similarity k-NN.

        Parameters:
        -----------
        X_test : numpy.ndarray
            Test features

        Returns:
        --------
        numpy.ndarray
            Predicted labels
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Classifier must be fitted before prediction")

        n_test = X_test.shape[0]
        predictions = np.zeros(n_test)

        # Sort test samples by descending order (similarity-based)
        sorted_indices = np.argsort(X_test, axis=1)[:, ::-1]

        for i in range(n_test):
            # Get top k similar neighbors
            neighbor_indices = sorted_indices[i, :self.n_neighbors]

            # Ensure indices are within training data range
            valid_indices = neighbor_indices[neighbor_indices < len(
                self.y_train)]

            if len(valid_indices) > 0:
                neighbor_labels = self.y_train[valid_indices]

                # Calculate mode
                unique_labels, counts = np.unique(
                    neighbor_labels, return_counts=True)
                predictions[i] = unique_labels[np.argmax(counts)]
            else:
                # Fallback to first training label if no valid neighbors
                predictions[i] = self.y_train[0]

        return predictions


def evaluate_classifier_timing(classifier, X_train, y_train, X_test, y_test):
    """
    Evaluate classifier with detailed timing information.

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

    Returns:
    --------
    dict
        Dictionary containing performance metrics and timing
    """
    # Fit the classifier
    start_time = time.time()
    classifier.fit(X_train, y_train)
    fit_time = time.time() - start_time

    # Make predictions with timing
    start_time = time.time()
    y_pred = classifier.predict(X_test)
    pred_time = time.time() - start_time

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    time_per_sample = pred_time / len(X_test)

    return {
        'accuracy': accuracy,
        'fit_time': fit_time,
        'prediction_time': pred_time,
        'time_per_sample': time_per_sample,
        'y_pred': y_pred
    }
