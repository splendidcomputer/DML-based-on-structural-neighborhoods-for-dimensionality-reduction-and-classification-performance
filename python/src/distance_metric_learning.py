"""
Distance Metric Learning based on Structural Neighborhoods

This module implements the Distance Metric Learning algorithm based on structural neighborhoods
for dimensionality reduction and classification performance improvement.

Paper: Distance metric learning based on structural neighborhoods for dimensionality reduction 
       and classification performance improvement
Authors: Mostafa Razavi Ghods, Mohammad Hossein Moattar, Yahya Forghani
arXiv: https://arxiv.org/abs/1902.03453
"""

import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import MDS, Isomap, LocallyLinearEmbedding
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import time
import warnings
warnings.filterwarnings('ignore')


class DistanceMetricLearning:
    """
    Distance Metric Learning based on Structural Neighborhoods

    This class implements the algorithm for learning distance metrics based on 
    the structural neighborhoods extracted from manifold learning techniques.
    """

    def __init__(self, n_neighbors=10, n_points_patch=7, max_iterations=30,
                 lambda_reg=1.0, tolerance=1e-4):
        """
        Initialize the Distance Metric Learning model.

        Parameters:
        -----------
        n_neighbors : int, default=10
            Number of nearest neighbors to consider on the manifold
        n_points_patch : int, default=7
            Number of nearest neighbors to consider on a patch
        max_iterations : int, default=30
            Maximum number of iterations for the optimization
        lambda_reg : float, default=1.0
            Regularization parameter
        tolerance : float, default=1e-4
            Convergence tolerance
        """
        self.n_neighbors = n_neighbors
        self.n_points_patch = n_points_patch
        self.max_iterations = max_iterations
        self.lambda_reg = lambda_reg
        self.tolerance = tolerance

        # Learned parameters
        self.W = None  # Linear transformation matrix
        self.t = None  # Translation vector

    def _calculate_distance_matrix(self, X):
        """
        Calculate pairwise Euclidean distance matrix.

        Parameters:
        -----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Input data

        Returns:
        --------
        numpy.ndarray of shape (n_samples, n_samples)
            Pairwise distance matrix
        """
        n_samples = X.shape[0]
        dist_matrix = np.zeros((n_samples, n_samples))

        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                dist = np.linalg.norm(X[i] - X[j], ord=2)
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist

        return dist_matrix

    def _sample_data(self, X, y, max_sample_size=1000):
        """
        Sample data to reduce computational complexity while maintaining class balance.

        Parameters:
        -----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Input features
        y : numpy.ndarray of shape (n_samples,)
            Target labels
        max_sample_size : int, default=1000
            Maximum number of samples to keep

        Returns:
        --------
        tuple of (numpy.ndarray, numpy.ndarray)
            Sampled features and labels
        """
        if X.shape[0] <= max_sample_size:
            return X, y

        unique_classes = np.unique(y)
        n_classes = len(unique_classes)

        # Find the smallest class size
        class_sizes = [np.sum(y == cls) for cls in unique_classes]
        smallest_class_size = min(class_sizes)

        # Sample equally from each class
        sampled_X = []
        sampled_y = []

        for cls in unique_classes:
            class_indices = np.where(y == cls)[0]
            if len(class_indices) > smallest_class_size:
                selected_indices = np.random.choice(class_indices,
                                                    size=smallest_class_size,
                                                    replace=False)
            else:
                selected_indices = class_indices

            sampled_X.append(X[selected_indices])
            sampled_y.append(y[selected_indices])

        sampled_X = np.vstack(sampled_X)
        sampled_y = np.hstack(sampled_y)

        return sampled_X, sampled_y

    def _apply_dimensionality_reduction(self, X, y, method='PCA', d=2):
        """
        Apply dimensionality reduction technique.

        Parameters:
        -----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Input features
        y : numpy.ndarray of shape (n_samples,)
            Target labels
        method : str, default='PCA'
            Dimensionality reduction method
        d : int, default=2
            Target dimensionality

        Returns:
        --------
        numpy.ndarray of shape (n_samples, d)
            Transformed data
        """
        if method == 'PCA':
            reducer = PCA(n_components=d)
            return reducer.fit_transform(X)

        elif method == 'LDA':
            reducer = LinearDiscriminantAnalysis(n_components=d)
            return reducer.fit_transform(X, y)

        elif method == 'MDS':
            reducer = MDS(n_components=d, random_state=42)
            return reducer.fit_transform(X)

        elif method == 'Isomap':
            reducer = Isomap(n_components=d, n_neighbors=min(
                self.n_neighbors, X.shape[0]-1))
            return reducer.fit_transform(X)

        elif method == 'LLE':
            reducer = LocallyLinearEmbedding(n_components=d,
                                             n_neighbors=min(
                                                 self.n_neighbors, X.shape[0]-1),
                                             random_state=42)
            return reducer.fit_transform(X)

        else:
            raise ValueError(
                f"Unsupported dimensionality reduction method: {method}")

    def _create_similarity_matrix(self, em_X, sampled_y):
        """
        Create similarity/dissimilarity matrix based on manifold neighborhoods.

        Parameters:
        -----------
        em_X : numpy.ndarray of shape (n_samples, n_features)
            Embedded data from manifold learning
        sampled_y : numpy.ndarray of shape (n_samples,)
            Labels for the sampled data

        Returns:
        --------
        numpy.ndarray of shape (n_samples, n_samples)
            Similarity/dissimilarity matrix
        """
        n_sampled = em_X.shape[0]

        # Calculate distance matrix and find nearest/farthest neighbors
        dist_matrix = self._calculate_distance_matrix(em_X)

        # Sort neighbors by distance
        sort_order = np.argsort(dist_matrix, axis=1)
        nn_ind = sort_order[:, 1:]  # Nearest neighbors (excluding self)
        fn_ind = sort_order[:, ::-1][:, 1:]  # Farthest neighbors

        # Initialize matrices
        Y = np.zeros((n_sampled, n_sampled))
        sd_matrix = np.zeros((n_sampled, n_sampled))

        for i in range(n_sampled):
            j = 0
            n_sim = 0
            n_dissim = 0
            sim_indices = []
            dissim_indices = []

            # Find similar and dissimilar patches
            while (n_sim < self.n_points_patch or n_dissim < self.n_points_patch) and j < nn_ind.shape[1]:

                # Look for similar points among farthest neighbors
                if (n_sim < self.n_points_patch and
                    j < fn_ind.shape[1] and
                        sampled_y[i] == sampled_y[fn_ind[i, j]]):

                    sim_indices.append(fn_ind[i, j])
                    n_sim += 1
                    Y[i, fn_ind[i, j]] = 1

                # Look for dissimilar points among nearest neighbors
                if (n_dissim < self.n_points_patch and
                    j < nn_ind.shape[1] and
                        sampled_y[i] != sampled_y[nn_ind[i, j]]):

                    dissim_indices.append(nn_ind[i, j])
                    n_dissim += 1

                j += 1

            # Set similarity/dissimilarity values
            if len(sim_indices) >= self.n_points_patch:
                sd_matrix[i, sim_indices[:self.n_points_patch]] = 1  # Similar

            if len(dissim_indices) >= self.n_points_patch:
                # Dissimilar
                sd_matrix[i, dissim_indices[:self.n_points_patch]] = -1

        return Y, sd_matrix

    def _learn_distance_metric(self, sampled_X, Y, B):
        """
        Learn the distance metric using the DLSR algorithm.

        Parameters:
        -----------
        sampled_X : numpy.ndarray of shape (n_samples, n_features)
            Sampled training data
        Y : numpy.ndarray of shape (n_samples, n_samples)
            Initial similarity matrix
        B : numpy.ndarray of shape (n_samples, n_samples)
            Similarity/dissimilarity constraint matrix

        Returns:
        --------
        tuple of (numpy.ndarray, numpy.ndarray)
            Learned transformation matrix W and translation vector t
        """
        n_sampled, D = sampled_X.shape

        # Initialize matrices
        M = np.zeros_like(B)
        W0 = np.zeros((D, Y.shape[1]))
        t0 = np.zeros(Y.shape[1])

        # Create centering matrix
        en = np.ones((n_sampled, 1))
        H = np.eye(n_sampled) - (1/n_sampled) * (en @ en.T)

        # Pre-compute U matrix
        try:
            U = np.linalg.inv(sampled_X.T @ H @ sampled_X +
                              self.lambda_reg * np.eye(D)) @ (sampled_X.T @ H)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if matrix is singular
            U = np.linalg.pinv(sampled_X.T @ H @ sampled_X +
                               self.lambda_reg * np.eye(D)) @ (sampled_X.T @ H)

        # Iterative optimization
        for iteration in range(self.max_iterations):
            # Update R
            R = Y + B * M

            # Update W
            W = U @ R

            # Update t
            t = (1/n_sampled) * R.T @ en.flatten() - \
                (1/n_sampled) * W.T @ sampled_X.T @ en.flatten()

            # Update P and M
            P = sampled_X @ W + np.outer(en.flatten(), t) - Y
            M = np.maximum(B * P, 0)

            # Check convergence
            conv_criterion = (np.linalg.norm(W - W0, 'fro')**2 +
                              np.linalg.norm(t - t0, 2)**2)

            if conv_criterion < self.tolerance:
                break

            W0 = W.copy()
            t0 = t.copy()

        return W, t

    def fit(self, X, y, dr_method='PCA', target_dim=2):
        """
        Fit the Distance Metric Learning model.

        Parameters:
        -----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Training features
        y : numpy.ndarray of shape (n_samples,)
            Training labels
        dr_method : str, default='PCA'
            Dimensionality reduction method
        target_dim : int, default=2
            Target dimensionality for embedding
        """
        # Sample data if necessary
        sampled_X, sampled_y = self._sample_data(X, y)

        # Apply dimensionality reduction
        em_X = self._apply_dimensionality_reduction(
            sampled_X, sampled_y, dr_method, target_dim)

        # Create similarity matrices
        Y, B = self._create_similarity_matrix(em_X, sampled_y)

        # Learn distance metric
        self.W, self.t = self._learn_distance_metric(sampled_X, Y, B)

        return self

    def transform(self, X):
        """
        Transform data using the learned distance metric.

        Parameters:
        -----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Input data to transform

        Returns:
        --------
        numpy.ndarray of shape (n_samples, n_transformed_features)
            Transformed data
        """
        if self.W is None or self.t is None:
            raise ValueError("Model must be fitted before transform")

        return X @ self.W + self.t

    def fit_transform(self, X, y, dr_method='PCA', target_dim=2):
        """
        Fit the model and transform the data.

        Parameters:
        -----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Training features
        y : numpy.ndarray of shape (n_samples,)
            Training labels
        dr_method : str, default='PCA'
            Dimensionality reduction method
        target_dim : int, default=2
            Target dimensionality for embedding

        Returns:
        --------
        numpy.ndarray of shape (n_samples, n_transformed_features)
            Transformed training data
        """
        self.fit(X, y, dr_method, target_dim)
        return self.transform(X)


class DMLClassifier:
    """
    Wrapper class that combines Distance Metric Learning with classification.
    """

    def __init__(self, dml_params=None, classifier_type='knn', classifier_params=None):
        """
        Initialize the DML Classifier.

        Parameters:
        -----------
        dml_params : dict, optional
            Parameters for the Distance Metric Learning
        classifier_type : str, default='knn'
            Type of classifier ('knn' or 'svm')
        classifier_params : dict, optional
            Parameters for the classifier
        """
        self.dml_params = dml_params or {}
        self.classifier_type = classifier_type
        self.classifier_params = classifier_params or {}

        self.dml = DistanceMetricLearning(**self.dml_params)

        if classifier_type == 'knn':
            self.classifier = KNeighborsClassifier(**self.classifier_params)
        elif classifier_type == 'svm':
            self.classifier = SVC(**self.classifier_params)
        else:
            raise ValueError(f"Unsupported classifier type: {classifier_type}")

    def fit(self, X, y, dr_method='PCA', target_dim=2):
        """
        Fit the DML classifier.

        Parameters:
        -----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Training features
        y : numpy.ndarray of shape (n_samples,)
            Training labels
        dr_method : str, default='PCA'
            Dimensionality reduction method
        target_dim : int, default=2
            Target dimensionality for embedding
        """
        # Transform data using DML
        X_transformed = self.dml.fit_transform(X, y, dr_method, target_dim)

        # Fit classifier on transformed data
        self.classifier.fit(X_transformed, y)

        return self

    def predict(self, X):
        """
        Make predictions using the fitted DML classifier.

        Parameters:
        -----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Test features

        Returns:
        --------
        numpy.ndarray of shape (n_samples,)
            Predicted labels
        """
        # Transform test data
        X_transformed = self.dml.transform(X)

        # Make predictions
        return self.classifier.predict(X_transformed)

    def score(self, X, y):
        """
        Calculate accuracy score.

        Parameters:
        -----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Test features
        y : numpy.ndarray of shape (n_samples,)
            True labels

        Returns:
        --------
        float
            Accuracy score
        """
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
