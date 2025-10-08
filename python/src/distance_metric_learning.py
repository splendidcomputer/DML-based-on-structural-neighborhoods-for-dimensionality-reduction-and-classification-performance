"""
Distance Metric Learning based on Structural Neighborhoods

This module implements the Distance Metric Learning algorithm from the paper:
"Distance metric learning based on structural neighborhoods for dimensionality 
reduction and classification performance improvement" by Mostafa Razavi Ghods et al.
"""

import numpy as np
import warnings
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import MDS, Isomap, LocallyLinearEmbedding
from sklearn.neighbors import NearestNeighbors
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from scipy.optimize import minimize
import time

warnings.filterwarnings('ignore', category=FutureWarning)


class SimilarityKNNClassifier(BaseEstimator):
    """Similarity-based k-NN classifier following MATLAB sim-k-NN implementation."""
    
    def __init__(self, n_neighbors=7):
        self.n_neighbors = n_neighbors
        self.X_train_ = None
        self.y_train_ = None
        
    def fit(self, X, y):
        """Fit the similarity-based k-NN classifier."""
        self.X_train_ = check_array(X)
        self.y_train_ = np.array(y)
        return self
        
    def predict(self, X):
        """Predict using similarity-based k-NN following MATLAB implementation."""
        X = check_array(X)
        
        if self.X_train_ is None:
            raise ValueError("Classifier must be fitted before making predictions")
        
        n_test = X.shape[0]
        predictions = np.zeros(n_test, dtype=self.y_train_.dtype)
        
        for i in range(n_test):
            # Following MATLAB: [~, nnInd] = sort(mappedTst, 2, 'descend');
            # This suggests sorting by similarity/distance scores
            
            # Compute similarities (negative distances for descending sort)
            test_point = X[i].reshape(1, -1)
            similarities = -np.linalg.norm(self.X_train_ - test_point, axis=1)
            
            # Sort in descending order (most similar first)
            sorted_indices = np.argsort(similarities)[::-1]
            
            # Get k nearest neighbors
            k_neighbors = min(self.n_neighbors, len(sorted_indices))
            neighbor_indices = sorted_indices[:k_neighbors]
            neighbor_labels = self.y_train_[neighbor_indices]
            
            # Predict using majority vote
            unique_labels, counts = np.unique(neighbor_labels, return_counts=True)
            predictions[i] = unique_labels[np.argmax(counts)]
        
        return predictions


class AutoencoderTransformer(BaseEstimator, TransformerMixin):
    """Simple Autoencoder for dimensionality reduction using MLPRegressor."""
    
    def __init__(self, n_components=2, hidden_layer_size=None, random_state=42):
        self.n_components = n_components
        self.hidden_layer_size = hidden_layer_size
        self.random_state = random_state
        self.encoder_ = None
        self.decoder_ = None
        
    def fit(self, X, y=None):
        """Fit the autoencoder."""
        X = check_array(X)
        n_features = X.shape[1]
        
        if self.hidden_layer_size is None:
            self.hidden_layer_size = max(self.n_components + 1, n_features // 2)
            
        # Create encoder (input -> hidden -> bottleneck)
        self.encoder_ = MLPRegressor(
            hidden_layer_sizes=(self.hidden_layer_size, self.n_components),
            activation='tanh',
            solver='adam',
            alpha=0.01,
            max_iter=500,
            random_state=self.random_state,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        # Train encoder to map to reduced space and back
        # For simplicity, we'll use PCA as a proxy target for the bottleneck
        pca = PCA(n_components=self.n_components, random_state=self.random_state)
        target_reduced = pca.fit_transform(X)
        
        # Train the encoder
        self.encoder_.fit(X, target_reduced)
        
        return self
        
    def transform(self, X):
        """Transform data to reduced dimension."""
        if self.encoder_ is None:
            raise ValueError("Autoencoder must be fitted before transforming.")
        X = check_array(X)
        return self.encoder_.predict(X)
        
    def fit_transform(self, X, y=None):
        """Fit and transform data."""
        return self.fit(X, y).transform(X)


class DistanceMetricLearning(BaseEstimator, TransformerMixin):
    """
    Distance Metric Learning based on structural neighborhoods.
    
    Parameters:
    -----------
    n_neighbors : int, default=10
        Number of nearest neighbors to consider in manifold space
    n_points_patch : int, default=7  
        Number of points in each neighborhood patch
    max_iterations : int, default=30
        Maximum number of optimization iterations
    lambda_reg : float, default=1.0
        Regularization parameter for optimization
    reduction_perc : float, default=1.0
        Percentage of data to use for training (for large datasets)
    verbose : bool, default=False
        Whether to print progress information
    """
    
    def __init__(self, n_neighbors=10, n_points_patch=7, max_iterations=30, 
                 lambda_reg=1.0, reduction_perc=1.0, verbose=False):
        self.n_neighbors = n_neighbors
        self.n_points_patch = n_points_patch
        self.max_iterations = max_iterations
        self.lambda_reg = lambda_reg
        self.reduction_perc = reduction_perc
        self.verbose = verbose
        
        # Will be set during fitting
        self.W = None
        self.t = None
        self.scaler = None
        self.dr_method = None
        self.target_dim = None
        self.manifold_transformer = None
        
    def fit(self, X, y, dr_method='PCA', target_dim=2):
        """Fit the Distance Metric Learning model."""
        X, y = check_X_y(X, y)
        
        self.dr_method = dr_method
        self.target_dim = target_dim
        
        if self.verbose:
            print(f"Starting DML with {dr_method}, target_dim={target_dim}")
            
        # Step 1: Data preprocessing and normalization
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Step 2: Data reduction for large datasets
        if self.reduction_perc < 1.0:
            n_samples = int(len(X_scaled) * self.reduction_perc)
            indices = np.random.choice(len(X_scaled), n_samples, replace=False)
            X_scaled = X_scaled[indices]
            y = y[indices]
            
        # Step 3: Manifold learning (dimensionality reduction)
        X_manifold = self._apply_dimensionality_reduction(X_scaled, y, dr_method, target_dim)
        
        if self.verbose:
            print(f"Manifold shape: {X_manifold.shape}")
            
        # Step 4: Learn neighborhood structure on manifold
        neighborhoods = self._learn_neighborhood_structure(X_manifold, X_scaled)
        
        # Step 5: Learn distance metric using DLSR optimization
        self.W, self.t = self._learn_distance_metric(X_scaled, neighborhoods)
        
        if self.verbose:
            print("DML fitting completed")
            
        return self
        
    def transform(self, X):
        """Transform data using learned distance metric."""
        if self.W is None or self.t is None:
            raise ValueError("Model must be fitted before transforming data")
            
        X = check_array(X)
        X_scaled = self.scaler.transform(X)
        
        # Apply learned transformation
        # Ensure broadcasting compatibility
        if len(self.t.shape) == 1 and self.t.shape[0] == X_scaled.shape[1]:
            # t is per-feature
            X_with_translation = X_scaled + self.t
        else:
            # t is scalar or incompatible, use zero translation
            X_with_translation = X_scaled
            
        # Apply linear transformation W
        if self.W.shape[1] == X_with_translation.shape[1]:
            X_transformed = np.dot(X_with_translation, self.W.T)
        else:
            # Shape mismatch, just return scaled data
            if self.verbose:
                print(f"Warning: Shape mismatch in transformation, returning scaled data")
            X_transformed = X_scaled
        
        return X_transformed
        
    def fit_transform(self, X, y, dr_method='PCA', target_dim=2):
        """Fit the model and transform the training data."""
        return self.fit(X, y, dr_method, target_dim).transform(X)
        
    def _apply_dimensionality_reduction(self, X, y, method, target_dim):
        """Apply dimensionality reduction to input data."""
        n_features = X.shape[1]
        n_samples = X.shape[0]
        
        # Adjust target dimension based on data constraints
        if method == 'LDA':
            n_classes = len(np.unique(y))
            max_dim = min(n_features, n_classes - 1)
            target_dim = min(target_dim, max_dim)
        else:
            target_dim = min(target_dim, n_features, n_samples - 1)
            
        if target_dim <= 0:
            target_dim = 1
            
        try:
            if method == 'PCA':
                self.manifold_transformer = PCA(n_components=target_dim, random_state=42)
            elif method == 'LDA':
                self.manifold_transformer = LinearDiscriminantAnalysis(n_components=target_dim)
            elif method == 'MDS':
                self.manifold_transformer = MDS(n_components=target_dim, random_state=42, 
                                              dissimilarity='euclidean', max_iter=300)
            elif method == 'Isomap':
                n_neighbors = min(self.n_neighbors, n_samples - 1)
                self.manifold_transformer = Isomap(n_neighbors=n_neighbors, 
                                                 n_components=target_dim)
            elif method == 'LLE':
                n_neighbors = min(self.n_neighbors, n_samples - 1)
                self.manifold_transformer = LocallyLinearEmbedding(
                    n_neighbors=n_neighbors, n_components=target_dim, random_state=42)
            elif method == 'KernelPCA':
                self.manifold_transformer = KernelPCA(n_components=target_dim, 
                                                    kernel='rbf', random_state=42)
            elif method == 'Autoencoder':
                # Simple autoencoder using MLPRegressor
                hidden_layer_size = max(target_dim + 1, n_features // 2)
                self.manifold_transformer = AutoencoderTransformer(
                    n_components=target_dim, hidden_layer_size=hidden_layer_size)
            else:
                raise ValueError(f"Unknown DR method: {method}")
                
            # Fit and transform
            if method == 'LDA':
                X_manifold = self.manifold_transformer.fit_transform(X, y)
            else:
                X_manifold = self.manifold_transformer.fit_transform(X)
                
        except Exception as e:
            if self.verbose:
                print(f"Warning: {method} failed ({e}), falling back to PCA")
            # Fallback to PCA
            self.manifold_transformer = PCA(n_components=target_dim, random_state=42)
            X_manifold = self.manifold_transformer.fit_transform(X)
            
        return X_manifold
        
    def _learn_neighborhood_structure(self, X_manifold, X_original):
        """Learn neighborhood structure from manifold representation."""
        n_samples = X_manifold.shape[0]
        
        # Find k-nearest neighbors on manifold
        n_neighbors = min(self.n_neighbors, n_samples - 1)
        nn_manifold = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm='ball_tree')
        nn_manifold.fit(X_manifold)
        
        # Get neighborhoods (excluding self)
        _, manifold_indices = nn_manifold.kneighbors(X_manifold)
        manifold_indices = manifold_indices[:, 1:]  # Remove self-indices
        
        # Build neighborhood structure
        neighborhoods = {}
        
        for i in range(n_samples):
            # Get manifold neighbors
            manifold_neighbors = manifold_indices[i]
            
            # Create patches of neighbors
            n_points = min(self.n_points_patch, len(manifold_neighbors))
            
            neighborhoods[i] = {
                'similar': manifold_neighbors[:n_points],
                'dissimilar': self._find_dissimilar_points(i, manifold_neighbors, n_samples)
            }
            
        return neighborhoods
        
    def _find_dissimilar_points(self, point_idx, similar_indices, n_samples):
        """Find dissimilar points (not in similar set)."""
        all_indices = set(range(n_samples))
        similar_set = set(similar_indices) | {point_idx}
        dissimilar_indices = list(all_indices - similar_set)
        
        # Randomly sample some dissimilar points
        n_dissimilar = min(len(dissimilar_indices), self.n_points_patch)
        if n_dissimilar > 0:
            return np.random.choice(dissimilar_indices, n_dissimilar, replace=False)
        else:
            return np.array([])
            
    def _learn_distance_metric(self, X, neighborhoods):
        """Learn distance metric using DLSR optimization (following MATLAB implementation)."""
        n_samples, n_features = X.shape
        
        # Create similar/dissimilar matrix Y and similarity/dissimilarity matrix B
        Y = np.zeros((n_samples, n_samples))
        B = np.zeros((n_samples, n_samples))
        
        for i, neighborhood in neighborhoods.items():
            # Similar points (should be close, target distance = 0)
            for j in neighborhood['similar']:
                if j < n_samples:
                    Y[i, j] = 0.0  # Target distance
                    B[i, j] = 1.0  # Similarity weight
                    
            # Dissimilar points (should be far, target distance = large value)
            for j in neighborhood['dissimilar']:
                if j < n_samples:
                    Y[i, j] = 1.0  # Target distance (normalized)
                    B[i, j] = -0.1  # Dissimilarity weight (negative)
        
        # DLSR Algorithm (following MATLAB implementation)
        max_iterations = min(self.max_iterations, 30)
        
        # Initialize matrices
        M = np.zeros_like(B)
        W = np.zeros((n_features, n_samples))
        t = np.zeros(n_samples)
        
        # Centering matrix
        en = np.ones(n_samples)
        H = np.eye(n_samples) - (1/n_samples) * np.outer(en, en)
        
        # Regularization parameter
        lambda_reg = self.lambda_reg if self.lambda_reg > 0 else 1.0
        
        # Pre-compute U matrix
        try:
            U = np.linalg.solve(
                X.T @ H @ X + lambda_reg * np.eye(n_features),
                X.T @ H
            )
        except np.linalg.LinAlgError:
            if self.verbose:
                print("Warning: Singular matrix encountered, using pseudo-inverse")
            U = np.linalg.pinv(X.T @ H @ X + lambda_reg * np.eye(n_features)) @ (X.T @ H)
        
        # Iterative optimization
        W_prev = W.copy()
        t_prev = t.copy()
        
        for iteration in range(max_iterations):
            # Update R
            R = Y + B * M
            
            # Update W
            W = U @ R
            
            # Update t  
            t = (1/n_samples) * (R.T @ en) - (1/n_samples) * (W.T @ X.T @ en)
            
            # Update P
            P = X @ W + np.outer(en, t) - Y
            
            # Update M
            M = np.maximum(B * P, 0)
            
            # Check convergence
            w_change = np.linalg.norm(W - W_prev, 'fro')**2
            t_change = np.linalg.norm(t - t_prev)**2
            
            if w_change + t_change < 1e-4:
                if self.verbose:
                    print(f"DLSR converged after {iteration + 1} iterations")
                break
                
            W_prev = W.copy()
            t_prev = t.copy()
        
        # Return the transformation matrix and translation vector
        # For compatibility with the transform method, we need to reshape appropriately
        # The MATLAB algorithm produces W: (n_features, n_samples) and t: (n_samples,)
        # But for transformation, we need to extract a proper feature transformation
        
        # Use the first few components as the transformation matrix
        target_dim = min(W.shape[1], W.shape[0])  # Take minimum to avoid shape issues
        if target_dim > 0:
            W_transform = W[:, :target_dim].T  # (target_dim, n_features)
            t_transform = np.mean(t[:target_dim]) * np.ones(n_features)  # (n_features,)
        else:
            # Fallback: use identity transformation
            W_transform = np.eye(min(3, n_features))
            t_transform = np.zeros(n_features)
        
        return W_transform, t_transform


class DMLClassifier(BaseEstimator):
    """Complete DML classifier that combines distance metric learning with classification."""
    
    def __init__(self, dml_params=None, classifier_type='knn', classifier_params=None):
        self.dml_params = dml_params or {}
        self.classifier_type = classifier_type
        self.classifier_params = classifier_params or {}
        
        # Initialize components
        self.dml = DistanceMetricLearning(**self.dml_params)
        self.classifier = None
        
    def fit(self, X, y, dr_method='PCA', target_dim=2):
        """Fit the DML classifier."""
        # Apply DML transformation
        X_transformed = self.dml.fit_transform(X, y, dr_method, target_dim)
        
        # Initialize classifier
        if self.classifier_type == 'knn':
            from sklearn.neighbors import KNeighborsClassifier
            self.classifier = KNeighborsClassifier(**self.classifier_params)
        elif self.classifier_type == 'sim-knn':
            # Similarity-based k-NN classifier (following MATLAB implementation)
            self.classifier = SimilarityKNNClassifier(**self.classifier_params)
        elif self.classifier_type == 'svm':
            from sklearn.svm import SVC
            self.classifier = SVC(**self.classifier_params)
        else:
            raise ValueError(f"Unknown classifier type: {self.classifier_type}")
            
        # Fit classifier on transformed data
        self.classifier.fit(X_transformed, y)
        
        return self
        
    def predict(self, X):
        """Predict using the fitted DML classifier."""
        if self.classifier is None:
            raise ValueError("Classifier must be fitted before making predictions")
            
        # Transform test data
        X_transformed = self.dml.transform(X)
        
        # Predict using classifier
        return self.classifier.predict(X_transformed)
