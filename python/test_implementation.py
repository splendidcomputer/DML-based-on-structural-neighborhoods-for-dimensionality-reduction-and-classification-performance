"""
Simple test script to verify the Python DML implementation works correctly.
"""

import numpy as np
import sys
import os

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from distance_metric_learning import DistanceMetricLearning, DMLClassifier
        from data_utils import DataLoader, preprocess_data
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False


def test_synthetic_data_generation():
    """Test synthetic data generation."""
    print("\nTesting synthetic data generation...")
    try:
        from data_utils import DataLoader

        loader = DataLoader()
        X, y = loader._generate_synthetic_data(100, 10, 3)

        assert X.shape == (100, 10), f"Expected shape (100, 10), got {X.shape}"
        assert y.shape == (100,), f"Expected shape (100,), got {y.shape}"
        assert len(
            np.unique(y)) <= 3, f"Expected at most 3 classes, got {len(np.unique(y))}"

        print("✓ Synthetic data generation works")
        return True
    except Exception as e:
        print(f"✗ Synthetic data generation error: {e}")
        return False


def test_dml_basic_functionality():
    """Test basic DML functionality."""
    print("\nTesting DML basic functionality...")
    try:
        from distance_metric_learning import DistanceMetricLearning
        from data_utils import preprocess_data

        # Generate simple test data
        np.random.seed(42)
        X = np.random.randn(50, 8)
        y = np.random.randint(0, 2, 50)

        # Preprocess
        X, y, _ = preprocess_data(X, y)

        # Create and fit DML
        dml = DistanceMetricLearning(
            n_neighbors=5, n_points_patch=3, max_iterations=5)
        X_transformed = dml.fit_transform(X, y, dr_method='PCA', target_dim=2)

        assert X_transformed.shape[0] == X.shape[0], "Sample count should be preserved"
        assert dml.W is not None, "Transformation matrix W should be learned"
        assert dml.t is not None, "Translation vector t should be learned"

        print("✓ DML basic functionality works")
        return True
    except Exception as e:
        print(f"✗ DML basic functionality error: {e}")
        return False


def test_dml_classifier():
    """Test DML classifier."""
    print("\nTesting DML classifier...")
    try:
        from distance_metric_learning import DMLClassifier
        from data_utils import preprocess_data
        from sklearn.model_selection import train_test_split

        # Generate test data
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 3, 100)

        # Preprocess
        X, y, _ = preprocess_data(X, y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42)

        # Create and test classifier
        classifier = DMLClassifier(
            dml_params={'n_neighbors': 5,
                        'n_points_patch': 3, 'max_iterations': 5},
            classifier_type='knn',
            classifier_params={'n_neighbors': 3}
        )

        # Fit and predict
        classifier.fit(X_train, y_train, dr_method='PCA', target_dim=3)
        y_pred = classifier.predict(X_test)

        assert len(y_pred) == len(
            y_test), "Prediction count should match test count"
        assert all(pred in np.unique(y_train)
                   for pred in y_pred), "Predictions should be valid labels"

        print("✓ DML classifier works")
        return True
    except Exception as e:
        print(f"✗ DML classifier error: {e}")
        return False


def test_multiple_dr_methods():
    """Test multiple dimensionality reduction methods."""
    print("\nTesting multiple DR methods...")
    try:
        from distance_metric_learning import DistanceMetricLearning
        from data_utils import preprocess_data

        # Generate test data
        np.random.seed(42)
        X = np.random.randn(60, 8)
        y = np.random.randint(0, 2, 60)

        # Preprocess
        X, y, _ = preprocess_data(X, y)

        methods = ['PCA', 'LDA']  # Start with basic methods

        for method in methods:
            print(f"  Testing {method}...")
            dml = DistanceMetricLearning(
                n_neighbors=5, n_points_patch=3, max_iterations=3)
            X_transformed = dml.fit_transform(
                X, y, dr_method=method, target_dim=2)

            assert X_transformed.shape[0] == X.shape[0], f"{method}: Sample count should be preserved"
            assert X_transformed.shape[1] == 2 or X_transformed.shape[
                1] <= X.shape[1], f"{method}: Dimensionality should be reduced"

        print("✓ Multiple DR methods work")
        return True
    except Exception as e:
        print(f"✗ Multiple DR methods error: {e}")
        return False


def run_all_tests():
    """Run all tests."""
    print("Running DML Python Implementation Tests")
    print("=" * 50)

    tests = [
        test_imports,
        test_synthetic_data_generation,
        test_dml_basic_functionality,
        test_dml_classifier,
        test_multiple_dr_methods
    ]

    results = []

    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append(False)

    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"Passed: {sum(results)}/{len(results)}")

    if all(results):
        print("🎉 All tests passed! The implementation is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
