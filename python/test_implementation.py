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

def test_basic_functionality():
    """Test basic DML functionality."""
    print("\nTesting basic DML functionality...")
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
        dml = DistanceMetricLearning(n_neighbors=5, n_points_patch=3, max_iterations=5)
        X_transformed = dml.fit_transform(X, y, dr_method='PCA', target_dim=2)

        print(f"Original shape: {X.shape}, Transformed shape: {X_transformed.shape}")
        print("✓ DML basic functionality works")
        return True
    except Exception as e:
        print(f"✗ DML basic functionality error: {e}")
        return False

def run_tests():
    """Run all tests."""
    print("Running DML Python Implementation Tests")
    print("=" * 50)

    tests = [test_imports, test_basic_functionality]
    results = []

    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append(False)

    print("\n" + "=" * 50)
    print(f"Passed: {sum(results)}/{len(results)}")

    if all(results):
        print("🎉 All tests passed! The implementation is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
