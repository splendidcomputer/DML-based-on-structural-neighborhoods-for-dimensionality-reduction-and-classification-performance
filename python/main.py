"""
Main experiment runner for Distance Metric Learning experiments.
"""

import numpy as np
import pandas as pd
import time
import os
import sys
from pathlib import Path
import warnings

# Add current directory to Python path
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

# DML imports
from src.distance_metric_learning import DistanceMetricLearning, DMLClassifier
from utils.data_utils import DataLoader, preprocess_data, CrossValidator

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class DMLExperimentRunner:
    """Experiment runner for DML algorithm evaluation."""
    
    def __init__(self, results_dir="results", verbose=True):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.verbose = verbose
        
        # Initialize components
        self.data_loader = DataLoader()
        self.cv = CrossValidator(n_splits=5)  # Use 5-fold for faster results
        
        # Results storage
        self.all_results = []
        
    def run_single_experiment(self, dataset_name, dr_method, target_dim, classifier_type):
        """Run a single DML experiment."""
        if self.verbose:
            print(f"Running: {dataset_name} + {dr_method} + {classifier_type} (dim={target_dim})")
            
        start_time = time.time()
        
        try:
            # Load and preprocess data
            X, y = self.data_loader.load_dataset(dataset_name)
            X, y, preprocessors = preprocess_data(X, y)
            
            # Create DML classifier
            dml_clf = DMLClassifier(
                dml_params={'n_neighbors': 10, 'n_points_patch': 7, 'max_iterations': 10},
                classifier_type=classifier_type,
                classifier_params={'n_neighbors': 5} if classifier_type == 'knn' else {}
            )
            
            # Perform cross-validation
            cv_results = self.cv.cross_validate(dml_clf, X, y)
            
            # Calculate timing
            total_time = time.time() - start_time
            
            # Store results
            result = {
                'dataset': dataset_name,
                'dr_method': dr_method,
                'target_dim': target_dim,
                'classifier': classifier_type,
                'accuracy_mean': cv_results['test_accuracy_mean'],
                'accuracy_std': cv_results['test_accuracy_std'],
                'total_time': total_time,
                'n_samples': len(X),
                'n_features': X.shape[1],
                'n_classes': len(np.unique(y)),
            }
            
            if self.verbose:
                print(f"  Accuracy: {result['accuracy_mean']:.4f} ± {result['accuracy_std']:.4f}")
                print(f"  Time: {result['total_time']:.2f}s")
                
            return result
            
        except Exception as e:
            if self.verbose:
                print(f"  Error: {e}")
            return {
                'dataset': dataset_name,
                'dr_method': dr_method,
                'target_dim': target_dim,
                'classifier': classifier_type,
                'accuracy_mean': 0.0,
                'accuracy_std': 0.0,
                'total_time': 0.0,
                'error': str(e)
            }
            
    def run_experiments(self, datasets, dr_methods, classifiers, target_dims):
        """Run experiments across all combinations."""
        total_experiments = len(datasets) * len(dr_methods) * len(classifiers) * len(target_dims)
        
        if self.verbose:
            print(f"Starting {total_experiments} experiments...")
            print(f"Datasets: {datasets}")
            print(f"DR Methods: {dr_methods}")
            print(f"Classifiers: {classifiers}")
            print(f"Target Dims: {target_dims}")
            print("-" * 50)
            
        experiment_count = 0
        
        # Run all combinations
        from itertools import product
        for dataset, dr_method, classifier, target_dim in product(datasets, dr_methods, classifiers, target_dims):
            experiment_count += 1
            
            if self.verbose:
                print(f"[{experiment_count}/{total_experiments}]", end=" ")
                
            result = self.run_single_experiment(dataset, dr_method, target_dim, classifier)
            self.all_results.append(result)
            
        if self.verbose:
            print("-" * 50)
            print(f"Completed {experiment_count} experiments")
            
        # Save results
        self._save_results()
        self._generate_summary()
        
    def _save_results(self):
        """Save experiment results to CSV."""
        if not self.all_results:
            return
            
        results_df = pd.DataFrame(self.all_results)
        results_file = self.results_dir / "all_results.csv"
        results_df.to_csv(results_file, index=False)
        
        if self.verbose:
            print(f"Results saved to {results_file}")
            
    def _generate_summary(self):
        """Generate summary reports."""
        if not self.all_results:
            return
            
        results_df = pd.DataFrame(self.all_results)
        
        # Overall summary
        print("\n" + "=" * 60)
        print("EXPERIMENT SUMMARY")
        print("=" * 60)
        
        # Best results per dataset
        print("\nBest Results per Dataset:")
        print("-" * 40)
        for dataset in results_df['dataset'].unique():
            dataset_results = results_df[results_df['dataset'] == dataset]
            best_idx = dataset_results['accuracy_mean'].idxmax()
            best = dataset_results.loc[best_idx]
            
            print(f"{dataset:12} | {best['dr_method']:3} + {best['classifier']:3} | "
                  f"Acc: {best['accuracy_mean']:.3f} ± {best['accuracy_std']:.3f} | "
                  f"Time: {best['total_time']:.1f}s")
        
        # Method comparison
        print("\nMethod Comparison (Average Accuracy):")
        print("-" * 40)
        method_stats = results_df.groupby(['dr_method', 'classifier'])['accuracy_mean'].agg(['mean', 'std', 'count'])
        for (dr_method, classifier), stats in method_stats.iterrows():
            print(f"{dr_method:3} + {classifier:3} | "
                  f"Avg: {stats['mean']:.3f} ± {stats['std']:.3f} | "
                  f"Experiments: {int(stats['count'])}")
        
        print("\n" + "=" * 60)


def run_quick_demo():
    """Run a quick demonstration of the DML algorithm."""
    print("=== DML Quick Demo ===")
    
    runner = DMLExperimentRunner(verbose=True)
    
    # Run on a small subset for demonstration
    runner.run_experiments(
        datasets=['Iris', 'Wine'],
        dr_methods=['PCA', 'LDA', 'MDS', 'Isomap', 'LLE', 'KernelPCA', 'Autoencoder'],
        classifiers=['knn'],
        target_dims=[2, 3]
    )
    
    print("\nDemo completed! Check the 'results' directory for outputs.")


def run_comprehensive_experiments():
    """Run comprehensive experiments."""
    print("=== DML Comprehensive Experiments ===")
    
    runner = DMLExperimentRunner(verbose=True)
    
    # Run comprehensive experiments
    runner.run_experiments(
        datasets=['Vehicle', 'Bupa', 'Glass', 'Ionosphere', 'Iris', 'Wine', 'WDBC', 'Pima'],
        dr_methods=['PCA', 'LDA', 'MDS', 'Isomap', 'LLE', 'KernelPCA', 'Autoencoder'],
        classifiers=['knn', 'svm'],
        target_dims=[2, 3, 5]
    )
    
    print("\nComprehensive experiments completed!")


def main():
    """Main execution function."""
    print("Distance Metric Learning - Python Implementation")
    print("=" * 50)
    
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == '--demo':
        run_quick_demo()
    elif len(sys.argv) > 1 and sys.argv[1] == '--comprehensive':
        run_comprehensive_experiments()
    else:
        # Default: run demo
        run_quick_demo()


if __name__ == "__main__":
    main()
