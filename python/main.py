"""
Main execution script for Distance Metric Learning experiments.

This script replicates the functionality of the original MATLAB main.m file
by running DML experiments on multiple datasets with different dimensionality
reduction methods and classifiers.
"""

import warnings
from data_utils import DataLoader, CrossValidator, PerformanceEvaluator, preprocess_data
from distance_metric_learning import DMLClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

warnings.filterwarnings('ignore')


class DMLExperimentRunner:
    """Class to run comprehensive DML experiments."""

    def __init__(self, data_dir="../DataSets", results_dir="results", n_folds=10):
        """
        Initialize the experiment runner.

        Parameters:
        -----------
        data_dir : str
            Path to datasets directory
        results_dir : str
            Path to results directory
        n_folds : int
            Number of cross-validation folds
        """
        self.data_loader = DataLoader(data_dir)
        self.cv = CrossValidator(n_folds=n_folds)
        self.evaluator = PerformanceEvaluator()
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

        # Dataset names (matching original MATLAB)
        self.dataset_names = [
            'Vehicle', 'KDD', 'Bupa', 'Glass', 'Ionosphere', 'Monks',
            'New-thyroid', 'Pima', 'WDBC', 'Iris', 'Wine', 'Wholesale', 'CRC'
        ]

        # Dimensionality reduction methods
        self.dr_methods = [
            'PCA', 'LDA', 'MDS', 'Isomap', 'LLE'
            # Note: 'KernelPCA' and 'Autoencoder' can be added later
        ]

        # Classifier configurations
        self.classifier_configs = {
            'KNN': {
                'classifier_type': 'knn',
                'classifier_params': {'n_neighbors': 7}
            },
            'SVM': {
                'classifier_type': 'svm',
                'classifier_params': {'kernel': 'rbf', 'random_state': 42}
            }
        }

        # Results storage
        self.all_results = []

    def run_single_experiment(self, dataset_name, dr_method, classifier_name, target_dim):
        """
        Run a single experiment configuration.

        Parameters:
        -----------
        dataset_name : str
            Name of the dataset
        dr_method : str
            Dimensionality reduction method
        classifier_name : str
            Name of the classifier
        target_dim : int
            Target dimensionality

        Returns:
        --------
        dict
            Experiment results
        """
        print(
            f"Running: {dataset_name} | {dr_method} | {classifier_name} | dim={target_dim}")

        try:
            # Load dataset
            X, y = self.data_loader.load_dataset(dataset_name)

            # Preprocess data
            X, y, _ = preprocess_data(X, y)

            # Determine reduction percentage for large datasets
            reduction_perc = 0.01 if dataset_name == 'KDD' else 1.0

            # Create CV partitions
            cv_generator = self.cv.create_cv_partition(X, y, reduction_perc)

            # Get classifier configuration
            config = self.classifier_configs[classifier_name]

            # Define model parameters
            model_params = {
                'dml_params': {
                    'n_neighbors': 10,
                    'n_points_patch': 7,
                    'max_iterations': 30,
                    'lambda_reg': 1.0
                },
                'classifier_type': config['classifier_type'],
                'classifier_params': config['classifier_params']
            }

            # Run cross-validation
            results = self.evaluator.cross_validate_model(
                DMLClassifier, X, y, cv_generator,
                dr_method, target_dim, model_params
            )

            # Add experiment info to results
            results.update({
                'dataset': dataset_name,
                'dr_method': dr_method,
                'classifier': classifier_name,
                'target_dim': target_dim,
                'n_samples': len(X),
                'n_features': X.shape[1],
                'n_classes': len(np.unique(y))
            })

            return results

        except Exception as e:
            print(f"Error in experiment: {e}")
            return {
                'dataset': dataset_name,
                'dr_method': dr_method,
                'classifier': classifier_name,
                'target_dim': target_dim,
                'error': str(e)
            }

    def run_dimensionality_experiments(self, dataset_name, dr_method):
        """
        Run experiments across different dimensionalities for a dataset and DR method.

        Parameters:
        -----------
        dataset_name : str
            Name of the dataset
        dr_method : str
            Dimensionality reduction method

        Returns:
        --------
        list
            List of results for different dimensionalities
        """
        try:
            # Load dataset to determine dimensionality range
            X, y = self.data_loader.load_dataset(dataset_name)
            X, y, _ = preprocess_data(X, y)

            n_samples, n_features = X.shape
            D = min(n_features, n_samples)  # Maximum meaningful dimensionality

            # Create dimensionality range (matching MATLAB logic)
            interval = max(1, D // 5)
            dimensions = range(1, D + 1, interval)

            # Limit dimensions for computational efficiency
            max_dims = 20
            dimensions = [d for d in dimensions if d <= max_dims]

            results = []

            for target_dim in dimensions:
                for classifier_name in self.classifier_configs.keys():
                    result = self.run_single_experiment(
                        dataset_name, dr_method, classifier_name, target_dim
                    )
                    results.append(result)
                    self.all_results.append(result)

            return results

        except Exception as e:
            print(f"Error in dimensionality experiments: {e}")
            return []

    def run_all_experiments(self, datasets=None, dr_methods=None):
        """
        Run all experiments.

        Parameters:
        -----------
        datasets : list, optional
            List of datasets to run (default: all)
        dr_methods : list, optional
            List of DR methods to run (default: all)
        """
        datasets = datasets or self.dataset_names
        dr_methods = dr_methods or self.dr_methods

        print("Starting DML experiments...")
        print(f"Datasets: {datasets}")
        print(f"DR Methods: {dr_methods}")
        print(f"Classifiers: {list(self.classifier_configs.keys())}")

        for dataset_name in datasets:
            print(f"\n{'='*50}")
            print(f"Processing dataset: {dataset_name}")
            print(f"{'='*50}")

            for dr_method in dr_methods:
                print(f"\nDR Method: {dr_method}")
                self.run_dimensionality_experiments(dataset_name, dr_method)

        print("\nAll experiments completed!")

        # Save results
        self.save_results()

        # Generate plots
        self.generate_plots()

    def save_results(self):
        """Save results to files."""
        if not self.all_results:
            print("No results to save.")
            return

        # Convert to DataFrame
        df = pd.DataFrame(self.all_results)

        # Save comprehensive results
        results_file = self.results_dir / "all_results.csv"
        df.to_csv(results_file, index=False)
        print(f"Results saved to: {results_file}")

        # Save summary by dataset and method
        summary_metrics = ['avg_accuracy', 'avg_sensitivity',
                           'avg_specificity', 'avg_time_per_sample']

        for classifier in self.classifier_configs.keys():
            classifier_results = df[df['classifier'] == classifier]

            if not classifier_results.empty:
                # Create summary for each DR method
                for dr_method in self.dr_methods:
                    method_results = classifier_results[classifier_results['dr_method'] == dr_method]

                    if not method_results.empty:
                        # Save to Excel file (matching MATLAB format)
                        filename = self.results_dir / \
                            f"{classifier}_{dr_method}_results.xlsx"

                        # Group by dataset
                        for dataset in method_results['dataset'].unique():
                            dataset_results = method_results[method_results['dataset'] == dataset]

                            if not dataset_results.empty:
                                # Prepare data for Excel export
                                export_cols = ['target_dim'] + summary_metrics
                                export_data = dataset_results[export_cols].copy(
                                )

                                # Write to Excel
                                with pd.ExcelWriter(filename, mode='a' if filename.exists() else 'w') as writer:
                                    export_data.to_excel(
                                        writer, sheet_name=f"{dataset}_{dr_method}", index=False)

    def generate_plots(self):
        """Generate performance plots."""
        if not self.all_results:
            print("No results to plot.")
            return

        df = pd.DataFrame(self.all_results)

        # Filter out error results
        df = df[~df.get('error', pd.Series(dtype='object')).notna()]

        if df.empty:
            print("No valid results for plotting.")
            return

        # Create plots directory
        plots_dir = self.results_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        # Plot accuracy vs dimensionality for each dataset and DR method
        for dataset in df['dataset'].unique():
            dataset_df = df[df['dataset'] == dataset]

            for dr_method in dataset_df['dr_method'].unique():
                method_df = dataset_df[dataset_df['dr_method'] == dr_method]

                # Create subplot for different metrics
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                fig.suptitle(f'{dataset} - {dr_method}', fontsize=16)

                metrics = [
                    ('avg_accuracy', 'Accuracy'),
                    ('avg_sensitivity', 'Sensitivity'),
                    ('avg_specificity', 'Specificity'),
                    ('avg_time_per_sample', 'Time per Sample (s)')
                ]

                for idx, (metric, title) in enumerate(metrics):
                    ax = axes[idx // 2, idx % 2]

                    for classifier in method_df['classifier'].unique():
                        classifier_df = method_df[method_df['classifier']
                                                  == classifier]

                        if not classifier_df.empty and metric in classifier_df.columns:
                            ax.plot(classifier_df['target_dim'], classifier_df[metric],
                                    marker='o', label=classifier, linewidth=2)

                    ax.set_xlabel('Target Dimensionality')
                    ax.set_ylabel(title)
                    ax.set_title(title)
                    ax.legend()
                    ax.grid(True, alpha=0.3)

                plt.tight_layout()

                # Save plot
                plot_file = plots_dir / \
                    f"{dataset}_{dr_method}_performance.png"
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plt.close()

        print(f"Plots saved to: {plots_dir}")


def main():
    """Main execution function."""
    print("Distance Metric Learning - Python Implementation")
    print("=" * 50)

    # Initialize experiment runner
    runner = DMLExperimentRunner()

    # For demonstration, run on a subset of datasets
    # You can modify this to run on all datasets
    demo_datasets = ['Iris', 'Wine', 'WDBC']  # Start with smaller datasets
    demo_methods = ['PCA', 'LDA']  # Start with basic methods

    # Run experiments
    runner.run_all_experiments(datasets=demo_datasets, dr_methods=demo_methods)

    print("\nExperiment completed successfully!")
    print(f"Results saved in: {runner.results_dir}")


if __name__ == "__main__":
    main()
