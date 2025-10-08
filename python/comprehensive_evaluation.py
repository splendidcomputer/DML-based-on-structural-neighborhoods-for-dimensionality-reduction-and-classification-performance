#!/usr/bin/env python3
"""
Comprehensive Performance Evaluation - MATLAB-Style Implementation

This script performs complete performance evaluation across all DR methods,
classifiers, and dimensions following the exact MATLAB MyDML.m structure.
"""

import sys
import os
import numpy as np
import pandas as pd
import time
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from distance_metric_learning import DMLClassifier
from data_utils import DataLoader

class ComprehensiveDMLEvaluator:
    """Complete performance evaluator following MATLAB MyDML structure."""
    
    def __init__(self, n_folds=10, verbose=True):
        self.n_folds = n_folds
        self.verbose = verbose
        self.data_loader = DataLoader()
        
        # All DR methods as specified
        self.dr_methods = ['PCA', 'LDA', 'MDS', 'Isomap', 'LLE', 'KernelPCA', 'Autoencoder']
        
        # Classifiers: knn, sim-knn (similarity-based kNN), svm
        self.classifiers = ['knn', 'sim-knn', 'svm']
        
    def compute_metrics(self, y_true, y_pred):
        """Compute sensitivity, specificity, accuracy following MATLAB classperf."""
        # Convert to binary for multi-class (macro averaging)
        classes = np.unique(np.concatenate([y_true, y_pred]))
        n_classes = len(classes)
        
        if n_classes == 2:
            # Binary classification
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=classes).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        else:
            # Multi-class classification (macro averaging)
            cm = confusion_matrix(y_true, y_pred, labels=classes)
            sensitivity_per_class = []
            specificity_per_class = []
            
            for i, class_label in enumerate(classes):
                tp = cm[i, i]
                fn = np.sum(cm[i, :]) - tp
                fp = np.sum(cm[:, i]) - tp
                tn = np.sum(cm) - tp - fn - fp
                
                sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                
                sensitivity_per_class.append(sens)
                specificity_per_class.append(spec)
            
            sensitivity = np.mean(sensitivity_per_class)
            specificity = np.mean(specificity_per_class)
        
        accuracy = accuracy_score(y_true, y_pred)
        
        return sensitivity, specificity, accuracy, confusion_matrix(y_true, y_pred, labels=classes)
    
    def run_single_experiment(self, X, y, dr_method, target_dim, classifier_type):
        """Run single experiment with k-fold cross validation following MATLAB structure."""
        
        # Initialize results arrays for this dimension (following MATLAB structure)
        knn_spec = np.zeros(self.n_folds)
        sim_knn_spec = np.zeros(self.n_folds)
        svm_spec = np.zeros(self.n_folds)
        
        knn_sen = np.zeros(self.n_folds)
        sim_knn_sen = np.zeros(self.n_folds)
        svm_sen = np.zeros(self.n_folds)
        
        knn_acc = np.zeros(self.n_folds)
        sim_knn_acc = np.zeros(self.n_folds)
        svm_acc = np.zeros(self.n_folds)
        
        knn_time = np.zeros(self.n_folds)
        sim_knn_time = np.zeros(self.n_folds)
        svm_time = np.zeros(self.n_folds)
        
        knn_conf = []
        sim_knn_conf = []
        svm_conf = []
        
        # Stratified K-Fold Cross Validation (following MATLAB MyCVPartition)
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            if self.verbose:
                print(f"    Fold {fold_idx + 1}/{self.n_folds}...", end="")
                
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # For each classifier type, run the experiment
            for clf_type in ['knn', 'sim-knn', 'svm']:
                try:
                    start_time = time.time()
                    
                    # Create DML classifier
                    dml_params = {
                        'verbose': False,
                        'lambda_reg': 1.0,
                        'max_iterations': 30,
                        'n_neighbors': 10,  # K = 10 as in MATLAB
                        'n_points_patch': 7  # nPointsPatch = 7 as in MATLAB
                    }
                    
                    # Map sim-knn to our similarity classifier
                    actual_clf_type = 'sim-knn' if clf_type == 'sim-knn' else clf_type
                    
                    dml_classifier = DMLClassifier(
                        dml_params=dml_params,
                        classifier_type=actual_clf_type
                    )
                    
                    # Fit and predict
                    dml_classifier.fit(X_train, y_train, dr_method=dr_method, target_dim=target_dim)
                    y_pred = dml_classifier.predict(X_test)
                    
                    elapsed_time = time.time() - start_time
                    
                    # Compute metrics
                    sensitivity, specificity, accuracy, conf_mat = self.compute_metrics(y_test, y_pred)
                    
                    # Store results in appropriate arrays
                    if clf_type == 'knn':
                        knn_sen[fold_idx] = sensitivity
                        knn_spec[fold_idx] = specificity
                        knn_acc[fold_idx] = accuracy
                        knn_time[fold_idx] = elapsed_time / len(y_test)  # Per sample time
                        knn_conf.append(conf_mat)
                    elif clf_type == 'sim-knn':
                        sim_knn_sen[fold_idx] = sensitivity
                        sim_knn_spec[fold_idx] = specificity
                        sim_knn_acc[fold_idx] = accuracy
                        sim_knn_time[fold_idx] = elapsed_time / len(y_test)
                        sim_knn_conf.append(conf_mat)
                    elif clf_type == 'svm':
                        svm_sen[fold_idx] = sensitivity
                        svm_spec[fold_idx] = specificity
                        svm_acc[fold_idx] = accuracy
                        svm_time[fold_idx] = elapsed_time / len(y_test)
                        svm_conf.append(conf_mat)
                        
                except Exception as e:
                    if self.verbose:
                        print(f" {clf_type} failed: {str(e)[:50]}...", end="")
                    # Set default values for failed experiments
                    if clf_type == 'knn':
                        knn_sen[fold_idx] = 0.0
                        knn_spec[fold_idx] = 0.0
                        knn_acc[fold_idx] = 0.0
                        knn_time[fold_idx] = 0.0
                    elif clf_type == 'sim-knn':
                        sim_knn_sen[fold_idx] = 0.0
                        sim_knn_spec[fold_idx] = 0.0
                        sim_knn_acc[fold_idx] = 0.0
                        sim_knn_time[fold_idx] = 0.0
                    elif clf_type == 'svm':
                        svm_sen[fold_idx] = 0.0
                        svm_spec[fold_idx] = 0.0
                        svm_acc[fold_idx] = 0.0
                        svm_time[fold_idx] = 0.0
            
            if self.verbose:
                print(" ✓")
        
        # Return results following MATLAB structure
        return {
            'KNN': {
                'Sen': knn_sen, 'Spec': knn_spec, 'Acc': knn_acc, 'Time': knn_time, 'Conf': knn_conf
            },
            'simKNN': {
                'Sen': sim_knn_sen, 'Spec': sim_knn_spec, 'Acc': sim_knn_acc, 'Time': sim_knn_time, 'Conf': sim_knn_conf
            },
            'SVM': {
                'Sen': svm_sen, 'Spec': svm_spec, 'Acc': svm_acc, 'Time': svm_time, 'Conf': svm_conf
            }
        }
    
    def run_comprehensive_evaluation(self, dataset_name, X, y):
        """Run comprehensive evaluation following exact MATLAB MyDML structure."""
        
        print(f"\n{'='*60}")
        print(f"COMPREHENSIVE EVALUATION: {dataset_name}")
        print(f"{'='*60}")
        
        n_samples, n_features = X.shape
        D = min(n_features, n_samples)  # Following MATLAB: if D > nSamples, D = nSamples
        
        # Calculate interval following MATLAB: interval = floor(D / 5) + 1
        interval = max(1, D // 5 + 1)
        
        if self.verbose:
            print(f"Dataset: {n_samples} samples, {n_features} features")
            print(f"Dimension range: 1 to {D} (interval: {interval})")
            print(f"DR Methods: {', '.join(self.dr_methods)}")
            print(f"Classifiers: KNN, sim-KNN, SVM")
            print(f"Cross-validation: {self.n_folds} folds")
        
        # Initialize result structure following MATLAB
        results = {}
        for dr_method in self.dr_methods:
            results[dr_method] = {
                'KNN': {'AvgSen': [], 'AvgSpec': [], 'AvgAcc': [], 'AvgTime': [], 'AvgCM': []},
                'simKNN': {'AvgSen': [], 'AvgSpec': [], 'AvgAcc': [], 'AvgTime': [], 'AvgCM': []},
                'SVM': {'AvgSen': [], 'AvgSpec': [], 'AvgAcc': [], 'AvgTime': [], 'AvgCM': []},
                'd': []  # Dimensions tested
            }
        
        # Main evaluation loop: for d = 1 : interval : D
        dimensions = range(1, D + 1, interval)
        total_experiments = len(self.dr_methods) * len(dimensions)
        experiment_count = 0
        
        for dr_method in self.dr_methods:
            print(f"\n--- DR Method: {dr_method} ---")
            
            for d in dimensions:
                experiment_count += 1
                target_dim = min(d, n_features - 1, n_samples - 1)
                
                if target_dim < 1:
                    target_dim = 1
                
                print(f"[{experiment_count}/{total_experiments}] {dr_method} - Dimension {d} (target: {target_dim})")
                
                try:
                    # Run experiment for this DR method and dimension
                    fold_results = self.run_single_experiment(X, y, dr_method, target_dim, 'all')
                    
                    # Aggregate results following MATLAB structure
                    results[dr_method]['KNN']['AvgSen'].append(np.mean(fold_results['KNN']['Sen']))
                    results[dr_method]['KNN']['AvgSpec'].append(np.mean(fold_results['KNN']['Spec']))
                    results[dr_method]['KNN']['AvgAcc'].append(np.mean(fold_results['KNN']['Acc']))
                    results[dr_method]['KNN']['AvgTime'].append(np.mean(fold_results['KNN']['Time']))
                    if fold_results['KNN']['Conf']:
                        results[dr_method]['KNN']['AvgCM'].append(np.mean(fold_results['KNN']['Conf'], axis=0))
                    
                    results[dr_method]['simKNN']['AvgSen'].append(np.mean(fold_results['simKNN']['Sen']))
                    results[dr_method]['simKNN']['AvgSpec'].append(np.mean(fold_results['simKNN']['Spec']))
                    results[dr_method]['simKNN']['AvgAcc'].append(np.mean(fold_results['simKNN']['Acc']))
                    results[dr_method]['simKNN']['AvgTime'].append(np.mean(fold_results['simKNN']['Time']))
                    if fold_results['simKNN']['Conf']:
                        results[dr_method]['simKNN']['AvgCM'].append(np.mean(fold_results['simKNN']['Conf'], axis=0))
                    
                    results[dr_method]['SVM']['AvgSen'].append(np.mean(fold_results['SVM']['Sen']))
                    results[dr_method]['SVM']['AvgSpec'].append(np.mean(fold_results['SVM']['Spec']))
                    results[dr_method]['SVM']['AvgAcc'].append(np.mean(fold_results['SVM']['Acc']))
                    results[dr_method]['SVM']['AvgTime'].append(np.mean(fold_results['SVM']['Time']))
                    if fold_results['SVM']['Conf']:
                        results[dr_method]['SVM']['AvgCM'].append(np.mean(fold_results['SVM']['Conf'], axis=0))
                    
                    results[dr_method]['d'].append(d)
                    
                    # Display results following MATLAB format
                    print(f"  KNN    - Acc: {results[dr_method]['KNN']['AvgAcc'][-1]:.4f}, "
                          f"Sen: {results[dr_method]['KNN']['AvgSen'][-1]:.4f}, "
                          f"Spec: {results[dr_method]['KNN']['AvgSpec'][-1]:.4f}, "
                          f"Time: {results[dr_method]['KNN']['AvgTime'][-1]:.6f}")
                    print(f"  simKNN - Acc: {results[dr_method]['simKNN']['AvgAcc'][-1]:.4f}, "
                          f"Sen: {results[dr_method]['simKNN']['AvgSen'][-1]:.4f}, "
                          f"Spec: {results[dr_method]['simKNN']['AvgSpec'][-1]:.4f}, "
                          f"Time: {results[dr_method]['simKNN']['AvgTime'][-1]:.6f}")
                    print(f"  SVM    - Acc: {results[dr_method]['SVM']['AvgAcc'][-1]:.4f}, "
                          f"Sen: {results[dr_method]['SVM']['AvgSen'][-1]:.4f}, "
                          f"Spec: {results[dr_method]['SVM']['AvgSpec'][-1]:.4f}, "
                          f"Time: {results[dr_method]['SVM']['AvgTime'][-1]:.6f}")
                    
                except Exception as e:
                    print(f"  ERROR: {str(e)[:100]}...")
                    # Add zero results for failed experiments
                    for clf in ['KNN', 'simKNN', 'SVM']:
                        results[dr_method][clf]['AvgSen'].append(0.0)
                        results[dr_method][clf]['AvgSpec'].append(0.0)
                        results[dr_method][clf]['AvgAcc'].append(0.0)
                        results[dr_method][clf]['AvgTime'].append(0.0)
                    results[dr_method]['d'].append(d)
        
        return results
    
    def save_results(self, results, dataset_name):
        """Save results in MATLAB-style format (CSV files)."""
        
        results_dir = os.path.join(os.path.dirname(__file__), 'results', 'comprehensive')
        os.makedirs(results_dir, exist_ok=True)
        
        # Column headers following MATLAB format
        col_headers = ['Dimension', 'Accuracy', 'Sensitivity', 'Specificity', 'Time']
        
        for dr_method in self.dr_methods:
            for classifier in ['KNN', 'simKNN', 'SVM']:
                
                # Create filename following MATLAB convention
                filename = f"{dataset_name}_{classifier}_{dr_method}.csv"
                filepath = os.path.join(results_dir, filename)
                
                # Prepare data
                if results[dr_method]['d']:  # Check if we have results
                    data = {
                        'Dimension': results[dr_method]['d'],
                        'Accuracy': results[dr_method][classifier]['AvgAcc'],
                        'Sensitivity': results[dr_method][classifier]['AvgSen'],
                        'Specificity': results[dr_method][classifier]['AvgSpec'],
                        'Time': results[dr_method][classifier]['AvgTime']
                    }
                    
                    df = pd.DataFrame(data)
                    df.to_csv(filepath, index=False)
                    
                    if self.verbose:
                        print(f"Saved: {filename}")
        
        # Save comprehensive summary
        summary_file = os.path.join(results_dir, f"{dataset_name}_COMPREHENSIVE_SUMMARY.csv")
        self.create_comprehensive_summary(results, dataset_name, summary_file)
    
    def create_comprehensive_summary(self, results, dataset_name, summary_file):
        """Create comprehensive summary following MATLAB format."""
        
        summary_data = []
        
        for dr_method in self.dr_methods:
            for classifier in ['KNN', 'simKNN', 'SVM']:
                if results[dr_method]['d']:  # Check if we have results
                    avg_acc = np.mean(results[dr_method][classifier]['AvgAcc'])
                    avg_sen = np.mean(results[dr_method][classifier]['AvgSen'])
                    avg_spec = np.mean(results[dr_method][classifier]['AvgSpec'])
                    avg_time = np.mean(results[dr_method][classifier]['AvgTime'])
                    
                    best_acc = np.max(results[dr_method][classifier]['AvgAcc'])
                    best_dim = results[dr_method]['d'][np.argmax(results[dr_method][classifier]['AvgAcc'])]
                    
                    summary_data.append({
                        'Dataset': dataset_name,
                        'DR_Method': dr_method,
                        'Classifier': classifier,
                        'Avg_Accuracy': avg_acc,
                        'Avg_Sensitivity': avg_sen,
                        'Avg_Specificity': avg_spec,
                        'Avg_Time': avg_time,
                        'Best_Accuracy': best_acc,
                        'Best_Dimension': best_dim,
                        'Dimensions_Tested': len(results[dr_method]['d'])
                    })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(summary_file, index=False)
        
        if self.verbose:
            print(f"\nComprehensive summary saved: {os.path.basename(summary_file)}")
    
    def create_matlab_style_plots(self, results, dataset_name):
        """Create MATLAB-style performance plots."""
        
        plots_dir = os.path.join(os.path.dirname(__file__), 'results', 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Create plots for each DR method
        for dr_method in self.dr_methods:
            if not results[dr_method]['d']:
                continue
                
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            dimensions = results[dr_method]['d']
            
            # Accuracy plot
            ax1.plot(dimensions, results[dr_method]['KNN']['AvgAcc'], '-s', linewidth=2, label='k-NN')
            ax1.plot(dimensions, results[dr_method]['simKNN']['AvgAcc'], '-o', linewidth=2, label='sim-k-NN')
            ax1.plot(dimensions, results[dr_method]['SVM']['AvgAcc'], '-^', linewidth=2, label='SVM')
            ax1.set_xlabel('Dimensionality of the embedded space')
            ax1.set_ylabel('Mean Accuracy')
            ax1.set_title(f'Mean Accuracy using {dr_method}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Sensitivity plot
            ax2.plot(dimensions, results[dr_method]['KNN']['AvgSen'], '-s', linewidth=2, label='k-NN')
            ax2.plot(dimensions, results[dr_method]['simKNN']['AvgSen'], '-o', linewidth=2, label='sim-k-NN')
            ax2.plot(dimensions, results[dr_method]['SVM']['AvgSen'], '-^', linewidth=2, label='SVM')
            ax2.set_xlabel('Dimensionality of the embedded space')
            ax2.set_ylabel('Mean Sensitivity')
            ax2.set_title(f'Mean Sensitivity using {dr_method}')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Specificity plot
            ax3.plot(dimensions, results[dr_method]['KNN']['AvgSpec'], '-s', linewidth=2, label='k-NN')
            ax3.plot(dimensions, results[dr_method]['simKNN']['AvgSpec'], '-o', linewidth=2, label='sim-k-NN')
            ax3.plot(dimensions, results[dr_method]['SVM']['AvgSpec'], '-^', linewidth=2, label='SVM')
            ax3.set_xlabel('Dimensionality of the embedded space')
            ax3.set_ylabel('Mean Specificity')
            ax3.set_title(f'Mean Specificity using {dr_method}')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Time plot
            ax4.plot(dimensions, results[dr_method]['KNN']['AvgTime'], '-s', linewidth=2, label='k-NN')
            ax4.plot(dimensions, results[dr_method]['simKNN']['AvgTime'], '-o', linewidth=2, label='sim-k-NN')
            ax4.plot(dimensions, results[dr_method]['SVM']['AvgTime'], '-^', linewidth=2, label='SVM')
            ax4.set_xlabel('Dimensionality of the embedded space')
            ax4.set_ylabel('Mean elapsed time')
            ax4.set_title(f'Mean elapsed time using {dr_method}')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot as SVG (vector graphics for academic format)
            plot_filename = f"Mean_Results_{dr_method}_Data_{dataset_name}.svg"
            plot_path = os.path.join(plots_dir, plot_filename)
            plt.savefig(plot_path, format='svg', bbox_inches='tight')
            plt.close()
            
            if self.verbose:
                print(f"Plot saved: {plot_filename}")


def main():
    """Main execution function."""
    
    print("COMPREHENSIVE DML PERFORMANCE EVALUATION")
    print("Following MATLAB MyDML.m Structure")
    print("="*60)
    
    # Initialize evaluator
    evaluator = ComprehensiveDMLEvaluator(n_folds=10, verbose=True)
    
    # Load datasets
    datasets = {
        'Iris': load_iris(),
        'Wine': load_wine(),
        'BreastCancer': load_breast_cancer()
    }
    
    # Add custom datasets if available
    try:
        data_loader = DataLoader()
        # Try to load Vehicle dataset
        vehicle_data = data_loader.load_dataset('Vehicle')
        if vehicle_data is not None:
            X_vehicle, y_vehicle = vehicle_data
            datasets['Vehicle'] = (X_vehicle, y_vehicle)
    except:
        pass
    
    all_results = {}
    
    # Run comprehensive evaluation for each dataset
    for dataset_name, dataset in datasets.items():
        if isinstance(dataset, tuple):
            X, y = dataset
        else:
            X, y = dataset.data, dataset.target
        
        # Ensure labels are properly encoded
        if y.dtype == 'object' or np.min(y) < 0:
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        print(f"\nProcessing {dataset_name}: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
        
        try:
            # Run comprehensive evaluation
            results = evaluator.run_comprehensive_evaluation(dataset_name, X, y)
            all_results[dataset_name] = results
            
            # Save results
            evaluator.save_results(results, dataset_name)
            
            # Create plots
            evaluator.create_matlab_style_plots(results, dataset_name)
            
        except Exception as e:
            print(f"ERROR processing {dataset_name}: {str(e)}")
            continue
    
    # Create final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    
    # Find best performing combinations
    best_results = []
    for dataset_name, results in all_results.items():
        for dr_method in evaluator.dr_methods:
            for classifier in ['KNN', 'simKNN', 'SVM']:
                if results[dr_method]['d']:
                    best_acc = np.max(results[dr_method][classifier]['AvgAcc'])
                    best_idx = np.argmax(results[dr_method][classifier]['AvgAcc'])
                    best_dim = results[dr_method]['d'][best_idx]
                    
                    best_results.append({
                        'Dataset': dataset_name,
                        'DR_Method': dr_method,
                        'Classifier': classifier,
                        'Best_Accuracy': best_acc,
                        'Best_Dimension': best_dim,
                        'Sensitivity': results[dr_method][classifier]['AvgSen'][best_idx],
                        'Specificity': results[dr_method][classifier]['AvgSpec'][best_idx],
                        'Time': results[dr_method][classifier]['AvgTime'][best_idx]
                    })
    
    # Save final summary
    final_summary_df = pd.DataFrame(best_results)
    final_summary_df = final_summary_df.sort_values('Best_Accuracy', ascending=False)
    
    summary_path = os.path.join(os.path.dirname(__file__), 'results', 'FINAL_COMPREHENSIVE_SUMMARY.csv')
    final_summary_df.to_csv(summary_path, index=False)
    
    # Display top results
    print("Top 10 Best Results:")
    print(final_summary_df.head(10).to_string(index=False))
    
    print(f"\nAll results saved to: {os.path.dirname(summary_path)}")
    print("Comprehensive evaluation completed!")


if __name__ == "__main__":
    main()