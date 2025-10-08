# Distance Metric Learning - Python Implementation

This is a Python reimplementation of the Distance Metric Learning algorithm based on structural neighborhoods for dimensionality reduction and classification performance improvement.

## Original Work

This implementation is based on the paper:
- **Title**: Distance metric learning based on structural neighborhoods for dimensionality reduction and classification performance improvement
- **Authors**: Mostafa Razavi Ghods, Mohammad Hossein Moattar, Yahya Forghani
- **arXiv**: https://arxiv.org/abs/1902.03453

## Overview

The algorithm performs the following steps:

1. **Manifold Learning**: Extract a low-dimensional manifold from high-dimensional input data using various dimensionality reduction techniques (PCA, LDA, MDS, Isomap, LLE).

2. **Neighborhood Structure Learning**: Learn local neighborhood structures and relationships of data points in the ambient space based on the adjacencies of the same data points on the embedded low-dimensional manifold.

3. **Distance Metric Learning**: Learn a distance metric that minimizes the distance between similar data points and maximizes their distance from dissimilar data points using the local neighborhood relationships extracted from the manifold space.

4. **Classification**: Apply the learned distance metric for improved classification performance using k-NN and SVM classifiers.

## Directory Structure

```
python/
├── src/
│   └── distance_metric_learning.py    # Main DML implementation
├── utils/
│   ├── data_utils.py                  # Data loading and preprocessing utilities
│   └── classifiers.py                 # Additional classifier implementations
├── main.py                            # Main execution script
├── requirements.txt                   # Python dependencies
└── README.md                         # This file
```

## Installation

1. **Clone the repository** (if not already done):
   ```bash
   git clone <repository-url>
   cd DML-based-on-structural-neighborhoods-for-dimensionality-reduction-and-classification-performance/python
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv .venv 
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

```python
from src.distance_metric_learning import DMLClassifier
from utils.data_utils import DataLoader, preprocess_data

# Load data
loader = DataLoader()
X, y = loader.load_dataset('Iris')
X, y, _ = preprocess_data(X, y)

# Create and train DML classifier
dml_clf = DMLClassifier(
    dml_params={'n_neighbors': 10, 'n_points_patch': 7},
    classifier_type='knn',
    classifier_params={'n_neighbors': 7}
)

# Fit the model
dml_clf.fit(X, y, dr_method='PCA', target_dim=2)

# Make predictions
y_pred = dml_clf.predict(X_test)
```

### Running Full Experiments

To replicate the original MATLAB experiments:

```bash
python main.py
```

This will:
- Run experiments on multiple datasets
- Test different dimensionality reduction methods
- Evaluate using k-NN and SVM classifiers
- Generate performance plots and save results

### Customizing Experiments

You can modify the main script to run specific experiments:

```python
from main import DMLExperimentRunner

# Initialize runner
runner = DMLExperimentRunner()

# Run on specific datasets and methods
runner.run_all_experiments(
    datasets=['Iris', 'Wine', 'WDBC'],
    dr_methods=['PCA', 'LDA']
)
```

## Algorithm Components

### 1. Distance Metric Learning (`DistanceMetricLearning` class)

The core algorithm that learns a linear transformation matrix `W` and translation vector `t` to map data into a space where similar points are closer and dissimilar points are farther apart.

**Key parameters:**
- `n_neighbors`: Number of nearest neighbors on the manifold (default: 10)
- `n_points_patch`: Number of neighbors to consider in patches (default: 7)
- `max_iterations`: Maximum optimization iterations (default: 30)
- `lambda_reg`: Regularization parameter (default: 1.0)

### 2. Data Loading (`DataLoader` class)

Handles loading of various datasets with automatic fallback to synthetic data generation if original datasets are not available.

**Supported datasets:**
- Vehicle, Bupa, Glass, Ionosphere, Monks
- New-thyroid, Pima, WDBC, Iris, Wine
- Wholesale, CRC, KDD

### 3. Dimensionality Reduction

Supports multiple DR methods:
- **PCA**: Principal Component Analysis
- **LDA**: Linear Discriminant Analysis
- **MDS**: Multi-Dimensional Scaling
- **Isomap**: Isometric Mapping
- **LLE**: Locally Linear Embedding

### 4. Classification

Implements multiple classifiers:
- **k-NN**: k-Nearest Neighbors
- **SVM**: Support Vector Machine
- **Similarity k-NN**: Custom similarity-based k-NN

## Results

The algorithm generates:

1. **Performance metrics**: Accuracy, Sensitivity, Specificity, F1-score
2. **Timing information**: Training time, prediction time per sample
3. **Confusion matrices**: For detailed classification analysis
4. **Plots**: Performance vs. dimensionality for different methods
5. **Excel reports**: Detailed results matching original MATLAB format

Results are saved in the `results/` directory with:
- `all_results.csv`: Comprehensive results
- `plots/`: Performance visualization plots
- Individual Excel files for each classifier and DR method

## Key Features

### Faithful Reproduction
- Maintains the same algorithmic structure as the original MATLAB code
- Uses identical parameter values and optimization procedures
- Generates comparable output formats and visualizations

### Enhanced Functionality
- Robust error handling and data validation
- Automatic synthetic data generation for missing datasets
- Comprehensive logging and progress tracking
- Modular design for easy extension and customization

### Performance Optimizations
- Efficient distance matrix calculations
- Memory-conscious data sampling for large datasets
- Parallel-friendly design for cross-validation

## Examples

### Example 1: Single Dataset Analysis

```python
from src.distance_metric_learning import DistanceMetricLearning
from utils.data_utils import DataLoader, preprocess_data
import numpy as np

# Load and preprocess data
loader = DataLoader()
X, y = loader.load_dataset('Iris')
X, y, _ = preprocess_data(X, y)

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Apply DML
dml = DistanceMetricLearning(n_neighbors=5, n_points_patch=3)
X_train_transformed = dml.fit_transform(X_train, y_train, dr_method='PCA', target_dim=2)
X_test_transformed = dml.transform(X_test)

print(f"Original shape: {X_train.shape}")
print(f"Transformed shape: {X_train_transformed.shape}")
```

### Example 2: Comparing DR Methods

```python
from main import DMLExperimentRunner
import matplotlib.pyplot as plt

runner = DMLExperimentRunner()

# Run comparison on Iris dataset
methods = ['PCA', 'LDA', 'MDS']
runner.run_all_experiments(datasets=['Iris'], dr_methods=methods)

# Results will be saved and plots generated automatically
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed via `pip install -r requirements.txt`

2. **Dataset Not Found**: The loader will automatically generate synthetic data if original datasets are missing

3. **Memory Issues**: For large datasets, the algorithm automatically applies data sampling

4. **Convergence Issues**: Try adjusting `lambda_reg` parameter or increasing `max_iterations`

### Performance Tips

- Use `reduction_perc < 1.0` for very large datasets
- Start with basic DR methods (PCA, LDA) before trying complex ones
- Monitor memory usage for high-dimensional data

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@misc{https://doi.org/10.48550/arxiv.1902.03453,
  doi = {10.48550/ARXIV.1902.03453},
  url = {https://arxiv.org/abs/1902.03453},
  author = {Ghods, Mostafa Razavi and Moattar, Mohammad Hossein and Forghani, Yahya},
  keywords = {Machine Learning (cs.LG), Machine Learning (stat.ML), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Distance metric learning based on structural neighborhoods for dimensionality reduction and classification performance improvement},
  publisher = {arXiv},
  year = {2019},
  copyright = {Creative Commons Attribution 4.0 International}
}
```

## License

This implementation follows the same license terms as the original work (Creative Commons Attribution 4.0 International).

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- Bug fixes
- Performance improvements
- Additional dimensionality reduction methods
- New dataset support
- Documentation improvements

## Contact

For questions about this Python implementation, please open an issue in the repository.

For questions about the original algorithm, please refer to the original paper and authors.