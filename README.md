# Machine & Deep Learning

A hands-on implementation repository covering the full machine learning pipeline — from classical regression and classification to ensemble methods and deep learning with PyTorch.

## Overview

This repo follows a structured learning path through supervised learning techniques, with each topic organized in its own directory. Every notebook includes real-world datasets, hyperparameter tuning via `GridSearchCV`, and evaluation using standard metrics.

## Project Structure

```
machine_deep_learning/
├── linear_regression/      # Linear, polynomial, and regularized regression
├── logistic/               # Binary and multi-class logistic regression
├── knn/                    # K-Nearest Neighbors with pipeline and cross-validation
├── svm/                    # Support Vector Machines (classification and regression)
├── decision_trees/         # Decision trees and random forests
├── pytorch/                # Deep learning with PyTorch
├── matlib/                 # Matplotlib visualization basics
└── load_files/             # Utility for merging Excel files
```

## Topics Covered

### Linear Regression (`linear_regression/`)

| Notebook               | Description                                                                   |
| ---------------------- | ----------------------------------------------------------------------------- |
| `lr1.ipynb`            | Basic linear regression on economic data (interest rate vs. unemployment)     |
| `poly1.ipynb`          | Polynomial regression (degree 2) on advertising data                          |
| `poly2.ipynb`          | Comparing polynomial degrees 1–9; identifying overfitting via train/test RMSE |
| `reg.ipynb`            | Ridge, Lasso, and ElasticNet with varying alpha values                        |
| `regularization.ipynb` | ElasticNet with `StandardScaler`; compares all three regularization methods   |
| `grid_cv.ipynb`        | `GridSearchCV` on AMES housing dataset (274 features)                         |
| `grid_cv2.ipynb`       | GridSearch on preprocessed AMES data with 10% test split                      |
| `mlrl1.ipynb`          | California housing dataset; model persistence with `pickle` and `joblib`      |
| `np_pd.ipynb`          | NumPy and Pandas fundamentals (arrays, DataFrames, groupby)                   |
| `missing_values.ipynb` | Handling missing data in the AMES housing dataset                             |
| `hotel_booking.ipynb`  | Exploratory analysis of 119K hotel bookings                                   |

### Logistic Regression (`logistic/`)

| Notebook     | Description                                                                                |
| ------------ | ------------------------------------------------------------------------------------------ |
| `log1.ipynb` | Binary classification on hearing test data; confusion matrix, ROC, precision-recall curves |
| `log2.ipynb` | Multi-class classification on iris dataset; `GridSearchCV` over C and penalty              |

### K-Nearest Neighbors (`knn/`)

| Notebook    | Description                                                                        |
| ----------- | ---------------------------------------------------------------------------------- |
| `knn.ipynb` | Gene expression cancer classification; Pipeline + `GridSearchCV` to find optimal k |

### Support Vector Machines (`svm/`)

| Notebook                   | Description                                                               |
| -------------------------- | ------------------------------------------------------------------------- |
| `svm_classification.ipynb` | SVM with linear, RBF, sigmoid, and polynomial kernels on viral study data |
| `svm_regression.ipynb`     | SVR for cement compressive strength; GridSearch cuts MAE from 5.24 → 2.51 |
| `svm_margin_plot.py`       | Utility for visualizing SVM decision boundaries and support vectors       |

### Decision Trees & Ensemble Methods (`decision_trees/`)

| Notebook              | Description                                                                                     |
| --------------------- | ----------------------------------------------------------------------------------------------- |
| `dsc.ipynb`           | Decision tree classifier on Palmer penguins; feature importances and pruning                    |
| `random_forest.ipynb` | Random forest on penguins and banknote authentication; 240-combination GridSearch, 99% accuracy |

### Deep Learning — PyTorch (`pytorch/`)

| Notebook                 | Description                                                                                        |
| ------------------------ | -------------------------------------------------------------------------------------------------- |
| `torch_regression.ipynb` | Linear regression with `nn.Parameter` and `nn.Linear`; GPU training via CUDA; model saving/loading |

### Visualization (`matlib/`)

| Notebook     | Description                                                   |
| ------------ | ------------------------------------------------------------- |
| `mat1.ipynb` | Matplotlib basics: line plots, log scale, E=mc² visualization |

## Datasets Used

- **AMES Housing** — house sale prices with 274 engineered features
- **California Housing** — 20,640 samples predicting median house value
- **Advertising** — TV/radio/newspaper spend vs. sales
- **Palmer Penguins** — 3 penguin species across physical measurements
- **Iris** — classic 3-class flower classification
- **Gene Expression** — binary cancer classification (3,000 samples)
- **Banknote Authentication** — wavelet-transformed banknote features
- **Cement Slump** — 9 mix-design features predicting compressive strength
- **Hotel Bookings** — 119,390 booking records with 36 features

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Install dependencies
uv sync

# Launch Jupyter
uv run jupyter notebook
```

**Requirements:** Python >= 3.13

**Key dependencies:** `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `torch` (CUDA 12.8), `jupyter`

## Key Techniques

- **Preprocessing**: `StandardScaler`, `PolynomialFeatures`, `pd.get_dummies`, missing value imputation
- **Model selection**: `GridSearchCV`, `cross_val_score`, train/test splits
- **Pipelines**: `sklearn.pipeline.Pipeline` for reproducible workflows
- **Regularization**: Ridge (L2), Lasso (L1), ElasticNet
- **Ensemble methods**: Random forests, OOB scoring, feature importances
- **Deep learning**: Custom PyTorch training loop, GPU acceleration, `torch.save` / `load_state_dict`
- **Evaluation**: RMSE, MAE, accuracy, confusion matrix, classification report, ROC-AUC, precision-recall curves
