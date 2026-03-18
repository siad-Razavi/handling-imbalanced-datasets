# Handling Imbalanced Datasets

This project explores how class imbalance affects classification performance in machine learning and compares several common strategies for handling imbalanced datasets in Python.

## Overview

In many real-world classification tasks, one class is much less frequent than the other. This imbalance can make standard classifiers perform poorly on the minority class, even when overall accuracy appears high.  

In this notebook, I study the impact of imbalance using synthetic datasets and evaluate different approaches for improving classification results.

## Objectives

- Understand the effect of class imbalance on model performance
- Compare baseline classifiers with imbalance-handling techniques
- Evaluate models using more reliable metrics than accuracy alone

## Methods

The notebook includes:

- Synthetic dataset generation using `make_moons` and `make_classification`
- Experiments with different imbalance ratios
- Baseline classification models
- Random oversampling
- Random undersampling
- Cost-sensitive learning with class weights

## Models Used

- Support Vector Machine (SVM)
- Random Forest

## Evaluation Metrics

To better assess performance on imbalanced datasets, the following metrics are used:

- Accuracy
- Balanced Accuracy
- F1-score for the minority class

## Key Insight

The project shows that accuracy alone can be misleading for imbalanced classification problems. Metrics such as balanced accuracy and F1-score provide a more meaningful evaluation, especially for the minority class.

## Tools and Libraries

- Python
- NumPy
- Matplotlib
- scikit-learn

## Repository Contents

- `Imbalanced_datasets.ipynb` — main notebook containing all experiments, visualizations, and results

## Author

Said Abolhassan Razavi  
Master’s student in Artificial Intelligence at Université Paris-Saclay
