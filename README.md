# Fraud Detection Neural Networks

A tutorial comparing three PyTorch neural network approaches for credit card fraud detection under extreme class imbalance (0.17% fraud rate).

## Overview

The dataset contains 284,807 European credit card transactions from September 2013, with features V1–V28 derived via PCA, plus `Amount` and a binary `Class` label. The core challenge is that standard loss functions collapse on heavily imbalanced data — this tutorial demonstrates why loss function design matters more than model capacity.

Three configurations are compared across four hidden layer sizes (16, 32, 64, 128 nodes):

| Config | Loss Function | Key Technique |
|--------|--------------|---------------|
| 1 | BCELoss | Unweighted — baseline |
| 2 | BCEWithLogitsLoss | `pos_weight=10` to penalise missed fraud |
| 3 | Focal Loss | Dynamically down-weights easy negatives (α=0.25, γ=2) |

## Key Findings

- **Config 1** collapses entirely at 16–32 hidden nodes, predicting every transaction as non-fraudulent (99.83% accuracy, 0% recall)
- **Config 2** never degenerates and achieves the highest recall (83%) with the fewest false negatives — best when missing fraud is costly
- **Config 3** is the most consistent across all architectures, delivering stable F1 (0.80–0.82) and high precision (85–87%)

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it at `data/credit_card_transactions.csv`.

Then open `coursework.ipynb` in Jupyter or run `coursework.py` directly.

## Stack

- PyTorch — model training (CUDA and Apple MPS supported)
- scikit-learn — preprocessing and evaluation metrics
- pandas / seaborn / matplotlib — data exploration and visualisation

## References

- ULB Machine Learning Group. [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). Kaggle, 2018.
- Lin et al. [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002). ICCV, 2017.
- Kingma & Ba. [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980). arXiv, 2014.
