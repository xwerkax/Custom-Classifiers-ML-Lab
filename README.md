## Lab 3 – Custom Classifiers: Prior-Based and k-Nearest Neighbors

This repository contains solutions to Lab 3 of a Machine Learning course. The goal of this lab is to implement two custom classifiers from scratch using Python and follow scikit-learn's API design.

## 🧠 Overview

We implemented two classifiers:
1. **Random Prior Classifier** – predicts labels by sampling from the prior distribution of class frequencies.
2. **k-Nearest Neighbors (k-NN)** – classifies based on the majority label among the `k` nearest neighbors, using brute-force distance calculation.

All classifiers inherit from `BaseEstimator` and `ClassifierMixin` from `sklearn.base`.

## ✅ Task 1: Prior-Based Random Classifier

- No parameters in the constructor (`__init__` with `pass`).
- During `fit`, stores class counts using `np.unique(..., return_counts=True)`.
- During `predict`, samples labels using `np.random.choice`, with probabilities derived from the training set class distribution.
- Tested using a synthetic dataset from `make_classification` with **500 samples** and **class weights 0.8 (class 0) and 0.2 (class 1)**.
- Evaluation:
  - `Balanced Accuracy Score` (sklearn): ~50%
  - `Standard Accuracy`: ~70% due to class imbalance

## ✅ Task 2: Custom k-Nearest Neighbors Classifier

- Accepts `k` as a parameter in the constructor.
- `fit` stores training data and labels.
- `predict` computes distances using `scipy.spatial.distance.cdist`, finds the nearest `k` using `np.argsort`, and predicts the majority label.
- Compared to `KNeighborsClassifier` from `sklearn` (with `algorithm='brute'`).
- For `k=5`, both implementations yield identical results.
