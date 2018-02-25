#!/usr/bin/env python

"""
Naive bayes implementation in Python from scratch.

Python Version: 3.6.3

Naive Bayes implementation.
Maximizes the log likelihood to prevent underflow,
and applies Laplace smoothing to solve the zero observations problem.

API inspired by SciKit-learn.

Sources:
  * https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/
  * https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/naive_bayes.py
  * https://github.com/ashkonf/HybridNaiveBayes
"""

from datasets import load_loan_defaulters
from feature import ContinuousFeature
from feature import DiscreteFeature
from model_selection import get_accuracy
from naive_bayes import NaiveBayes


def main():
    dataset = load_loan_defaulters()
    design_matrix = [row[:-1] for row in dataset]
    target_values = [row[-1] for row in dataset]
    continuous_columns = (2,)
    clf = NaiveBayes(extract_features)
    clf.fit(design_matrix, target_values)
    predictions = clf.predict(design_matrix)
    print(predictions)
    accuracy = get_accuracy(dataset, predictions)
    print('Model Accuracy: {0}%'.format(accuracy))


def extract_features(example):
    return [
        DiscreteFeature(example[0]),
        DiscreteFeature(example[1]),
        ContinuousFeature(example[2])
    ]


if __name__ == '__main__':
    main()
