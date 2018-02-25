#!/usr/bin/env python

"""
Naive bayes implementation in Python from scratch.

Python Version: 3.6.3

Adapted from Jason Brownlee at Machine Learning Mastery.
Sources:
  https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/
  https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/naive_bayes.py
  https://github.com/ashkonf/HybridNaiveBayes
"""

from model_selection import get_accuracy
from naive_bayes import NaiveBayes
from datasets import load_loan_defaulters


def main():
    dataset = load_loan_defaulters()
    design_matrix = [row[:-1] for row in dataset]
    target_values = [row[-1] for row in dataset]
    continuous_columns = (2,)
    clf = NaiveBayes(continuous_columns)
    clf.fit(design_matrix, target_values)
    predictions = clf.predict(design_matrix)
    print(predictions)
    accuracy = get_accuracy(dataset, predictions)
    print('Model Accuracy: {0}%'.format(accuracy))


if __name__ == '__main__':
    main()
