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
from util.csv import load_csv


def main():
    filename = 'loan-defaulters.csv'
    dataset = load_csv(filename)
    X = [row[:-1] for row in dataset]
    y = [row[-1] for row in dataset]
    continuous_columns = [0, 0, 1]
    clf = NaiveBayes(continuous_columns)
    clf.fit(X, y)
    predictions = clf.predict(X)
    accuracy = get_accuracy(dataset, predictions)
    print('Model Accuracy: {0}%'.format(accuracy))


if __name__ == '__main__':
    main()
