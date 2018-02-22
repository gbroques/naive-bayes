#!/usr/bin/env python

"""
Naive bayes implementation in Python from scratch.

Python Version: 3.6.3

Adapted from Jason Brownlee at Machine Learning Mastery.
Source: https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/
"""

from model_selection import get_accuracy
from naive_bayes import NaiveBayes
from util.csv import load_csv
from util.data import split_mixed_dataset


def main():
    filename = 'loan-defaulters.csv'
    dataset = load_csv(filename)
    continuous_columns = [0, 0, 1]
    continuous_dataset, discrete_dataset = split_mixed_dataset(dataset, continuous_columns)
    clf = NaiveBayes()
    summaries = clf.summarize_by_class(continuous_dataset)
    predictions = clf.get_predictions(summaries, dataset)
    accuracy = get_accuracy(dataset, predictions)
    print('Model Accuracy: {0}%'.format(accuracy))


if __name__ == '__main__':
    main()
