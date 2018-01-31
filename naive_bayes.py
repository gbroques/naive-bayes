#!/usr/bin/env python

"""
Naive bayes implementation in Python from scratch.

Python Version: 3.6.3

Adapted from Jason Brownlee at Machine Learning Mastery.
Source: https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/
"""

import csv

def main():
    filename = 'pima-indians-diabetes.csv'
    dataset = load_csv(filename)
    print('Loaded data file {0} with {1} rows'.format(filename, len(dataset)))

def load_csv(filename):
    """Load csv data from a file and convert attributes to floats."""
    file = open(filename)
    lines = csv.reader(file)
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset

if __name__ == '__main__':
    main()