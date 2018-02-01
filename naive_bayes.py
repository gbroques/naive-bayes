#!/usr/bin/env python

"""
Naive bayes implementation in Python from scratch.

Python Version: 3.6.3

Adapted from Jason Brownlee at Machine Learning Mastery.
Source: https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/
"""

import csv
import random

def main():
    filename = 'pima-indians-diabetes.csv'
    dataset = load_csv(filename)
    print('Loaded data file {0} with {1} rows'.format(filename, len(dataset)))

    dataset = [[1], [2], [3], [4], [5]]
    split_ratio = 0.67
    train, test = split_dataset(dataset, split_ratio)
    print('Split {0} rows into train with {1} and test with {2}'.format(len(dataset), train, test))

def load_csv(filename):
    """Load CSV data from a file and convert the attributes to numbers.

    :param filename: The name of the CSV file to load.
    :return: A list of the dataset.
    """
    file = open(filename)
    lines = csv.reader(file)
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    file.close()
    return dataset

def split_dataset(dataset, train_ratio):
    """Split the dataset into a training set and test set.

    Splits the data randomly into the desired ratio.

    :param dataset: The dataset to split. Only supports lists.
    :param train_ratio: float
        The proportion of the dataset to include in the training set.
    :return: A list containing the training set and test set.
    """
    train_size = int(len(dataset) * train_ratio)
    train_set = []
    copy = list(dataset)
    while len(train_set) < train_size:
        index = random.randrange(len(copy))
        train_set.append(copy.pop(index))
    return [train_set, copy]

if __name__ == '__main__':
    main()