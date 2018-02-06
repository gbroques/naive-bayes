#!/usr/bin/env python

"""
Naive bayes implementation in Python from scratch.

Python Version: 3.6.3

Adapted from Jason Brownlee at Machine Learning Mastery.
Source: https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/
"""

import csv
import random
import math

def main():
    # 1. Handle Data
    #     - Load the data from CSV file
    #     - Split data into training and test sets
    filename = 'pima-indians-diabetes.csv'
    dataset = load_csv(filename)
    print('Loaded data file {0} with {1} rows'.format(filename, len(dataset)))

    dataset = [[1], [2], [3], [4], [5]]
    split_ratio = 0.67
    train, test = split_dataset(dataset, split_ratio)
    print('Split {0} rows into train with {1} and test with {2}'.format(len(dataset), train, test))

    # 2. Summarize Data
    #     - Separate data by class

    dataset = [[1, 20, 1], [2, 21, 0], [3, 22, 1]]
    separated = separate_by_class(dataset)
    print('Separated instances: {0}'.format(separated))

    #     - Calculate mean
    #     - Calculate standard deviation
        
    numbers = [1,2,3,4,5]
    print('Summary of {0}: mean={1}, stdev={2}'.format(numbers, mean(numbers), std_dev(numbers)))
    #     - Summarize dataset
    #     - Summarize attributes by class


def load_csv(filename):
    """Load CSV data from a file and convert the attributes to numbers.

    :param filename: The name of the CSV file to load.
    :return: A list of the dataset.
    """

    f = open(filename)
    lines = csv.reader(f)
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    f.close()
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


def separate_by_class(dataset):
    """Separate training dataset by class.

    Assumes the last element in each row is the class.

    :param dataset: List of training data.
    :return: A dictionary where keys are classes,
             and values are a list of instances belonging to that class.
    """

    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if vector[-1] not in separated:
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated


def mean(numbers):
    """Calculate the mean of a list of numbers.

    :param numbers: A list of numbers.
    :return: The mean.
    """

    total = float(len(numbers))
    return sum(numbers) / total


def std_dev(numbers):
    """Calculate the standard deviation for a list of numbers.

    Use the N - 1 method, or Bessel's correction,
    to get best guess for overall population.

    :param numbers: A list of numbers.
    :return: The standard deviation.
    """

    avg = mean(numbers)
    distances_to_mean = [pow(x - avg, 2) for x in numbers]
    total = float(len(numbers))
    variance = sum(distances_to_mean) / (total - 1)
    return math.sqrt(variance)

if __name__ == '__main__':
    main()
