#!/usr/bin/env python

"""
Naive bayes implementation in Python from scratch.

Python Version: 3.6.3

Adapted from Jason Brownlee at Machine Learning Mastery.
Source: https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/
"""

import csv
import math

from model_selection import get_accuracy
from statistics import mean
from statistics import stdev


def main():
    filename = 'loan-defaulters.csv'
    dataset = load_csv(filename)
    continuous_columns = [0, 0, 1]
    continuous_dataset, discrete_dataset = split_mixed_dataset(dataset, continuous_columns)
    summaries = summarize_by_class(continuous_dataset)
    predictions = get_predictions(summaries, dataset)
    accuracy = get_accuracy(dataset, predictions)
    print('Model Accuracy: {0}%'.format(accuracy))


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


def summarize(dataset):
    """Summarize data set by calculating mean and standard deviation for each attribute.

    :param dataset: The dataset. Only supports lists.
    :return: A list of tuples representing the mean,
             and standard deviation for each attribute.
    """
    summaries = [(mean(attr), stdev(attr)) for attr in zip(*dataset)]
    del summaries[-1]
    return summaries


def summarize_by_class(dataset):
    """Get the mean and standard deviation of each attribute for the instances of each class.

    :param dataset: The dataset. Only supports lists.
    :return: A list of tuples with the mean and standard deviation for each attribute,
             for all the instances of each class.
    """
    separated = separate_by_class(dataset)
    summaries = {}
    for class_, instances in separated.items():
        summaries[class_] = summarize(instances)
    return summaries


def split_mixed_dataset(dataset, continuous_columns):
    """Splits a dataset with continuous and discrete values into two separate datasets.

    :param dataset: A dataset with continuous and discrete values.
    :param continuous_columns: A binary array with 1 denoting which columns are continuous,
                               and 0 denoting which columns are discrete.
    :return: A tuple containing the continuous dataset first, followed by the discrete dataset.
    """
    continuous_dataset, discrete_dataset = [], []
    for i in range(len(dataset)):
        continuous_dataset.append([])
        discrete_dataset.append([])
        for j in range(len(continuous_columns)):
            if continuous_columns[j]:
                continuous_dataset[i].append(dataset[i][j])
            else:
                discrete_dataset[i].append(dataset[i][j])
        continuous_dataset[i].append(dataset[i][-1])
        discrete_dataset[i].append(dataset[i][-1])

    return continuous_dataset, discrete_dataset


def remove_last_column(dataset):
    return [row[:-1] for row in dataset]


def calculate_probability(x, avg, std_dev):
    """Calculate gaussian probability density function.

    :param x: Attribute value.
    :param avg: Mean or average.
    :param std_dev: Standard deviation.
    :return: Likelihood that the attribute belongs to the class.
    """
    exponent = math.exp(-(math.pow(x - avg, 2) / (2 * math.pow(std_dev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * std_dev)) * exponent


def calculate_class_probabilities(summaries, input_vector):
    """Calculate probabilities of the record belonging to each class.

    :param summaries: A list of tuples containing the mean,
                      and standard deviation for each attribute.
    :param input_vector: Test record to classify.
    :return: Likelihood of the test record belonging to each class.
    """
    probabilities = {}
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = 1
        for i in range(len(class_summaries)):
            avg, std_dev = class_summaries[i]
            x = input_vector[i]
            probabilities[class_value] *= calculate_probability(x, avg, std_dev)
    return probabilities


def predict(summaries, input_vector):
    """Predict the class an input record belongs to.

    :param summaries: Mean and standard deviation for each attribute.
    :param input_vector: The input record to predict the class for.
    :return: The class the input record belongs to.
    """
    probabilities = calculate_class_probabilities(summaries, input_vector)
    best_label, best_prob = None, -1
    for class_value, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value
    return best_label


def get_predictions(summaries, test_set):
    """Get predictions for each instance in the test set.

    :param summaries: Mean and standard deviation for each attribute.
    :param test_set: Set of data used for testing the model.
    :return: Class predictions for each instance in the test set.
    """
    predictions = []
    for i in range(len(test_set)):
        result = predict(summaries, test_set[i])
        predictions.append(result)
    return predictions


if __name__ == '__main__':
    main()
