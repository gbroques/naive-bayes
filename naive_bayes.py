#!/usr/bin/env python

"""
Naive bayes implementation in Python from scratch.

Python Version: 3.6.3

Adapted from Jason Brownlee at Machine Learning Mastery.
Source: https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/
"""

import math

from model_selection import get_accuracy
from statistics import mean
from statistics import stdev
from util.csv import load_csv
from util.data import separate_by_class
from util.data import split_mixed_dataset


def main():
    filename = 'loan-defaulters.csv'
    dataset = load_csv(filename)
    continuous_columns = [0, 0, 1]
    continuous_dataset, discrete_dataset = split_mixed_dataset(dataset, continuous_columns)
    summaries = summarize_by_class(continuous_dataset)
    predictions = get_predictions(summaries, dataset)
    accuracy = get_accuracy(dataset, predictions)
    print('Model Accuracy: {0}%'.format(accuracy))


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
