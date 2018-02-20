#!/usr/bin/env python

"""
Naive bayes implementation in Python from scratch.

Python Version: 3.6.3

Adapted from Jason Brownlee at Machine Learning Mastery.
Source: https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/
"""

import csv
import math
import random


def main():
    # 1. Handle Data
    # --------------

    # Load the data from CSV file
    filename = 'pima-indians-diabetes.csv'
    dataset = load_csv(filename)
    print('Loaded data file {0} with {1} rows'.format(filename, len(dataset)))
    #     - Split data into training and test sets
    dataset = [[1], [2], [3], [4], [5]]
    split_ratio = 0.67
    train, test = split_dataset(dataset, split_ratio)
    print('Split {0} rows into train with {1} and test with {2}'.format(len(dataset), train, test))

    # 2. Summarize Data
    # -----------------

    # Separate data by class
    dataset = [[1, 20, 1], [2, 21, 0], [3, 22, 1]]
    separated = separate_by_class(dataset)
    print('Separated instances: {0}'.format(separated))

    # Calculate mean and standard deviation        
    numbers = [1, 2, 3, 4, 5]
    print('Summary of {0}: mean={1}, stdev={2}'.format(numbers, mean(numbers), stdev(numbers)))

    # Summarize dataset
    dataset = [[1, 20, 0], [2, 21, 1], [3, 22, 0]]
    summary = summarize(dataset)
    print('Attribute summaries: {0}'.format(summary))

    # Summarize attributes by class
    dataset = [[1, 20, 1],
               [2, 21, 0],
               [3, 22, 1],
               [4, 22, 0]]
    summary = summarize_by_class(dataset)
    print('Summary by class value: {0}'.format(summary))

    # 3. Make Prediction
    # ------------------
    # Calculate Gaussian Probability Density Function
    x = 71.5
    avg = 73
    std_dev = 6.2
    probability = calculate_probability(x, avg, std_dev)
    print('Probability of belonging to this class : {0}'.format(probability))

    #  Calculate Class Probabilities
    summaries = {0: [(1, 0.5)], 1: [(20, 5.0)]}
    input_vector = [1.1, '?']
    probabilities = calculate_class_probabilities(summaries, input_vector)
    print('Probabilities for each class: {0}'.format(probabilities))

    # Make a Prediction
    summaries = {
        'A': [(1, 0.5)],
        'B': [(20, 5.0)]
    }
    input_vector = [1.1, '?']
    result = predict(summaries, input_vector)
    print('Prediction: {0}'.format(result))

    # Make Predictions
    summaries = {
        'A': [(1, 0.5)],
        'B': [(20, 5.0)]
    }
    test_set = [[1.1, '?'], [19.1, '?']]
    predictions = get_predictions(summaries, test_set)
    print('Predictions: {0}'.format(predictions))

    # Estimate Accuracy
    test_set = [
        [1, 1, 1, 'a'],
        [2, 2, 2, 'a'],
        [3, 3, 3, 'b']
    ]
    predictions = ['a', 'a', 'a']
    accuracy = get_accuracy(test_set, predictions)
    print('Accuracy: {0}'.format(accuracy))


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


def stdev(numbers):
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


def get_accuracy(test_set, predictions):
    """Get accuracy of predictions on test set.

    :param test_set: The set of records to test the model with.
    :param predictions: Predictions for the test set.
    :return: The accuracy of the model.
    """
    num_correct = 0
    num_records = len(test_set)
    for x in range(num_records):
        if test_set[x][-1] == predictions[x]:
            num_correct += 1
    return (num_correct / float(num_records)) * 100.0


if __name__ == '__main__':
    main()
