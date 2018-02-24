"""
Naive Bayes implementation.

API inspired by SciKit-learn.
Sources:
  https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/
  https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/naive_bayes.py
  https://github.com/ashkonf/HybridNaiveBayes
"""

import collections
from math import exp
from math import pi
from math import pow
from math import sqrt

from statistics import mean
from statistics import stdev
from util.data import separate_by_class
from util.data import split_continuous_features


class NaiveBayes:
    def __init__(self, continuous_columns):
        """Naive bayes constructor.
        :param continuous_columns: Binary list with same dimension as features.
                                   Specify which columns are continuous.
                                   1 denotes continuous columns, and 0 denotes discrete columns.
        """
        self._continuous_columns = continuous_columns

    def predict(self, X):
        """Perform classification on an array of test vectors X.

        :param X: Test vectors with dimension m x n,
                  where m is the number of samples,
                  and n is the number of features.
        :return: Predicted target values for X with dimension m,
                 where m is the number of samples.
        """
        return self.get_predictions(self.summaries, X)

    def fit(self, X, y):
        """Fit model according to training vectors X, and target values y.

        :param X: Training vectors with dimension m x n,
                  where m is the number of samples,
                  and n is the number of features.
        :param y: Target values with dimension m,
                  where m is the number of samples.
        :return: self
        """
        continuous_features, discrete_features = split_continuous_features(X, self._continuous_columns)
        label_counts = dict.fromkeys(tuple(set(y)), 0)
        self.feature_values = collections.defaultdict(
            lambda: collections.defaultdict(lambda: collections.defaultdict(lambda: [])))
        for i, feature in enumerate(discrete_features):
            label = y[i]
            label_counts[label] += 1
            for j, value in enumerate(feature):
                self.feature_values[label][j][value].append(value)

        self.priors = collections.Counter()
        for label in label_counts:
            # A label count can never be 0 because we only generate
            # a label count upon observing the first data point that
            # belongs to it. As a result, we don't worrying about
            # the argument to log being 0 here.
            self.priors[label] = label_counts[label]

        self.summaries = self.summarize_by_class(continuous_features, y)
        return self

    def predict_discrete(self, test_record):
        probabilities = dict.fromkeys(list(self.priors), 1.0)
        for label in self.priors:
            for i, value in enumerate(test_record):
                frequency = len(self.feature_values[label][i][value])
                probability = frequency / self.priors[label]
                probabilities[label] *= probability
        return max(probabilities, key=probabilities.get)

    def summarize_by_class(self, X, y):
        """Get the mean and standard deviation of each attribute for the instances of each class.

        :param X: Training vectors with dimension m x n,
                  where m is the number of samples,
                  and n is the number of features.
        :param y: Target values with dimension m,
                  where m is the number of samples.
        :return: A list of tuples with the mean and standard deviation for each attribute,
                 for all the instances of each class.
        """
        separated = separate_by_class(X, y)
        summaries = {}
        for class_, instances in separated.items():
            summaries[class_] = self.summarize(instances)
        return summaries

    def calculate_class_probabilities(self, summaries, input_vector):
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
                probabilities[class_value] *= self.gaussian_pdf(x, avg, std_dev)
        return probabilities

    def predict_with_summaries(self, summaries, input_vector):
        """Predict the class an input record belongs to.

        :param summaries: Mean and standard deviation for each attribute.
        :param input_vector: The input record to predict the class for.
        :return: The class the input record belongs to.
        """
        probabilities = self.calculate_class_probabilities(summaries, input_vector)
        best_label, best_prob = None, -1
        for class_value, probability in probabilities.items():
            if best_label is None or probability > best_prob:
                best_prob = probability
                best_label = class_value
        return best_label

    def get_predictions(self, summaries, test_set):
        """Get predictions for each instance in the test set.

        :param summaries: Mean and standard deviation for each attribute.
        :param test_set: Set of data used for testing the model.
        :return: Class predictions for each instance in the test set.
        """
        predictions = []
        for i in range(len(test_set)):
            result = self.predict_with_summaries(summaries, test_set[i])
            predictions.append(result)
        return predictions

    @staticmethod
    def summarize(X):
        """Summarize data set by calculating mean and standard deviation for each attribute.

        :param X: Training vectors with dimension m x n,
                  where m is the number of samples,
                  and n is the number of features.
        :return: A list of tuples representing the mean,
                 and standard deviation for each attribute.
        """
        summaries = [(mean(attr), stdev(attr)) for attr in zip(*X)]
        return summaries

    @staticmethod
    def gaussian_pdf(x, avg, std_dev):
        """Calculate gaussian probability density function.

        :param x: Attribute value.
        :param avg: Mean or average.
        :param std_dev: Standard deviation.
        :return: Likelihood that the attribute belongs to the class.
        """
        exponent = exp(-(pow(x - avg, 2) / (2 * pow(std_dev, 2))))
        return (1 / (sqrt(2 * pi) * std_dev)) * exponent
