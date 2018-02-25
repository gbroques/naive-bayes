"""
Naive Bayes implementation.
Maximizes the log likelihood to prevent underflow,
and applies Laplace smoothing to solve the zero observations problem.

API inspired by SciKit-learn.

Sources:
  * https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/
  * https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/naive_bayes.py
  * https://github.com/ashkonf/HybridNaiveBayes
"""

from collections import Counter
from collections import defaultdict
from math import log

from statistics import gaussian_pdf
from statistics import mean
from statistics import stdev


class NaiveBayes:
    def __init__(self, continuous_columns=()):
        """Naive bayes constructor.
        :param continuous_columns: Indices of which columns contain continuous values.
        """
        self.priors = defaultdict(dict)
        self.label_counts = Counter()
        self.possible_categories = defaultdict(set)
        self.frequencies = defaultdict(lambda: defaultdict(lambda: Counter()))
        self.continuous_features = defaultdict(lambda: defaultdict(lambda: []))
        self.gaussian_parameters = defaultdict(dict)
        self._continuous_columns = continuous_columns

    def predict(self, test_set):
        """Predict target values for test set.

        :param test_set: Test set with dimension m x n,
                         where m is the number of examples,
                         and n is the number of features.
        :return: Predicted target values for test set with dimension m,
                 where m is the number of examples.
        """
        predictions = []
        for i in range(len(test_set)):
            result = self.predict_record(test_set[i])
            predictions.append(result)
        return predictions

    def fit(self, design_matrix, target_values):
        """Fit model according to design matrix and target values.

        :param design_matrix: Training examples with dimension m x n,
                              where m is the number of examples,
                              and n is the number of features.
        :param target_values: Target values with dimension m,
                              where m is the number of examples.
        :return: self
        """
        for i, feature in enumerate(design_matrix):
            label = target_values[i]
            self.label_counts[label] += 1
            for j, value in enumerate(feature):
                if j in self._continuous_columns:
                    self.continuous_features[label][j].append(value)
                else:
                    self.frequencies[label][j][value] += 1
                    self.possible_categories[j].add(value)

        total_num_records = len(target_values)
        for label in self.label_counts:
            self.priors[label] = self.label_counts[label] / total_num_records
            for j in self._continuous_columns:
                features = self.continuous_features[label][j]
                avg = mean(features)
                std_dev = stdev(features)
                self.gaussian_parameters[label][j] = avg, std_dev
        return self

    def predict_record(self, test_record):
        log_probabilities = {k: log(v) for k, v in self.priors.items()}
        for label in self.label_counts:
            for i, value in enumerate(test_record):
                if i in self._continuous_columns:
                    gaussian_parameters = self.gaussian_parameters[label][i]
                    probability = gaussian_pdf(value, *gaussian_parameters)
                    log_probabilities[label] += log(probability)
                else:
                    frequency = self.frequencies[label][i][value] + 1
                    num_classes = len(self.possible_categories[i])
                    probability = frequency / (self.label_counts[label] + num_classes)
                    log_probabilities[label] += log(probability)
        return max(log_probabilities, key=log_probabilities.get)
