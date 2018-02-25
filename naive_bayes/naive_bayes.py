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

from exceptions import NotFittedError
from statistics import gaussian_pdf
from statistics import mean
from statistics import variance


class NaiveBayes:
    def __init__(self, extract_features):
        """Naive bayes constructor.
        :param extract_features: Callback to map a feature vector to discrete and continuous features.
        """
        self.priors = defaultdict(dict)
        self.label_counts = Counter()
        self.possible_categories = defaultdict(set)
        self.frequencies = defaultdict(lambda: defaultdict(lambda: Counter()))
        self.continuous_features = defaultdict(lambda: defaultdict(lambda: []))
        self.mean_variance = defaultdict(dict)
        self._extract_features = extract_features
        self._is_fitted = False

    def fit(self, design_matrix, target_values):
        """Fit model according to design matrix and target values.

        :param design_matrix: Training examples with dimension m x n,
                              where m is the number of examples,
                              and n is the number of features.
        :param target_values: Target values with dimension m,
                              where m is the number of examples.
        :return: self
        """
        for i, training_example in enumerate(design_matrix):
            label = target_values[i]
            self.label_counts[label] += 1
            features = self._extract_features(training_example)
            for j, feature in enumerate(features):
                if feature.is_continuous():
                    self.continuous_features[label][j].append(feature.value)
                else:
                    self.frequencies[label][j][feature.value] += 1
                    self.possible_categories[j].add(feature.value)

        total_num_records = len(target_values)
        for label in self.label_counts:
            self.priors[label] = self.label_counts[label] / total_num_records
            if self.continuous_features:
                for j, features in enumerate(self.continuous_features[label]):
                    self.mean_variance[label][j] = mean(features), variance(features)

        self._is_fitted = True
        return self

    def predict(self, test_set):
        """Predict target values for test set.

        :param test_set: Test set with dimension m x n,
                         where m is the number of examples,
                         and n is the number of features.
        :return: Predicted target values for test set with dimension m,
                 where m is the number of examples.
        """
        self.check_is_fitted()

        predictions = []
        for i in range(len(test_set)):
            result = self.predict_record(test_set[i])
            predictions.append(result)
        return predictions

    def predict_record(self, test_record):
        """Predict the label for the test record.

        Maximizes the log likelihood to prevent underflow.

        :param test_record: Test record to predict a label for.
        :return: The predicted label.
        """
        log_probabilities = {k: log(v) for k, v in self.priors.items()}
        for label in self.label_counts:
            features = self._extract_features(test_record)
            for i, feature in enumerate(features):
                probability = self._get_probability(i, feature, label)
                log_probabilities[label] += log(probability)
        return max(log_probabilities, key=log_probabilities.get)

    def check_is_fitted(self):
        if not self._is_fitted:
            raise NotFittedError("This instance of " +
                                 self.__class__.__name__ +
                                 " has not been fitted yet. Please call "
                                 "'fit' before you call 'predict'.")

    def _get_probability(self, feature_index, feature, label):
        if feature.is_continuous():
            mean_variance = self.mean_variance[label][feature_index]
            probability = gaussian_pdf(feature.value, *mean_variance)
        else:
            frequency = self.frequencies[label][feature_index][feature.value] + 1
            num_classes = len(self.possible_categories[feature_index])
            probability = frequency / (self.label_counts[label] + num_classes)
        return probability
