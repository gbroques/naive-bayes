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
from exceptions import ZeroObservationsError
from feature import ContinuousFeatureVectors
from feature import DiscreteFeatureVectors


class NaiveBayes:
    """Hybrid implementation of Naive Bayes.

    Supports discrete and continuous features.
    """

    def __init__(self, extract_features, use_smoothing=True):
        """Create a naive bayes classifier.

        :param extract_features: Callback to map a feature vector to discrete and continuous features.
        :param use_smoothing: Whether to use smoothing when calculating probability.
        """
        self.priors = defaultdict(dict)

        self.label_counts = Counter()
        self.discrete_feature_vectors = DiscreteFeatureVectors(use_smoothing)
        self.continuous_feature_vectors = ContinuousFeatureVectors()
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
                    self.continuous_feature_vectors.add(label, j, feature)
                else:
                    self.discrete_feature_vectors.add(label, j, feature)

        total_num_records = len(target_values)
        for label in set(target_values):
            self.priors[label] = self.label_counts[label] / total_num_records
            self.continuous_feature_vectors.set_mean_variance(label)

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
        self._check_is_fitted()

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
        self._check_is_fitted()

        log_likelihood = {k: log(v) for k, v in self.priors.items()}
        for label in self.label_counts:
            features = self._extract_features(test_record)
            for i, feature in enumerate(features):
                probability = self._get_probability(i, feature, label)
                try:
                    log_likelihood[label] += log(probability)
                except ValueError as e:
                    raise ZeroObservationsError(feature.value, label) from e
        return max(log_likelihood, key=log_likelihood.get)

    def _check_is_fitted(self):
        if not self._is_fitted:
            raise NotFittedError(self.__class__.__name__)

    def _get_probability(self, feature_index, feature, label):
        if feature.is_continuous():
            probability = self.continuous_feature_vectors.probability(label,
                                                                      feature_index)
        else:
            probability = self.discrete_feature_vectors.probability(label,
                                                                    feature_index,
                                                                    feature,
                                                                    self.label_counts[label])
        return probability
