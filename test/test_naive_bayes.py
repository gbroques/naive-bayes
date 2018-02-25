import unittest

from datasets import load_loan_defaulters
from exceptions import NotFittedError
from exceptions import ZeroObservationsError
from feature import ContinuousFeature
from feature import DiscreteFeature
from naive_bayes import NaiveBayes


class TestNaiveBayesWithSixSeparablePoints(unittest.TestCase):
    """Test the Naive Bayes classifier with a discrete toy dataset."""

    @classmethod
    def setUpClass(cls):
        cls.dataset = cls.get_six_separable_points()
        cls.design_matrix = [row[:-1] for row in cls.dataset]
        cls.target_values = [row[-1] for row in cls.dataset]

        cls.clf = NaiveBayes(cls.extract_features)
        cls.clf.fit(cls.design_matrix, cls.target_values)

    @staticmethod
    def get_six_separable_points():
        """Six separable points in a plane.
              ^
              | x
              | x x
        <-----+----->
          o o |
            o |
              v
        x - Denotes positive class "1"
        o - Denotes negative class "0"
        """
        return [[-2, -1, 0],
                [-1, -1, 0],
                [-1, -2, 0],
                [1, 1, 1],
                [1, 2, 1],
                [2, 1, 1]]

    @staticmethod
    def extract_features(example):
        return [DiscreteFeature(example[0]),
                DiscreteFeature(example[1])]

    def test_predict(self):
        predictions = self.clf.predict(self.design_matrix)

        self.assertEqual(self.target_values, predictions)

    def test_priors(self):
        expected_priors = {0: 0.5, 1: 0.5}

        self.assertEqual(expected_priors, self.clf.priors)

    def test_frequencies(self):
        expected_frequencies = {
            0: {  # Class
                0: {  # Column Index
                    -2: 1,  # Frequency of a particular value for the given class and column
                    -1: 2
                },
                1: {
                    -1: 2,
                    -2: 1
                }
            },
            1: {
                0: {
                    1: 2,
                    2: 1
                },
                1: {
                    1: 2,
                    2: 1
                }
            }
        }
        self.assertEqual(expected_frequencies, self.clf.discrete_feature_vectors.frequencies)

    def test_label_counts(self):
        expected_label_counts = {0: 3, 1: 3}

        self.assertEqual(expected_label_counts, self.clf.label_counts)

    def test_possible_categories(self):
        expected_possible_categories = {0: {-2, -1, 1, 2}, 1: {-2, -1, 1, 2}}

        self.assertEqual(expected_possible_categories, self.clf.discrete_feature_vectors.possible_categories)

    def test_continuous_features(self):
        self.assertFalse(self.clf.continuous_feature_vectors.continuous_features)

    def test_mean_variance(self):
        self.assertFalse(self.clf.continuous_feature_vectors.mean_variance)

    def test_raise_not_fitted_error_if_predict_is_called_before_predict(self):
        clf = NaiveBayes(lambda x: [DiscreteFeature(x[0])])
        with self.assertRaises(NotFittedError):
            clf.predict([0, 1])


class TestNaiveBayesWithBinaryDataset(unittest.TestCase):
    """Test the Naive Bayes classifier with a toy binary dataset."""

    @classmethod
    def setUpClass(cls):
        dataset = cls.get_toy_binary_dataset()
        cls.design_matrix = [row[:-1] for row in dataset]
        cls.target_values = [row[-1] for row in dataset]

    def test_predict_record_with_binary_dataset(self):
        expected_prediction = 1

        test_record = [1, 1, 0]
        clf = NaiveBayes(self.extract_features)
        clf.fit(self.design_matrix, self.target_values)
        prediction = clf.predict_record(test_record)

        self.assertEqual(expected_prediction, prediction)

    def test_zero_observations_error(self):
        clf = NaiveBayes(self.extract_features, use_smoothing=False)
        clf.fit(self.design_matrix, self.target_values)

        test_record = [0, 1, 0]
        with self.assertRaises(ZeroObservationsError):
            clf.predict_record(test_record)

    @staticmethod
    def extract_features(feature_vector):
        return [DiscreteFeature(feature_vector[0]),
                DiscreteFeature(feature_vector[1]),
                DiscreteFeature(feature_vector[2])]

    @staticmethod
    def get_toy_binary_dataset():
        """The third with value 0 is never observed with class 0.

        This is called the zero observations problem,
        and is dealt with by additive smoothing.
        """
        return [[0, 0, 1, 0],
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [1, 0, 1, 0],
                [1, 0, 1, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 1],
                [1, 0, 1, 1],
                [1, 1, 1, 1],
                [1, 0, 1, 1]]


class TestNaiveBayesWithContinuousData(unittest.TestCase):
    """Test the Naive Bayes classifier with continuous data."""

    @classmethod
    def setUpClass(cls):
        cls.dataset = load_loan_defaulters()
        cls.design_matrix = [row[:-1] for row in cls.dataset]
        cls.target_values = [row[-1] for row in cls.dataset]

        cls.clf = NaiveBayes(cls.extract_features)
        cls.clf.fit(cls.design_matrix, cls.target_values)

    def test_continuous_features(self):
        expected_continuous_features = {
            0.0: {  # Class Label
                # Continuous feature index and list of feature values belonging to class
                2: [125000.0, 100000.0, 70000.0, 120000.0, 60000.0, 220000.0, 75000.0]
            },
            1.0: {
                2: [95000.0, 85000.0, 90000.0]
            }
        }

        self.assertEqual(expected_continuous_features, self.clf.continuous_feature_vectors.continuous_features)

    def test_mean_variance(self):
        expected_mean_variance = {
            0.0: {2: (110000.0, 2975000000.0)},
            1.0: {2: (90000.0, 25000000.0)}
        }
        self.assertEqual(expected_mean_variance, self.clf.continuous_feature_vectors.mean_variance)

    @staticmethod
    def extract_features(feature_vector):
        return [
            DiscreteFeature(feature_vector[0]),
            DiscreteFeature(feature_vector[1]),
            ContinuousFeature(feature_vector[2])
        ]


if __name__ == '__main__':
    unittest.main()
