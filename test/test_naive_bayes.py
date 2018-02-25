import unittest

from naive_bayes import NaiveBayes


class TestNaiveBayesWithSixSeparablePoints(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataset = cls.get_six_separable_points()
        cls.design_matrix = [row[:-1] for row in cls.dataset]
        cls.target_values = [row[-1] for row in cls.dataset]

        cls.clf = NaiveBayes()
        cls.clf.fit(cls.design_matrix, cls.target_values)

    @staticmethod
    def get_six_separable_points():
        """Six separable points in a plane.
              ^
              | x
              | x x
        <----------->
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
        self.assertEqual(expected_frequencies, self.clf.frequencies)

    def test_label_counts(self):
        expected_label_counts = {0: 3, 1: 3}

        self.assertEqual(expected_label_counts, self.clf.label_counts)

    def test_possible_categories(self):
        expected_possible_categories = {0: {-2, -1, 1, 2}, 1: {-2, -1, 1, 2}}

        self.assertEqual(expected_possible_categories, self.clf.possible_categories)


class TestNaiveBayesWithBinaryDataset(unittest.TestCase):
    def test_predict_record_with_binary_dataset(self):
        expected_prediction = 1
        dataset = self.get_toy_binary_dataset()
        design_matrix = [row[:-1] for row in dataset]
        target_values = [row[-1] for row in dataset]

        clf = NaiveBayes()
        clf.fit(design_matrix, target_values)
        test_record = [0, 1, 0]
        prediction = clf.predict_record(test_record)

        self.assertEqual(expected_prediction, prediction)

    @staticmethod
    def get_toy_binary_dataset():
        return [[0, 0, 0, 1],
                [0, 0, 1, 0],
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [0, 0, 1, 1],
                [1, 0, 1, 1],
                [1, 0, 1, 0],
                [1, 0, 1, 0],
                [1, 1, 1, 1],
                [1, 0, 1, 1]]


if __name__ == '__main__':
    unittest.main()
