import unittest

from naive_bayes import NaiveBayes


class TestNaiveBayes(unittest.TestCase):

    def test_fit_and_predict(self):
        dataset = self.get_six_separable_points()
        design_matrix = [row[:-1] for row in dataset]
        target_values = [row[-1] for row in dataset]

        clf = NaiveBayes()
        clf.fit(design_matrix, target_values)
        predictions = clf.predict(design_matrix)

        self.assertEqual(target_values, predictions)

    def test_priors(self):
        expected_priors = {0: 0.5, 1: 0.5}
        dataset = self.get_six_separable_points()
        design_matrix = [row[:-1] for row in dataset]
        target_values = [row[-1] for row in dataset]

        clf = NaiveBayes()
        clf.fit(design_matrix, target_values)

        self.assertEqual(expected_priors, clf.priors)

    def test_frequencies(self):
        expected_frequencies = {
            0: {  # Class
                0: {  # Column Index
                    -2: 1,  # Occurrences of a particular value for the given class and column
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
        dataset = self.get_six_separable_points()
        design_matrix = [row[:-1] for row in dataset]
        target_values = [row[-1] for row in dataset]

        clf = NaiveBayes()
        clf.fit(design_matrix, target_values)
        self.assertEqual(expected_frequencies, clf.frequencies)

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
