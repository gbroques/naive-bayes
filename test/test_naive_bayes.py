import unittest

from naive_bayes import NaiveBayes


class TestNaiveBayes(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        continuous_columns = (2,)
        cls.clf = NaiveBayes(continuous_columns)

    def test_gaussian_pdf(self):
        expected_probability = 0.0624896575937
        x = 71.5
        mean = 73
        stdev = 6.2

        probability = self.clf.gaussian_pdf(x, mean, stdev)

        self.assertAlmostEqual(expected_probability, probability)

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
