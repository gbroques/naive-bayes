import unittest

from naive_bayes import NaiveBayes

test_data = [[6, 148, 72, 35, 0, 33.6, 0.627, 50, 1],
             [1, 85, 66, 29, 0, 26.6, 0.351, 31, 0],
             [8, 183, 64, 0, 0, 23.3, 0.672, 32, 1]]


class TestNaiveBayes(unittest.TestCase):
    filename = 'test.csv'

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
        dataset = self.get_toy_binary_dataset()
        test_record = [0, 1, 0]
        clf = NaiveBayes()
        X = [row[:-1] for row in dataset]
        y = [row[-1] for row in dataset]
        clf.fit(X, y)
        prediction = clf.predict_record(test_record)
        expected_prediction = 1
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
