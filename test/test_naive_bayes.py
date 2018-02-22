import unittest

from naive_bayes import NaiveBayes

test_data = [[6, 148, 72, 35, 0, 33.6, 0.627, 50, 1],
             [1, 85, 66, 29, 0, 26.6, 0.351, 31, 0],
             [8, 183, 64, 0, 0, 23.3, 0.672, 32, 1]]


class TestNaiveBayes(unittest.TestCase):
    filename = 'test.csv'

    @classmethod
    def setUpClass(cls):
        cls.clf = NaiveBayes()

    def test_summarize(self):
        dataset = [[1, 20, 1], [2, 21, 0], [3, 22, 1]]

        summaries = self.clf.summarize(dataset)

        expected_summaries = [(2.0, 1.0), (21.0, 1.0)]
        self.assertEqual(summaries, expected_summaries)

    def test_summarize_by_class(self):
        dataset = [[1, 20, 1],
                   [2, 21, 0],
                   [3, 22, 1],
                   [4, 22, 0]]
        summary = self.clf.summarize_by_class(dataset)

        expected_summary = {
            0: [(3.0, 1.4142135623730951),
                (21.5, 0.7071067811865476)],
            1: [(2.0, 1.4142135623730951),
                (21.0, 1.4142135623730951)]
        }
        self.assertEqual(summary, expected_summary)

    def test_gaussian_pdf(self):
        expected_probability = 0.0624896575937
        x = 71.5
        mean = 73
        stdev = 6.2

        probability = self.clf.gaussian_pdf(x, mean, stdev)

        self.assertAlmostEqual(expected_probability, probability)

    def test_calculate_class_probabilities(self):
        expected_class_zero_probability = 0.7820853879509118
        expected_class_one_probability = 6.298736258150442e-05
        summaries = {0: [(1, 0.5)], 1: [(20, 5.0)]}
        input_vector = [1.1, '?']
        probabilities = self.clf.calculate_class_probabilities(summaries, input_vector)
        self.assertAlmostEqual(expected_class_zero_probability, probabilities[0])
        self.assertAlmostEqual(expected_class_one_probability, probabilities[1])

    def test_predict(self):
        expected_prediction = 'A'

        summaries = {
            'A': [(1, 0.5)],
            'B': [(20, 5.0)]
        }
        input_vector = [1.1, '?']

        prediction = self.clf.predict_with_summaries(summaries, input_vector)

        self.assertEqual(expected_prediction, prediction)

    def test_get_predictions(self):
        expected_predictions = ['A', 'B']
        summaries = {
            'A': [(1, 0.5)],
            'B': [(20, 5.0)]
        }
        test_set = [[1.1, '?'], [19.1, '?']]

        predictions = self.clf.get_predictions(summaries, test_set)

        self.assertEqual(expected_predictions, predictions)


if __name__ == '__main__':
    unittest.main()
