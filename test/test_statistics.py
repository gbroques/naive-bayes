import unittest

from statistics import gaussian_pdf
from statistics import mean
from statistics import variance


class TestStatistics(unittest.TestCase):
    def test_mean(self):
        expected_mean = 3.0
        numbers = [1, 2, 3, 4, 5]

        self.assertEqual(expected_mean, mean(numbers))

    def test_variance(self):
        expected_variance = 2.5
        numbers = [1, 2, 3, 4, 5]

        self.assertEqual(expected_variance, variance(numbers))

    def test_gaussian_pdf(self):
        expected_probability = 0.0624896575937
        x = 71.5
        mean = 73
        variance = 38.44

        probability = gaussian_pdf(x, mean, variance)

        self.assertAlmostEqual(expected_probability, probability)


if __name__ == '__main__':
    unittest.main()
