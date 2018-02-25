import unittest

from statistics import gaussian_pdf
from statistics import mean
from statistics import stdev


class TestStatistics(unittest.TestCase):
    def test_mean(self):
        numbers = [1, 2, 3, 4, 5]

        avg = mean(numbers)

        expected_mean = 3.0
        self.assertEqual(avg, expected_mean)

    def test_stdev(self):
        numbers = [1, 2, 3, 4, 5]

        standard_deviation = stdev(numbers)

        expected_stdev = 1.5811388300841898
        self.assertEqual(standard_deviation, expected_stdev)

    def test_gaussian_pdf(self):
        expected_probability = 0.0624896575937
        x = 71.5
        avg = 73
        std_deviation = 6.2

        probability = gaussian_pdf(x, avg, std_deviation)

        self.assertAlmostEqual(expected_probability, probability)


if __name__ == '__main__':
    unittest.main()
