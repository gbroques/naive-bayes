import unittest

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


if __name__ == '__main__':
    unittest.main()
