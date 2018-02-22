import csv
import os
import unittest

from util.csv import load_csv
from util.data import remove_last_column
from util.data import separate_by_class
from util.data import split_continuous_features

test_data = [[6, 148, 72, 35, 0, 33.6, 0.627, 50, 1],
             [1, 85, 66, 29, 0, 26.6, 0.351, 31, 0],
             [8, 183, 64, 0, 0, 23.3, 0.672, 32, 1]]


class TestUtil(unittest.TestCase):
    filename = 'test.csv'

    @classmethod
    def setUpClass(cls):
        cls.create_csv(cls.filename)

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.filename)

    @classmethod
    def create_csv(cls, filename):
        with open(filename, 'w') as f:
            writer = csv.writer(f)
            for row in test_data:
                writer.writerow(row)

    def test_load_csv_file(self):
        dataset = load_csv(self.filename)

        self.assertEqual(len(dataset), len(test_data))

        self.assertEqual(dataset[-1], test_data[-1])

    def test_split_mixed_dataset(self):
        dataset = [[1, 0, 125000], [0, 1, 100000]]
        expected_continuous_dataset = [[125000], [100000]]
        expected_discrete_dataset = [[1, 0], [0, 1]]

        continuous_columns = [0, 0, 1]
        continuous_dataset, discrete_dataset = split_continuous_features(dataset, continuous_columns)

        self.assertEqual(expected_continuous_dataset, continuous_dataset)
        self.assertEqual(expected_discrete_dataset, discrete_dataset)

    def test_remove_last_column(self):
        dataset = [[1, 0, 125000], [0, 1, 100000]]
        expected_dataset = [[1, 0], [0, 1]]

        dataset_without_last_column = remove_last_column(dataset)

        self.assertEqual(expected_dataset, dataset_without_last_column)

    def test_separate_by_class(self):
        X = [[1, 20], [2, 21], [3, 22]]
        y = [1, 0, 1]

        separated = separate_by_class(X, y)

        expected_separation = {
            0: [[2, 21]],
            1: [[1, 20], [3, 22]]
        }
        self.assertEqual(separated, expected_separation)


if __name__ == '__main__':
    unittest.main()
