import csv
import os
import unittest

from model_selection import get_accuracy
from model_selection import split_dataset
from util.csv import load_csv

test_data = [[6, 148, 72, 35, 0, 33.6, 0.627, 50, 1],
             [1, 85, 66, 29, 0, 26.6, 0.351, 31, 0],
             [8, 183, 64, 0, 0, 23.3, 0.672, 32, 1]]


class TestModelSelection(unittest.TestCase):
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

    def test_split_dataset(self):
        dataset = load_csv(self.filename)

        split_ratio = 0.67
        train, test = split_dataset(dataset, split_ratio)

        num_rows = len(dataset)
        train_size = int(num_rows * split_ratio)
        test_size = num_rows - train_size
        self.assertEqual(len(train), train_size)
        self.assertEqual(len(test), test_size)

    def test_get_accuracy(self):
        expected_accuracy = 66.6666667
        test_set = [
            [1, 1, 1, 'a'],
            [2, 2, 2, 'a'],
            [3, 3, 3, 'b']
        ]
        predictions = ['a', 'a', 'a']

        accuracy = get_accuracy(test_set, predictions)

        self.assertAlmostEqual(expected_accuracy, accuracy)


if __name__ == '__main__':
    unittest.main()
