import csv
import os
import unittest

from util import load_csv

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


if __name__ == '__main__':
    unittest.main()
