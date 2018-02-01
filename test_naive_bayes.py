import unittest
import naive_bayes

class TestNaiveBayes(unittest.TestCase):

    def test_load_csv_file(self):
        filename = 'pima-indians-diabetes.csv'
        dataset = naive_bayes.load_csv(filename)

        num_rows = len(dataset)
        self.assertEqual(num_rows, 768)

        last_row = [1, 93, 70, 31, 0, 30.4 , 0.315, 23, 0]
        self.assertEqual(dataset[num_rows - 1], last_row)

    def test_split_dataset(self):
        filename = 'pima-indians-diabetes.csv'
        dataset = naive_bayes.load_csv(filename)

        split_ratio = 0.67
        train, test = naive_bayes.split_dataset(dataset, split_ratio)

        num_rows = len(dataset)
        train_size = int(num_rows * split_ratio)
        test_size = num_rows - train_size
        self.assertEqual(len(train), train_size)
        self.assertEqual(len(test), test_size)

if __name__ == '__main__':
    unittest.main()