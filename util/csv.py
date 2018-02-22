"""
Utility methods for csv data.
"""

import csv


def load_csv(filename):
    """Load CSV data from a file and convert the attributes to numbers.

    :param filename: The name of the CSV file to load.
    :return: A list of the dataset.
    """
    f = open(filename)
    lines = csv.reader(f)
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    f.close()
    return dataset
