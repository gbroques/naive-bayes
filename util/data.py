"""
Utility methods for munging on data.
"""


def separate_by_class(X, y):
    """Separate training dataset by class.

    Assumes the last element in each row is the class.

    :param X: Training vectors with dimension m x n,
              where m is the number of samples,
              and n is the number of features.
    :param y: Target values with dimension m,
              where m is the number of samples.
    :return: A dictionary where keys are classes,
             and values are a list of instances belonging to that class.
    """
    separated = {}
    for i in range(len(X)):
        vector = X[i]
        if y[i] not in separated:
            separated[y[i]] = []
        separated[y[i]].append(vector)
    return separated


def split_continuous_features(X, continuous_columns):
    """Splits a dataset with continuous and discrete values into two separate datasets.

    :param X: Training vectors with dimension m x n,
              where m is the number of samples,
              and n is the number of features.
    :param continuous_columns: A binary array with 1 denoting which columns are continuous,
                               and 0 denoting which columns are discrete.
    :return: A tuple containing the continuous dataset first, followed by the discrete dataset.
    """
    continuous_dataset, discrete_dataset = [], []
    for i in range(len(X)):
        continuous_dataset.append([])
        discrete_dataset.append([])
        for j in range(len(continuous_columns)):
            if continuous_columns[j]:
                continuous_dataset[i].append(X[i][j])
            else:
                discrete_dataset[i].append(X[i][j])

    return continuous_dataset, discrete_dataset


def remove_last_column(dataset):
    return [row[:-1] for row in dataset]
