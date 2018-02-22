"""
Utility methods for munging on data.
"""


def separate_by_class(dataset):
    """Separate training dataset by class.

    Assumes the last element in each row is the class.

    :param dataset: List of training data.
    :return: A dictionary where keys are classes,
             and values are a list of instances belonging to that class.
    """
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if vector[-1] not in separated:
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated


def split_mixed_dataset(dataset, continuous_columns):
    """Splits a dataset with continuous and discrete values into two separate datasets.

    :param dataset: A dataset with continuous and discrete values.
    :param continuous_columns: A binary array with 1 denoting which columns are continuous,
                               and 0 denoting which columns are discrete.
    :return: A tuple containing the continuous dataset first, followed by the discrete dataset.
    """
    continuous_dataset, discrete_dataset = [], []
    for i in range(len(dataset)):
        continuous_dataset.append([])
        discrete_dataset.append([])
        for j in range(len(continuous_columns)):
            if continuous_columns[j]:
                continuous_dataset[i].append(dataset[i][j])
            else:
                discrete_dataset[i].append(dataset[i][j])
        continuous_dataset[i].append(dataset[i][-1])
        discrete_dataset[i].append(dataset[i][-1])

    return continuous_dataset, discrete_dataset


def remove_last_column(dataset):
    return [row[:-1] for row in dataset]
