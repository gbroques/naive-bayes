import random


def split_dataset(dataset, train_ratio):
    """Split the dataset into a training set and test set.
    Splits the data randomly into the desired ratio.

    :param dataset: The dataset to split. Only supports lists.
    :param train_ratio: float
        The proportion of the dataset to include in the training set.
    :return: A list containing the training set and test set.
    """
    train_size = int(len(dataset) * train_ratio)
    train_set = []
    copy = list(dataset)
    while len(train_set) < train_size:
        index = random.randrange(len(copy))
        train_set.append(copy.pop(index))
    return [train_set, copy]


def get_accuracy(test_set, predictions):
    """Get accuracy of predictions on test set.

    :param test_set: The set of records to test the model with.
    :param predictions: Predictions for the test set.
    :return: The accuracy of the model.
    """
    num_correct = 0
    num_records = len(test_set)
    for x in range(num_records):
        if test_set[x][-1] == predictions[x]:
            num_correct += 1
    return (num_correct / float(num_records)) * 100.0
