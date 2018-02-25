import os

from util import load_csv


def load_loan_defaulters():
    """Load and return the loan defaulters dataset.

    Source: "Introduction to Data Mining" (1st Edition) by Pang-Ning Tan
    Figure 5.9, Page 230

    =================  ====================
    Classes                               2
    Samples per class                [7, 3]
    Samples total                        10
    Dimensionality                        3
    Features           binary, categorical,
                       continuous
    =================  ====================
    """
    return _load_dataset('loan-defaulters.csv')


def load_pima_indians():
    return _load_dataset('pima-indians.csv')


def _load_dataset(filename):
    return load_csv(_get_path(filename))


def _get_path(filename):
    path = os.path.join('datasets', 'data', filename)
    return os.path.abspath(path)
