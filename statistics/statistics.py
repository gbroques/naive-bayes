from math import exp
from math import pi
from math import pow
from math import sqrt


def mean(numbers):
    """Calculate the mean of a list of numbers.

    :param numbers: A list of numbers.
    :return: The mean.
    """
    total = float(len(numbers))
    return sum(numbers) / total


def variance(numbers):
    """Calculate the variance for a list of numbers.

    Use the N - 1 method, or Bessel's correction,
    to get best guess for overall population.

    :param numbers: A list of numbers.
    :return: The variance or spread.
    """
    avg = mean(numbers)
    distances_to_mean = [pow(x - avg, 2) for x in numbers]
    total = float(len(numbers))
    return sum(distances_to_mean) / (total - 1)


def gaussian_pdf(x, avg, variance):
    """Calculate gaussian probability density function.

    :param x: Attribute value.
    :param avg: Mean or average.
    :param variance: Variance or standard deviation squared.
    :return: Likelihood that the attribute belongs to the class.
    """
    exponent = exp(-(pow(x - avg, 2) / (2 * variance)))
    return (1 / (sqrt(2 * pi) * sqrt(variance))) * exponent
