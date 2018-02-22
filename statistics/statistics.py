import math


def mean(numbers):
    """Calculate the mean of a list of numbers.

    :param numbers: A list of numbers.
    :return: The mean.
    """
    total = float(len(numbers))
    return sum(numbers) / total


def stdev(numbers):
    """Calculate the standard deviation for a list of numbers.

    Use the N - 1 method, or Bessel's correction,
    to get best guess for overall population.

    :param numbers: A list of numbers.
    :return: The standard deviation.
    """
    avg = mean(numbers)
    distances_to_mean = [pow(x - avg, 2) for x in numbers]
    total = float(len(numbers))
    variance = sum(distances_to_mean) / (total - 1)
    return math.sqrt(variance)
