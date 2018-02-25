from abc import ABC, abstractmethod


class Feature(ABC):

    def __init__(self, value):
        self.value = value

    @abstractmethod
    def probability(self):
        """Return the probability of the feature belonging to a label.

        :return: The probability of the feature belonging to the label.
        """

    @abstractmethod
    def is_continuous(self):
        """Return whether the feature is continuous.
        :return: True if the feature is continuous.
        """

    @abstractmethod
    def is_discrete(self):
        """Return whether the feature is continuous.
        :return: True if the feature is continuous.
        """
