from abc import ABC, abstractmethod


class Feature(ABC):

    def __init__(self, value):
        self.value = value

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
