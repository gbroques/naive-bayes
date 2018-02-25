from abc import ABC, abstractmethod


class FeatureVectors(ABC):

    @abstractmethod
    def add(self, label, index, feature):
        """Add a feature to the container."""
