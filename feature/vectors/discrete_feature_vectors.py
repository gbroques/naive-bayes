from collections import Counter, defaultdict

from feature.vectors.feature_vectors import FeatureVectors


class DiscreteFeatureVectors(FeatureVectors):
    """Collection of discrete feature vectors."""

    def __init__(self, use_smoothing):
        """Construct a container for discrete feature vectors.

        :param use_smoothing: A boolean to indicate whether to use smoothing.
        """
        self.use_laplace_smoothing = use_smoothing
        self.possible_categories = defaultdict(set)
        self.frequencies = defaultdict(lambda: defaultdict(lambda: Counter()))

    def add(self, label, index, feature):
        self.frequencies[label][index][feature.value] += 1
        self.possible_categories[index].add(feature.value)

    def probability(self, label, index, feature, num_label_instances):
        """Calculate probability with Laplace smoothing optionally."""
        frequency = self.frequencies[label][index][feature.value]
        frequency += 1 if self.use_laplace_smoothing else 0

        num_classes = len(self.possible_categories[index])
        num_label_instances += num_classes if self.use_laplace_smoothing else 0
        return frequency / num_label_instances
