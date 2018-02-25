from .feature import Feature


class DiscreteFeature(Feature):

    def __init__(self, value):
        super().__init__(value)
        self.frequency = 0
        self.num_categories = 0
        self.num_label_instances = 0

    def probability(self):
        frequency = self.frequency + 1
        return frequency / (self.num_label_instances + self.num_categories)

    def is_continuous(self):
        return False

    def is_discrete(self):
        return True
