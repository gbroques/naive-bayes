from statistics import gaussian_pdf
from .feature import Feature


class ContinuousFeature(Feature):

    def __init__(self, value):
        super().__init__(value)
        self.mean = 0
        self.variance = 0

    def probability(self):
        return gaussian_pdf(self.value, self.mean, self.variance)

    def is_continuous(self):
        return True

    def is_discrete(self):
        return False
