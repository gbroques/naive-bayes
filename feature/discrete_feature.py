from .feature import Feature


class DiscreteFeature(Feature):

    def is_continuous(self):
        return False

    def is_discrete(self):
        return True
