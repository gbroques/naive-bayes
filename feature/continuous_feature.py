from .feature import Feature


class ContinuousFeature(Feature):

    def is_continuous(self):
        return True

    def is_discrete(self):
        return False
