"""Custom exceptions."""


class NotFittedError(ValueError):
    """Raise if predict is called before fit."""


class ZeroObservationsError(ValueError):
    """Raise when zero observations occur"""
