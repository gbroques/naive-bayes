"""Custom exceptions."""


class NotFittedError(ValueError):
    """Raise if predict is called before fit."""


class ZeroObservationsError(ValueError):
    """Raise in place of ValueError when calculating natural log of zero."""
