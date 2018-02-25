"""Custom exceptions."""


class NotFittedError(ValueError):
    """A custom exception to raise if predict is called before fit."""
