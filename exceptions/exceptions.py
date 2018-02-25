"""Custom exceptions."""


class NotFittedError(ValueError):
    """Raise if predict is called before fit."""

    def __init__(self, class_name):
        message = self.message(class_name)
        super(NotFittedError, self).__init__(message)

    @staticmethod
    def message(class_name):
        return ("This instance of " + class_name +
                " has not been fitted yet. Please call "
                "'fit' before you call 'predict'.")


class ZeroObservationsError(ValueError):
    """Raise in place of ValueError when calculating natural log of zero."""

    def __init__(self, value, label):
        message = self.message(value, label)
        super(ZeroObservationsError, self).__init__(message)

    @staticmethod
    def message(value, label):
        return ("Value " + str(value) +
                " never occurs with class " + str(label) +
                ". Please set use_smoothing to True in the constructor.")
