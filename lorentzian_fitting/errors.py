"""
Custom Exception Classes

This module defines custom exceptions for the Lorentzian fitting package.
"""


class LorentzianFittingError(Exception):
    """Base exception class for Lorentzian fitting errors."""
    
    def __init__(self, message: str, error_code: str = None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)


class FittingError(LorentzianFittingError):
    """Raised when curve fitting fails."""
    pass


class ParameterError(LorentzianFittingError):
    """Raised when parameters are invalid."""
    pass


class DataError(LorentzianFittingError):
    """Raised when input data is invalid."""
    pass


class ConvergenceError(FittingError):
    """Raised when fitting algorithm fails to converge."""
    pass
