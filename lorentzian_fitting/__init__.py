"""
Lorentzian Fitting Package

A comprehensive package for fitting Lorentzian profiles to astronomical data
with automated model selection and comparison.
"""

from .models import single_lorentzian, multiple_lorentzian
from .fitting import LorentzianFitter
from .comparison import ModelComparison, automated_model_selection
from .errors import FittingError, ParameterError, DataError, ConvergenceError

__version__ = "0.1.0"
__all__ = [
    'single_lorentzian', 'multiple_lorentzian',
    'LorentzianFitter', 'ModelComparison', 'automated_model_selection',
    'FittingError', 'ParameterError', 'DataError', 'ConvergenceError'
]
