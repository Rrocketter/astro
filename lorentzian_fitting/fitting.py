"""
Lorentzian Fitting Implementation

This module contains the main fitting routines for single and multiple
Lorentzian components using scipy.optimize.curve_fit.
"""

import numpy as np
import warnings
from typing import Tuple, Dict, List, Optional, Union
from scipy.optimize import curve_fit

from .models import (single_lorentzian, multiple_lorentzian, 
                     validate_parameters, get_parameter_bounds, 
                     generate_initial_guess)
from .errors import FittingError, ParameterError, DataError, ConvergenceError
from .metrics import (calculate_aic, calculate_aicc, calculate_bic, 
                     calculate_parameter_correlations, calculate_goodness_of_fit,
                     calculate_confidence_intervals)


class LorentzianFitter:
    """
    Main class for fitting Lorentzian profiles to data.
    """
    
    def __init__(self, max_iterations: int = 1000, tolerance: float = 1e-8):
        """
        Initialize the fitter.
        
        Parameters:
        -----------
        max_iterations : int
            Maximum number of iterations for fitting
        tolerance : float
            Tolerance for convergence
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        
    def _validate_input_data(self, x: np.ndarray, y: np.ndarray, 
                           yerr: Optional[np.ndarray] = None) -> None:
        """Validate input data arrays."""
        if len(x) != len(y):
            raise DataError("x and y arrays must have the same length")
        
        if len(x) < 4:
            raise DataError("Need at least 4 data points for fitting")
        
        if yerr is not None and len(yerr) != len(x):
            raise DataError("yerr array must have the same length as x and y")
        
        if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
            raise DataError("Input data contains NaN or infinite values")
        
        if yerr is not None and (np.any(yerr <= 0) or np.any(~np.isfinite(yerr))):
            raise DataError("Error values must be positive and finite")
    
    def _calculate_uncertainties(self, pcov: np.ndarray, 
                               reduced_chi_squared: float) -> np.ndarray:
        """Calculate parameter uncertainties from covariance matrix."""
        if pcov is None:
            raise FittingError("Covariance matrix is None - fitting may have failed")
        
        # Scale covariance matrix by reduced chi-squared if > 1
        if reduced_chi_squared > 1:
            pcov_scaled = pcov * reduced_chi_squared
        else:
            pcov_scaled = pcov
        
        # Extract diagonal elements (variances) and take square root
        variances = np.diag(pcov_scaled)
        
        # Handle negative variances (shouldn't happen with good fit)
        if np.any(variances < 0):
            warnings.warn("Negative variances detected in covariance matrix")
            variances = np.abs(variances)
        
        return np.sqrt(variances)
    
    def _calculate_fit_statistics(self, x: np.ndarray, y: np.ndarray, 
                                y_fit: np.ndarray, yerr: Optional[np.ndarray],
                                n_params: int) -> Dict[str, float]:
        """Calculate comprehensive fit quality statistics."""
        residuals = y - y_fit
        n_points = len(x)
        dof = n_points - n_params
        
        if dof <= 0:
            raise FittingError("Insufficient degrees of freedom for fitting")
        
        if yerr is not None:
            chi_squared = np.sum((residuals / yerr) ** 2)
        else:
            # Use residual variance for unweighted chi-squared
            residual_var = np.var(residuals, ddof=dof)
            if residual_var > 0:
                chi_squared = np.sum(residuals ** 2) / residual_var
            else:
                chi_squared = 0.0
        
        reduced_chi_squared = chi_squared / dof
        
        # Calculate information criteria
        aic = calculate_aic(chi_squared, n_params, n_points)
        aicc = calculate_aicc(chi_squared, n_params, n_points)
        bic = calculate_bic(chi_squared, n_params, n_points)
        
        # Calculate additional goodness-of-fit metrics
        gof_metrics = calculate_goodness_of_fit(y, y_fit, yerr)
        
        # Combine all statistics
        fit_stats = {
            'chi_squared': chi_squared,
            'reduced_chi_squared': reduced_chi_squared,
            'degrees_of_freedom': dof,
            'aic': aic,
            'aicc': aicc,
            'bic': bic,
            'n_points': n_points,
            'n_params': n_params,
            **gof_metrics
        }
        
        return fit_stats
    
    def fit_single(self, x: np.ndarray, y: np.ndarray, 
                   yerr: Optional[np.ndarray] = None,
                   initial_guess: Optional[List[float]] = None,
                   confidence_level: float = 0.68) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Fit a single Lorentzian component.
        
        Parameters:
        -----------
        x : array-like
            Independent variable
        y : array-like
            Dependent variable
        yerr : array-like, optional
            Uncertainties in y
        initial_guess : list, optional
            Initial parameter guess [amplitude, center, width, baseline]
        confidence_level : float
            Confidence level for parameter intervals (default: 0.68 for 1σ)
            
        Returns:
        --------
        tuple
            (fitted_parameters, parameter_errors, fit_info)
        """
        self._validate_input_data(x, y, yerr)
        
        # Generate initial guess if not provided
        if initial_guess is None:
            initial_guess = generate_initial_guess(x, y, n_components=1)
        
        if len(initial_guess) != 4:
            raise ParameterError("Single Lorentzian requires 4 parameters")
        
        # Get parameter bounds
        lower_bounds, upper_bounds = get_parameter_bounds(x, y, n_components=1)
        bounds = (lower_bounds, upper_bounds)
        
        try:
            # Perform fitting
            if yerr is not None:
                sigma = yerr
                absolute_sigma = True
            else:
                sigma = None
                absolute_sigma = False
            
            popt, pcov = curve_fit(
                single_lorentzian, x, y,
                p0=initial_guess,
                sigma=sigma,
                absolute_sigma=absolute_sigma,
                bounds=bounds,
                maxfev=self.max_iterations,
                ftol=self.tolerance,
                xtol=self.tolerance
            )
            
            # Calculate fitted curve
            y_fit = single_lorentzian(x, *popt)
            
            # Calculate fit statistics
            fit_stats = self._calculate_fit_statistics(x, y, y_fit, yerr, n_params=4)
            
            # Calculate parameter uncertainties
            param_errors = self._calculate_uncertainties(pcov, fit_stats['reduced_chi_squared'])
            
            # Calculate confidence intervals
            conf_lower, conf_upper = calculate_confidence_intervals(
                popt, param_errors, confidence_level)
            
            # Calculate parameter correlations
            correlations = calculate_parameter_correlations(pcov)
            
            # Prepare fit info
            fit_info = {
                'fitted_curve': y_fit,
                'residuals': y - y_fit,
                'covariance_matrix': pcov,
                'correlation_matrix': correlations,
                'confidence_intervals': {
                    'lower': conf_lower,
                    'upper': conf_upper,
                    'level': confidence_level
                },
                **fit_stats
            }
            
            return popt, param_errors, fit_info
            
        except RuntimeError as e:
            if "Optimal parameters not found" in str(e):
                raise ConvergenceError(f"Fitting failed to converge: {e}")
            else:
                raise FittingError(f"Fitting error: {e}")
        except Exception as e:
            raise FittingError(f"Unexpected error during fitting: {e}")
    
    def fit_multiple(self, x: np.ndarray, y: np.ndarray, 
                     n_components: int,
                     yerr: Optional[np.ndarray] = None,
                     initial_guess: Optional[List[float]] = None,
                     confidence_level: float = 0.68) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Fit multiple Lorentzian components.
        
        Parameters:
        -----------
        x : array-like
            Independent variable
        y : array-like
            Dependent variable
        n_components : int
            Number of Lorentzian components
        yerr : array-like, optional
            Uncertainties in y
        initial_guess : list, optional
            Initial parameter guess
        confidence_level : float
            Confidence level for parameter intervals (default: 0.68 for 1σ)
            
        Returns:
        --------
        tuple
            (fitted_parameters, parameter_errors, fit_info)
        """
        if n_components < 1:
            raise ParameterError("Number of components must be at least 1")
        
        if n_components == 1:
            return self.fit_single(x, y, yerr, initial_guess)
        
        self._validate_input_data(x, y, yerr)
        
        expected_params = n_components * 3 + 1
        
        # Generate initial guess if not provided
        if initial_guess is None:
            initial_guess = generate_initial_guess(x, y, n_components)
        
        if len(initial_guess) != expected_params:
            raise ParameterError(f"Expected {expected_params} parameters for {n_components} components")
        
        # Get parameter bounds
        lower_bounds, upper_bounds = get_parameter_bounds(x, y, n_components)
        bounds = (lower_bounds, upper_bounds)
        
        try:
            # Perform fitting
            if yerr is not None:
                sigma = yerr
                absolute_sigma = True
            else:
                sigma = None
                absolute_sigma = False
            
            popt, pcov = curve_fit(
                multiple_lorentzian, x, y,
                p0=initial_guess,
                sigma=sigma,
                absolute_sigma=absolute_sigma,
                bounds=bounds,
                maxfev=self.max_iterations * n_components,  # Scale iterations with complexity
                ftol=self.tolerance,
                xtol=self.tolerance
            )
            
            # Calculate fitted curve
            y_fit = multiple_lorentzian(x, *popt)
            
            # Calculate fit statistics
            fit_stats = self._calculate_fit_statistics(x, y, y_fit, yerr, expected_params)
            
            # Calculate parameter uncertainties
            param_errors = self._calculate_uncertainties(pcov, fit_stats['reduced_chi_squared'])
            
            # Calculate confidence intervals
            conf_lower, conf_upper = calculate_confidence_intervals(
                popt, param_errors, confidence_level)
            
            # Calculate parameter correlations
            correlations = calculate_parameter_correlations(pcov)
            
            # Prepare fit info
            fit_info = {
                'fitted_curve': y_fit,
                'residuals': y - y_fit,
                'covariance_matrix': pcov,
                'correlation_matrix': correlations,
                'confidence_intervals': {
                    'lower': conf_lower,
                    'upper': conf_upper,
                    'level': confidence_level
                },
                'n_components': n_components,
                **fit_stats
            }
            
            return popt, param_errors, fit_info
            
        except RuntimeError as e:
            if "Optimal parameters not found" in str(e):
                raise ConvergenceError(f"Multi-component fitting failed to converge: {e}")
            else:
                raise FittingError(f"Multi-component fitting error: {e}")
        except Exception as e:
            raise FittingError(f"Unexpected error during multi-component fitting: {e}")
