"""
Lorentzian Models

This module contains the mathematical definitions of Lorentzian functions
and related utilities for parameter handling.
"""

import numpy as np
from typing import List, Tuple, Union
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


def single_lorentzian(x: np.ndarray, amplitude: float, center: float, 
                     width: float, baseline: float = 0.0) -> np.ndarray:
    """
    Single Lorentzian function.
    
    Parameters:
    -----------
    x : array-like
        Independent variable
    amplitude : float
        Peak amplitude (height above baseline)
    center : float
        Peak center position
    width : float
        Full width at half maximum (FWHM)
    baseline : float
        Constant baseline offset
        
    Returns:
    --------
    np.ndarray
        Lorentzian function values
    """
    gamma = width / 2.0  # Half-width at half-maximum
    return baseline + amplitude * (gamma**2) / ((x - center)**2 + gamma**2)


def multiple_lorentzian(x: np.ndarray, *params) -> np.ndarray:
    """
    Multiple Lorentzian components with shared baseline.
    
    Parameters are organized as:
    [amp1, center1, width1, amp2, center2, width2, ..., baseline]
    
    Parameters:
    -----------
    x : array-like
        Independent variable
    *params : float
        Flattened parameters for all components plus baseline
        
    Returns:
    --------
    np.ndarray
        Sum of all Lorentzian components plus baseline
    """
    if len(params) < 4:
        raise ValueError("Need at least 4 parameters (amp, center, width, baseline)")
    
    if (len(params) - 1) % 3 != 0:
        raise ValueError("Invalid parameter count. Expected format: [amp1, center1, width1, ..., baseline]")
    
    n_components = (len(params) - 1) // 3
    baseline = params[-1]
    
    result = np.full_like(x, baseline, dtype=float)
    
    for i in range(n_components):
        idx = i * 3
        amplitude = params[idx]
        center = params[idx + 1]
        width = params[idx + 2]
        result += single_lorentzian(x, amplitude, center, width, 0.0)
    
    return result


def validate_parameters(params: List[float], n_components: int) -> bool:
    """
    Validate parameter array for Lorentzian fitting.
    
    Parameters:
    -----------
    params : list
        Parameter array
    n_components : int
        Expected number of Lorentzian components
        
    Returns:
    --------
    bool
        True if parameters are valid
    """
    expected_length = n_components * 3 + 1  # 3 params per component + baseline
    
    if len(params) != expected_length:
        return False
    
    # Check for reasonable parameter values
    for i in range(n_components):
        idx = i * 3
        amplitude = params[idx]
        width = params[idx + 2]
        
        # Amplitude should be non-zero
        if abs(amplitude) < 1e-10:
            return False
            
        # Width should be positive
        if width <= 0:
            return False
    
    return True


def get_parameter_bounds(x: np.ndarray, y: np.ndarray, 
                        n_components: int) -> Tuple[List[float], List[float]]:
    """
    Generate reasonable parameter bounds for fitting.
    
    Parameters:
    -----------
    x : array-like
        Independent variable
    y : array-like
        Dependent variable
    n_components : int
        Number of Lorentzian components
        
    Returns:
    --------
    tuple
        (lower_bounds, upper_bounds) for parameters
    """
    x_range = np.ptp(x)
    y_range = np.ptp(y)
    y_min, y_max = np.min(y), np.max(y)
    
    lower_bounds = []
    upper_bounds = []
    
    for i in range(n_components):
        # Amplitude bounds
        lower_bounds.extend([-2 * y_range, np.min(x) - x_range, 0.001 * x_range])
        upper_bounds.extend([2 * y_range, np.max(x) + x_range, 2 * x_range])
    
    # Baseline bounds
    lower_bounds.append(y_min - y_range)
    upper_bounds.append(y_max + y_range)
    
    return lower_bounds, upper_bounds


def generate_initial_guess(x: np.ndarray, y: np.ndarray, 
                          n_components: int) -> List[float]:
    """
    Generate initial parameter guess for fitting.
    
    Parameters:
    -----------
    x : array-like
        Independent variable
    y : array-like
        Dependent variable
    n_components : int
        Number of Lorentzian components
        
    Returns:
    --------
    list
        Initial parameter guess
    """
    x_range = np.ptp(x)
    baseline_guess = np.median(y)
    
    # Subtract baseline estimate for peak finding
    y_corrected = y - baseline_guess
    
    params = []
    
    if n_components == 1:
        # Single component - use global maximum
        max_idx = np.argmax(np.abs(y_corrected))
        amplitude_guess = y_corrected[max_idx]
        center_guess = x[max_idx]
        width_guess = x_range / 10
        
        params.extend([amplitude_guess, center_guess, width_guess])
    
    else:
        # Multiple components - distribute across x range
        centers = np.linspace(np.min(x) + x_range/4, 
                             np.max(x) - x_range/4, 
                             n_components)
        
        for i, center in enumerate(centers):
            # Find nearest data point to estimate amplitude
            nearest_idx = np.argmin(np.abs(x - center))
            amplitude_guess = y_corrected[nearest_idx] / n_components
            width_guess = x_range / (n_components * 3)
            
            params.extend([amplitude_guess, center, width_guess])
    
    # Add baseline
    params.append(baseline_guess)
    
    return params


def fit_lorentzian(x: np.ndarray, y: np.ndarray, n_components: int = 1, 
                  initial_guess: List[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit Lorentzian function(s) to data using scipy.optimize.curve_fit.
    
    Parameters:
    -----------
    x : array-like
        Independent variable
    y : array-like
        Dependent variable
    n_components : int
        Number of Lorentzian components to fit
    initial_guess : list, optional
        Initial parameter guess
        
    Returns:
    --------
    tuple
        (fitted_parameters, parameter_covariance)
    """
    if initial_guess is None:
        initial_guess = generate_initial_guess(x, y, n_components)
    
    bounds = get_parameter_bounds(x, y, n_components)
    
    try:
        if n_components == 1:
            popt, pcov = curve_fit(
                lambda x, amp, center, width, baseline: single_lorentzian(x, amp, center, width, baseline),
                x, y, p0=initial_guess, bounds=bounds
            )
        else:
            popt, pcov = curve_fit(
                multiple_lorentzian,
                x, y, p0=initial_guess, bounds=bounds
            )
        
        return popt, pcov
    
    except Exception as e:
        raise RuntimeError(f"Fitting failed: {str(e)}")


def find_peaks_auto(x: np.ndarray, y: np.ndarray, min_height: float = None) -> List[int]:
    """
    Automatically find peaks in the data for initial parameter estimation.
    
    Parameters:
    -----------
    x : array-like
        Independent variable
    y : array-like
        Dependent variable
    min_height : float, optional
        Minimum peak height
        
    Returns:
    --------
    list
        Indices of detected peaks
    """
    if min_height is None:
        min_height = np.std(y) * 2
    
    peaks, _ = find_peaks(np.abs(y - np.median(y)), height=min_height)
    return peaks.tolist()
