"""
Fit Quality Metrics

This module contains functions for calculating various fit quality metrics
including information criteria (AIC, BIC) and uncertainty analysis.
"""

import numpy as np
from typing import Dict, Optional, Tuple
import warnings


def calculate_aic(chi_squared: float, n_params: int, n_points: int) -> float:
    """
    Calculate Akaike Information Criterion (AIC).
    
    AIC = 2k - 2ln(L) where k is number of parameters and L is likelihood
    For Gaussian errors: AIC ≈ χ² + 2k
    
    Parameters:
    -----------
    chi_squared : float
        Chi-squared statistic
    n_params : int
        Number of fitted parameters
    n_points : int
        Number of data points
        
    Returns:
    --------
    float
        AIC value
    """
    return chi_squared + 2 * n_params


def calculate_aicc(chi_squared: float, n_params: int, n_points: int) -> float:
    """
    Calculate corrected Akaike Information Criterion (AICc).
    
    AICc = AIC + 2k(k+1)/(n-k-1) for small sample correction
    
    Parameters:
    -----------
    chi_squared : float
        Chi-squared statistic
    n_params : int
        Number of fitted parameters
    n_points : int
        Number of data points
        
    Returns:
    --------
    float
        AICc value
    """
    aic = calculate_aic(chi_squared, n_params, n_points)
    
    if n_points - n_params - 1 <= 0:
        warnings.warn("Cannot calculate AICc: insufficient degrees of freedom")
        return np.inf
    
    correction = (2 * n_params * (n_params + 1)) / (n_points - n_params - 1)
    return aic + correction


def calculate_bic(chi_squared: float, n_params: int, n_points: int) -> float:
    """
    Calculate Bayesian Information Criterion (BIC).
    
    BIC = ln(n)k - 2ln(L) where n is sample size, k is parameters, L is likelihood
    For Gaussian errors: BIC ≈ χ² + k*ln(n)
    
    Parameters:
    -----------
    chi_squared : float
        Chi-squared statistic
    n_params : int
        Number of fitted parameters
    n_points : int
        Number of data points
        
    Returns:
    --------
    float
        BIC value
    """
    return chi_squared + n_params * np.log(n_points)


def calculate_delta_ic(ic_values: Dict[str, float]) -> Dict[str, float]:
    """
    Calculate delta information criteria relative to the best model.
    
    Parameters:
    -----------
    ic_values : dict
        Dictionary with model names as keys and IC values as values
        
    Returns:
    --------
    dict
        Dictionary with delta IC values
    """
    if not ic_values:
        return {}
    
    min_ic = min(ic_values.values())
    return {model: ic - min_ic for model, ic in ic_values.items()}


def calculate_akaike_weights(aic_values: Dict[str, float]) -> Dict[str, float]:
    """
    Calculate Akaike weights for model comparison.
    
    Weight_i = exp(-0.5 * Δ_i) / Σ exp(-0.5 * Δ_j)
    where Δ_i is the AIC difference for model i
    
    Parameters:
    -----------
    aic_values : dict
        Dictionary with model names as keys and AIC values as values
        
    Returns:
    --------
    dict
        Dictionary with Akaike weights
    """
    if not aic_values:
        return {}
    
    delta_aic = calculate_delta_ic(aic_values)
    
    # Calculate relative likelihoods
    rel_likelihoods = {model: np.exp(-0.5 * delta) 
                      for model, delta in delta_aic.items()}
    
    # Normalize to get weights
    total_likelihood = sum(rel_likelihoods.values())
    
    if total_likelihood == 0:
        warnings.warn("All relative likelihoods are zero")
        return {model: 0.0 for model in aic_values.keys()}
    
    return {model: likelihood / total_likelihood 
            for model, likelihood in rel_likelihoods.items()}


def bootstrap_uncertainties(x: np.ndarray, y: np.ndarray, yerr: np.ndarray,
                          fit_function, n_bootstrap: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate parameter uncertainties using bootstrap resampling.
    
    Parameters:
    -----------
    x : array-like
        Independent variable
    y : array-like
        Dependent variable
    yerr : array-like
        Uncertainties in y
    fit_function : callable
        Function that performs the fit and returns (params, param_errors, fit_info)
    n_bootstrap : int
        Number of bootstrap samples
        
    Returns:
    --------
    tuple
        (parameter_means, parameter_std_devs)
    """
    n_points = len(x)
    bootstrap_params = []
    
    for i in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_points, size=n_points, replace=True)
        x_boot = x[indices]
        y_boot = y[indices]
        yerr_boot = yerr[indices]
        
        try:
            params, _, _ = fit_function(x_boot, y_boot, yerr_boot)
            bootstrap_params.append(params)
        except Exception:
            # Skip failed fits
            continue
    
    if not bootstrap_params:
        raise RuntimeError("All bootstrap fits failed")
    
    bootstrap_params = np.array(bootstrap_params)
    param_means = np.mean(bootstrap_params, axis=0)
    param_stds = np.std(bootstrap_params, axis=0, ddof=1)
    
    return param_means, param_stds


def calculate_confidence_intervals(params: np.ndarray, param_errors: np.ndarray,
                                 confidence_level: float = 0.68) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate confidence intervals for fitted parameters.
    
    Parameters:
    -----------
    params : array-like
        Fitted parameter values
    param_errors : array-like
        Parameter uncertainties (standard errors)
    confidence_level : float
        Confidence level (0.68 for 1σ, 0.95 for 2σ, etc.)
        
    Returns:
    --------
    tuple
        (lower_bounds, upper_bounds)
    """
    from scipy import stats
    
    # Calculate t-statistic for given confidence level
    alpha = 1 - confidence_level
    t_value = stats.norm.ppf(1 - alpha/2)
    
    margin = t_value * param_errors
    lower_bounds = params - margin
    upper_bounds = params + margin
    
    return lower_bounds, upper_bounds


def calculate_parameter_correlations(covariance_matrix: np.ndarray) -> np.ndarray:
    """
    Calculate parameter correlation matrix from covariance matrix.
    
    Parameters:
    -----------
    covariance_matrix : array-like
        Parameter covariance matrix
        
    Returns:
    --------
    np.ndarray
        Correlation matrix
    """
    if covariance_matrix is None:
        raise ValueError("Covariance matrix is None")
    
    # Extract standard deviations (diagonal elements)
    std_devs = np.sqrt(np.diag(covariance_matrix))
    
    # Calculate correlation matrix
    correlation_matrix = np.zeros_like(covariance_matrix)
    
    for i in range(len(std_devs)):
        for j in range(len(std_devs)):
            if std_devs[i] > 0 and std_devs[j] > 0:
                correlation_matrix[i, j] = (covariance_matrix[i, j] / 
                                          (std_devs[i] * std_devs[j]))
            else:
                correlation_matrix[i, j] = 0.0
    
    return correlation_matrix


def calculate_goodness_of_fit(y_data: np.ndarray, y_fit: np.ndarray,
                            yerr: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate various goodness-of-fit statistics.
    
    Parameters:
    -----------
    y_data : array-like
        Observed data values
    y_fit : array-like
        Fitted model values
    yerr : array-like, optional
        Uncertainties in y_data
        
    Returns:
    --------
    dict
        Dictionary of goodness-of-fit statistics
    """
    residuals = y_data - y_fit
    n_points = len(y_data)
    
    # R-squared
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_data - np.mean(y_data))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # Root mean square error
    rmse = np.sqrt(np.mean(residuals**2))
    
    # Mean absolute error
    mae = np.mean(np.abs(residuals))
    
    # Normalized root mean square error
    if np.mean(y_data) != 0:
        nrmse = rmse / np.mean(y_data)
    else:
        nrmse = np.inf
    
    # Weighted statistics if errors provided
    if yerr is not None:
        weighted_residuals = residuals / yerr
        weighted_chi_squared = np.sum(weighted_residuals**2)
        weighted_rmse = np.sqrt(np.mean(weighted_residuals**2))
    else:
        weighted_chi_squared = ss_res
        weighted_rmse = rmse
    
    return {
        'r_squared': r_squared,
        'rmse': rmse,
        'mae': mae,
        'nrmse': nrmse,
        'weighted_chi_squared': weighted_chi_squared,
        'weighted_rmse': weighted_rmse,
        'residual_std': np.std(residuals),
        'residual_mean': np.mean(residuals)
    }
