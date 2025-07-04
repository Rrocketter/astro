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


def likelihood_ratio_test(chi2_simple: float, chi2_complex: float, 
                         df_simple: int, df_complex: int) -> Tuple[float, float]:
    """
    Perform likelihood ratio test between nested models.
    
    Parameters:
    -----------
    chi2_simple : float
        Chi-squared for simpler model
    chi2_complex : float
        Chi-squared for more complex model
    df_simple : int
        Degrees of freedom for simpler model
    df_complex : int
        Degrees of freedom for more complex model
        
    Returns:
    --------
    tuple
        (test_statistic, p_value)
    """
    from scipy import stats
    
    if df_simple <= df_complex:
        raise ValueError("Simple model must have more degrees of freedom")
    
    # Test statistic is difference in chi-squared
    test_stat = chi2_simple - chi2_complex
    df_diff = df_simple - df_complex
    
    # P-value from chi-squared distribution
    p_value = 1 - stats.chi2.cdf(test_stat, df_diff)
    
    return test_stat, p_value


def f_test(chi2_simple: float, chi2_complex: float,
           df_simple: int, df_complex: int) -> Tuple[float, float]:
    """
    Perform F-test for model comparison.
    
    Parameters:
    -----------
    chi2_simple : float
        Chi-squared for simpler model
    chi2_complex : float
        Chi-squared for more complex model
    df_simple : int
        Degrees of freedom for simpler model
    df_complex : int
        Degrees of freedom for more complex model
        
    Returns:
    --------
    tuple
        (f_statistic, p_value)
    """
    from scipy import stats
    
    if df_simple <= df_complex:
        raise ValueError("Simple model must have more degrees of freedom")
    
    # Calculate F-statistic
    numerator = (chi2_simple - chi2_complex) / (df_simple - df_complex)
    denominator = chi2_complex / df_complex
    
    if denominator == 0:
        return np.inf, 0.0
    
    f_stat = numerator / denominator
    df_num = df_simple - df_complex
    df_den = df_complex
    
    # P-value from F-distribution
    p_value = 1 - stats.f.cdf(f_stat, df_num, df_den)
    
    return f_stat, p_value


def model_selection_criteria(results: Dict[str, Dict]) -> Dict[str, any]:
    """
    Apply model selection criteria to compare multiple models.
    
    Parameters:
    -----------
    results : dict
        Dictionary with model names as keys and fit results as values
        Each result should contain 'aic', 'bic', 'chi_squared', etc.
        
    Returns:
    --------
    dict
        Model selection summary with recommendations
    """
    if not results:
        return {}
    
    # Extract information criteria
    aic_values = {model: result['aic'] for model, result in results.items()}
    bic_values = {model: result['bic'] for model, result in results.items()}
    
    # Calculate deltas
    delta_aic = calculate_delta_ic(aic_values)
    delta_bic = calculate_delta_ic(bic_values)
    
    # Calculate Akaike weights
    akaike_weights = calculate_akaike_weights(aic_values)
    
    # Find best models
    best_aic = min(aic_values, key=aic_values.get)
    best_bic = min(bic_values, key=bic_values.get)
    
    # Apply decision thresholds
    strong_evidence_threshold = 10  # Delta > 10 = strong evidence
    decisive_evidence_threshold = 20  # Delta > 20 = decisive evidence
    
    recommendations = {}
    
    for model in results.keys():
        delta_aic_val = delta_aic[model]
        delta_bic_val = delta_bic[model]
        weight = akaike_weights[model]
        
        # AIC-based recommendation
        if delta_aic_val < 2:
            aic_support = "substantial"
        elif delta_aic_val < 4:
            aic_support = "considerably less"
        elif delta_aic_val < strong_evidence_threshold:
            aic_support = "much less"
        elif delta_aic_val < decisive_evidence_threshold:
            aic_support = "very strong evidence against"
        else:
            aic_support = "decisive evidence against"
        
        # BIC-based recommendation
        if delta_bic_val < 2:
            bic_support = "weak evidence against"
        elif delta_bic_val < 6:
            bic_support = "positive evidence against"
        elif delta_bic_val < strong_evidence_threshold:
            bic_support = "strong evidence against"
        else:
            bic_support = "very strong evidence against"
        
        recommendations[model] = {
            'delta_aic': delta_aic_val,
            'delta_bic': delta_bic_val,
            'akaike_weight': weight,
            'aic_support': aic_support,
            'bic_support': bic_support
        }
    
    return {
        'best_aic_model': best_aic,
        'best_bic_model': best_bic,
        'model_recommendations': recommendations,
        'summary': {
            'aic_values': aic_values,
            'bic_values': bic_values,
            'delta_aic': delta_aic,
            'delta_bic': delta_bic,
            'akaike_weights': akaike_weights
        }
    }


def assess_model_adequacy(residuals: np.ndarray, yerr: Optional[np.ndarray] = None,
                         alpha: float = 0.05) -> Dict[str, any]:
    """
    Assess model adequacy using residual analysis.
    
    Parameters:
    -----------
    residuals : array-like
        Model residuals
    yerr : array-like, optional
        Data uncertainties
    alpha : float
        Significance level for tests
        
    Returns:
    --------
    dict
        Dictionary of adequacy test results
    """
    from scipy import stats
    
    n_points = len(residuals)
    
    if yerr is not None:
        standardized_residuals = residuals / yerr
    else:
        standardized_residuals = residuals / np.std(residuals)
    
    # Normality tests
    shapiro_stat, shapiro_p = stats.shapiro(standardized_residuals)
    
    # Randomness test (runs test)
    median_resid = np.median(standardized_residuals)
    runs, n1, n2 = _runs_test(standardized_residuals > median_resid)
    
    # Expected runs and variance
    expected_runs = 2 * n1 * n2 / (n1 + n2) + 1
    var_runs = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2)**2 * (n1 + n2 - 1))
    
    if var_runs > 0:
        z_runs = (runs - expected_runs) / np.sqrt(var_runs)
        runs_p = 2 * (1 - stats.norm.cdf(abs(z_runs)))
    else:
        z_runs = 0
        runs_p = 1.0
    
    # Outlier detection (beyond 3 sigma)
    outliers = np.abs(standardized_residuals) > 3
    n_outliers = np.sum(outliers)
    outlier_fraction = n_outliers / n_points
    
    return {
        'normality_test': {
            'shapiro_statistic': shapiro_stat,
            'shapiro_p_value': shapiro_p,
            'is_normal': shapiro_p > alpha
        },
        'randomness_test': {
            'runs_statistic': z_runs,
            'runs_p_value': runs_p,
            'is_random': runs_p > alpha
        },
        'outlier_analysis': {
            'n_outliers': n_outliers,
            'outlier_fraction': outlier_fraction,
            'outlier_indices': np.where(outliers)[0].tolist()
        },
        'residual_stats': {
            'mean': np.mean(standardized_residuals),
            'std': np.std(standardized_residuals),
            'skewness': stats.skew(standardized_residuals),
            'kurtosis': stats.kurtosis(standardized_residuals)
        }
    }


def _runs_test(binary_sequence: np.ndarray) -> Tuple[int, int, int]:
    """
    Helper function for runs test.
    
    Parameters:
    -----------
    binary_sequence : array-like
        Boolean array
        
    Returns:
    --------
    tuple
        (number_of_runs, n_true, n_false)
    """
    n1 = np.sum(binary_sequence)  # Number of True values
    n2 = len(binary_sequence) - n1  # Number of False values
    
    # Count runs
    runs = 1
    for i in range(1, len(binary_sequence)):
        if binary_sequence[i] != binary_sequence[i-1]:
            runs += 1
    
    return runs, n1, n2


def cross_validation_score(x: np.ndarray, y: np.ndarray, yerr: np.ndarray,
                          fit_function, k_folds: int = 5) -> Dict[str, float]:
    """
    Calculate cross-validation score for model assessment.
    
    Parameters:
    -----------
    x : array-like
        Independent variable
    y : array-like
        Dependent variable
    yerr : array-like
        Uncertainties in y
    fit_function : callable
        Function that performs the fit
    k_folds : int
        Number of cross-validation folds
        
    Returns:
    --------
    dict
        Cross-validation metrics
    """
    from sklearn.model_selection import KFold
    
    n_points = len(x)
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    cv_scores = []
    cv_chi2 = []
    
    for train_idx, test_idx in kf.split(x):
        x_train, x_test = x[train_idx], x[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        yerr_train, yerr_test = yerr[train_idx], yerr[test_idx]
        
        try:
            # Fit on training data
            params, _, _ = fit_function(x_train, y_train, yerr_train)
            
            # Predict on test data (need to determine model type)
            # This is a simplified version - would need model-specific prediction
            y_pred = np.zeros_like(y_test)  # Placeholder
            
            # Calculate test score
            residuals = y_test - y_pred
            chi2 = np.sum((residuals / yerr_test) ** 2)
            score = 1 - np.sum(residuals**2) / np.sum((y_test - np.mean(y_test))**2)
            
            cv_scores.append(score)
            cv_chi2.append(chi2)
            
        except Exception:
            # Skip failed folds
            continue
    
    if not cv_scores:
        return {'mean_score': 0.0, 'std_score': 0.0, 'mean_chi2': np.inf}
    
    return {
        'mean_score': np.mean(cv_scores),
        'std_score': np.std(cv_scores),
        'mean_chi2': np.mean(cv_chi2),
        'std_chi2': np.std(cv_chi2),
        'n_successful_folds': len(cv_scores)
    }
