"""
Model Comparison Framework

This module provides automated model selection and comparison functionality
for Lorentzian fitting with multiple components.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import warnings

from .fitting import LorentzianFitter
from .metrics import (model_selection_criteria, likelihood_ratio_test, f_test,
                     assess_model_adequacy, calculate_delta_ic)
from .errors import FittingError


class ModelComparison:
    """
    Class for comparing multiple Lorentzian models and selecting the best fit.
    """
    
    def __init__(self, delta_aic_threshold: float = 20.0, 
                 delta_bic_threshold: float = 10.0,
                 max_components: int = 5):
        """
        Initialize model comparison.
        
        Parameters:
        -----------
        delta_aic_threshold : float
            Threshold for AIC difference to justify additional components
        delta_bic_threshold : float
            Threshold for BIC difference to justify additional components
        max_components : int
            Maximum number of components to test
        """
        self.delta_aic_threshold = delta_aic_threshold
        self.delta_bic_threshold = delta_bic_threshold
        self.max_components = max_components
        self.fitter = LorentzianFitter()
        
    def compare_models(self, x: np.ndarray, y: np.ndarray, 
                      yerr: Optional[np.ndarray] = None,
                      test_components: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Compare models with different numbers of components.
        
        Parameters:
        -----------
        x : array-like
            Independent variable
        y : array-like
            Dependent variable
        yerr : array-like, optional
            Uncertainties in y
        test_components : list, optional
            List of component numbers to test (default: [0, 1, 2, ...])
            
        Returns:
        --------
        dict
            Comprehensive model comparison results
        """
        if test_components is None:
            test_components = list(range(0, self.max_components + 1))
        
        results = {}
        fit_results = {}
        
        # Test each model
        for n_comp in test_components:
            model_name = f"{n_comp}_component{'s' if n_comp != 1 else ''}"
            
            try:
                if n_comp == 0:
                    # Baseline model (constant)
                    baseline = np.mean(y)
                    y_fit = np.full_like(y, baseline)
                    residuals = y - y_fit
                    
                    if yerr is not None:
                        chi2 = np.sum((residuals / yerr) ** 2)
                    else:
                        chi2 = np.sum(residuals ** 2) / np.var(residuals)
                    
                    n_params = 1
                    dof = len(y) - n_params
                    
                    # Calculate information criteria manually
                    from .metrics import calculate_aic, calculate_bic, calculate_goodness_of_fit
                    
                    fit_info = {
                        'fitted_curve': y_fit,
                        'residuals': residuals,
                        'chi_squared': chi2,
                        'reduced_chi_squared': chi2 / dof,
                        'degrees_of_freedom': dof,
                        'aic': calculate_aic(chi2, n_params, len(y)),
                        'bic': calculate_bic(chi2, n_params, len(y)),
                        'n_params': n_params,
                        'n_points': len(y)
                    }
                    
                    # Add goodness of fit metrics
                    gof_metrics = calculate_goodness_of_fit(y, y_fit, yerr)
                    fit_info.update(gof_metrics)
                    
                    params = np.array([baseline])
                    param_errors = np.array([np.std(residuals) / np.sqrt(len(y))])
                    
                else:
                    # Fit Lorentzian model
                    if n_comp == 1:
                        params, param_errors, fit_info = self.fitter.fit_single(x, y, yerr)
                    else:
                        params, param_errors, fit_info = self.fitter.fit_multiple(
                            x, y, n_comp, yerr)
                
                results[model_name] = fit_info
                fit_results[model_name] = {
                    'params': params,
                    'param_errors': param_errors,
                    'n_components': n_comp
                }
                
            except Exception as e:
                warnings.warn(f"Failed to fit {model_name}: {e}")
                continue
        
        if not results:
            raise FittingError("All model fits failed")
        
        # Perform model selection
        selection_results = model_selection_criteria(results)
        
        # Perform statistical tests between nested models
        statistical_tests = self._perform_statistical_tests(results, test_components)
        
        # Assess model adequacy
        adequacy_results = {}
        for model_name, result in results.items():
            adequacy = assess_model_adequacy(result['residuals'], yerr)
            adequacy_results[model_name] = adequacy
        
        # Make recommendations
        recommendations = self._make_recommendations(selection_results, statistical_tests)
        
        return {
            'fit_results': fit_results,
            'statistics': results,
            'model_selection': selection_results,
            'statistical_tests': statistical_tests,
            'adequacy_assessment': adequacy_results,
            'recommendations': recommendations
        }
    
    def _perform_statistical_tests(self, results: Dict[str, Dict], 
                                  test_components: List[int]) -> Dict[str, Any]:
        """Perform statistical tests between nested models."""
        tests = {}
        
        # Sort components for nested comparisons
        sorted_components = sorted(test_components)
        
        for i in range(len(sorted_components) - 1):
            simple_comp = sorted_components[i]
            complex_comp = sorted_components[i + 1]
            
            simple_name = f"{simple_comp}_component{'s' if simple_comp != 1 else ''}"
            complex_name = f"{complex_comp}_component{'s' if complex_comp != 1 else ''}"
            
            if simple_name in results and complex_name in results:
                simple_result = results[simple_name]
                complex_result = results[complex_name]
                
                # Likelihood ratio test
                try:
                    lr_stat, lr_p = likelihood_ratio_test(
                        simple_result['chi_squared'],
                        complex_result['chi_squared'],
                        simple_result['degrees_of_freedom'],
                        complex_result['degrees_of_freedom']
                    )
                    
                    # F-test
                    f_stat, f_p = f_test(
                        simple_result['chi_squared'],
                        complex_result['chi_squared'],
                        simple_result['degrees_of_freedom'],
                        complex_result['degrees_of_freedom']
                    )
                    
                    test_name = f"{simple_name}_vs_{complex_name}"
                    tests[test_name] = {
                        'likelihood_ratio': {
                            'statistic': lr_stat,
                            'p_value': lr_p,
                            'significant': lr_p < 0.05
                        },
                        'f_test': {
                            'statistic': f_stat,
                            'p_value': f_p,
                            'significant': f_p < 0.05
                        }
                    }
                    
                except Exception as e:
                    warnings.warn(f"Statistical test failed for {test_name}: {e}")
        
        return tests
    
    def _make_recommendations(self, selection_results: Dict, 
                            statistical_tests: Dict) -> Dict[str, str]:
        """Make final model recommendations based on all criteria."""
        recommendations = {}
        
        best_aic = selection_results.get('best_aic_model')
        best_bic = selection_results.get('best_bic_model')
        
        delta_aic = selection_results.get('summary', {}).get('delta_aic', {})
        delta_bic = selection_results.get('summary', {}).get('delta_bic', {})
        
        # Primary recommendation based on AIC with threshold
        if best_aic:
            # Check if the best AIC model is decisively better
            other_models = [m for m in delta_aic.keys() if m != best_aic]
            min_delta_aic = min([delta_aic[m] for m in other_models]) if other_models else np.inf
            
            if min_delta_aic >= self.delta_aic_threshold:
                recommendations['primary'] = f"Strong evidence for {best_aic} (ΔAICc ≥ {self.delta_aic_threshold})"
            elif min_delta_aic >= 10:
                recommendations['primary'] = f"Very strong evidence for {best_aic} (ΔAICc ≥ 10)"
            elif min_delta_aic >= 4:
                recommendations['primary'] = f"Considerable evidence for {best_aic} (ΔAICc ≥ 4)"
            else:
                recommendations['primary'] = f"Weak evidence for {best_aic} (ΔAICc < 4)"
        
        # Secondary recommendation based on BIC
        if best_bic:
            other_models = [m for m in delta_bic.keys() if m != best_bic]
            min_delta_bic = min([delta_bic[m] for m in other_models]) if other_models else np.inf
            
            if min_delta_bic >= self.delta_bic_threshold:
                recommendations['secondary'] = f"BIC strongly favors {best_bic} (ΔBIC ≥ {self.delta_bic_threshold})"
            else:
                recommendations['secondary'] = f"BIC favors {best_bic} (ΔBIC = {min_delta_bic:.1f})"
        
        # Check for consistency
        if best_aic == best_bic:
            recommendations['consensus'] = f"Both AIC and BIC favor {best_aic}"
        else:
            recommendations['consensus'] = f"AIC favors {best_aic}, BIC favors {best_bic}"
        
        # Statistical test recommendations
        significant_improvements = []
        for test_name, test_result in statistical_tests.items():
            if (test_result.get('likelihood_ratio', {}).get('significant', False) or 
                test_result.get('f_test', {}).get('significant', False)):
                significant_improvements.append(test_name)
        
        if significant_improvements:
            recommendations['statistical'] = f"Significant improvements: {', '.join(significant_improvements)}"
        else:
            recommendations['statistical'] = "No statistically significant improvements found"
        
        return recommendations


def automated_model_selection(x: np.ndarray, y: np.ndarray, 
                            yerr: Optional[np.ndarray] = None,
                            max_components: int = 5) -> Tuple[int, Dict[str, Any]]:
    """
    Automated model selection for Lorentzian fitting.
    
    Parameters:
    -----------
    x : array-like
        Independent variable
    y : array-like
        Dependent variable
    yerr : array-like, optional
        Uncertainties in y
    max_components : int
        Maximum number of components to test
        
    Returns:
    --------
    tuple
        (best_n_components, comparison_results)
    """
    comparator = ModelComparison(max_components=max_components)
    results = comparator.compare_models(x, y, yerr)
    
    # Extract best model based on AIC (with thresholds)
    best_aic_model = results['model_selection']['best_aic_model']
    
    # Parse number of components from model name
    if 'component' in best_aic_model:
        best_n_components = int(best_aic_model.split('_')[0])
    else:
        best_n_components = 1
    
    return best_n_components, results
