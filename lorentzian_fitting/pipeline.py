"""
Automated Fitting Pipeline

This module provides a comprehensive automated pipeline for Lorentzian fitting
with intelligent model selection and robust error handling.
"""

import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass

from .fitting import LorentzianFitter
from .comparison import ModelComparison
from .metrics import calculate_delta_ic, calculate_akaike_weights
from .errors import FittingError, ParameterError, DataError


@dataclass
class PipelineSettings:
    """Configuration settings for the automated pipeline."""
    
    # Model selection thresholds
    delta_aic_threshold: float = 20.0  # Decisive evidence threshold
    delta_bic_threshold: float = 10.0  # Strong evidence threshold
    strong_evidence_threshold: float = 10.0  # Very strong evidence
    moderate_evidence_threshold: float = 4.0  # Considerable evidence
    
    # Component limits
    max_components: int = 5
    min_components: int = 0
    
    # Fitting parameters
    max_iterations: int = 2000
    tolerance: float = 1e-8
    confidence_level: float = 0.68
    
    # Robustness settings
    n_retry_attempts: int = 3
    enable_bootstrap: bool = False
    bootstrap_samples: int = 100
    enable_monte_carlo: bool = True
    monte_carlo_samples: int = 500
    
    # Error estimation
    confidence_levels: List[float] = None  # [0.68, 0.95] for 1σ and 2σ
    use_robust_errors: bool = True
    min_parameter_snr: float = 2.0  # Minimum signal-to-noise for parameters
    
    # Edge case handling
    handle_edge_cases: bool = True
    min_data_points: int = 10
    max_condition_number: float = 1e12
    outlier_detection: bool = True
    outlier_threshold: float = 3.0

    def __post_init__(self):
        """Set default confidence levels if not provided."""
        if self.confidence_levels is None:
            self.confidence_levels = [0.68, 0.95]

class AutomatedPipeline:
    """
    Automated pipeline for Lorentzian fitting with intelligent model selection.
    """
    
    def __init__(self, settings: Optional[PipelineSettings] = None):
        """
        Initialize the automated pipeline.
        
        Parameters:
        -----------
        settings : PipelineSettings, optional
            Configuration settings for the pipeline
        """
        self.settings = settings or PipelineSettings()
        self.fitter = LorentzianFitter(
            max_iterations=self.settings.max_iterations,
            tolerance=self.settings.tolerance
        )
        self.comparison = ModelComparison(
            delta_aic_threshold=self.settings.delta_aic_threshold,
            delta_bic_threshold=self.settings.delta_bic_threshold,
            max_components=self.settings.max_components
        )
        
        # Initialize robust error estimation components
        if self.settings.enable_bootstrap or self.settings.enable_monte_carlo:
            np.random.seed(42)  # For reproducible error estimates
    
    def fit_with_model_selection(self, x: np.ndarray, y: np.ndarray,
                               yerr: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Perform automated fitting with intelligent model selection.
        
        Parameters:
        -----------
        x : array-like
            Independent variable
        y : array-like
            Dependent variable
        yerr : array-like, optional
            Uncertainties in y
            
        Returns:
        --------
        dict
            Comprehensive fitting and selection results
        """
        # Validate input data
        self._validate_input(x, y, yerr)
        
        # Enhanced input validation with edge case detection
        edge_case_info = self._detect_edge_cases(x, y, yerr)
        
        # Apply preprocessing if needed
        x_proc, y_proc, yerr_proc = self._preprocess_data(x, y, yerr, edge_case_info)
        
        # Determine component range to test
        test_components = self._determine_component_range(x_proc, y_proc)
        
        # Fit all models with enhanced robustness
        fit_results = self._fit_all_models_robust(x_proc, y_proc, yerr_proc, test_components)
        
        # Enhanced error estimation for all successful fits
        if self.settings.use_robust_errors:
            fit_results = self._enhance_error_estimates(x_proc, y_proc, yerr_proc, fit_results)
        
        # Perform comprehensive model comparison
        comparison_results = self._perform_model_comparison(fit_results)
        
        # Apply intelligent selection logic with edge case considerations
        selection_result = self._apply_selection_logic_robust(comparison_results, edge_case_info)
        
        # Validate final selection with comprehensive checks
        final_result = self._validate_and_finalize_robust(selection_result, x_proc, y_proc, yerr_proc, edge_case_info)
        
        return final_result
    
    def _validate_input(self, x: np.ndarray, y: np.ndarray, 
                       yerr: Optional[np.ndarray]) -> None:
        """Validate input data for pipeline processing."""
        if len(x) < 4:
            raise DataError("Insufficient data points for meaningful fitting")
        
        if len(x) != len(y):
            raise DataError("x and y arrays must have same length")
        
        if yerr is not None and len(yerr) != len(x):
            raise DataError("yerr array must have same length as x and y")
        
        # Check for minimum data quality
        if np.ptp(y) < 1e-10:
            raise DataError("Data has insufficient dynamic range")
        
        # Check for reasonable signal-to-noise
        if yerr is not None:
            snr = np.max(y) / np.median(yerr)
            if snr < 2:
                warnings.warn("Low signal-to-noise ratio detected")
    
    def _detect_edge_cases(self, x: np.ndarray, y: np.ndarray, 
                          yerr: Optional[np.ndarray]) -> Dict[str, Any]:
        """Detect potential edge cases in the data."""
        edge_cases = {
            'detected_issues': [],
            'severity': 'none',  # 'none', 'minor', 'major', 'critical'
            'recommendations': []
        }
        
        if not self.settings.handle_edge_cases:
            return edge_cases
        
        # Check data quantity
        if len(x) < self.settings.min_data_points:
            edge_cases['detected_issues'].append('insufficient_data')
            edge_cases['severity'] = 'critical'
            edge_cases['recommendations'].append('Need more data points')
        
        # Check for outliers
        if self.settings.outlier_detection and yerr is not None:
            outliers = self._detect_outliers(y, yerr)
            if outliers['n_outliers'] > 0:
                edge_cases['detected_issues'].append('outliers_present')
                edge_cases['outlier_info'] = outliers
                if outliers['fraction'] > 0.1:
                    edge_cases['severity'] = 'major'
                    edge_cases['recommendations'].append('Consider outlier removal')
                else:
                    edge_cases['severity'] = 'minor'
        
        # Check data quality
        snr_issues = self._assess_data_quality(y, yerr)
        if snr_issues['low_snr']:
            edge_cases['detected_issues'].append('low_signal_to_noise')
            edge_cases['snr_info'] = snr_issues
            if edge_cases['severity'] != 'critical':
                edge_cases['severity'] = 'major'
            edge_cases['recommendations'].append('Low S/N may affect fit reliability')
        
        # Check for systematic trends
        trend_info = self._check_systematic_trends(x, y)
        if trend_info['has_trend']:
            edge_cases['detected_issues'].append('systematic_trend')
            edge_cases['trend_info'] = trend_info
            if edge_cases['severity'] == 'none':
                edge_cases['severity'] = 'minor'
            edge_cases['recommendations'].append('Consider detrending data')
        
        # Check for numerical issues
        numerical_issues = self._check_numerical_issues(x, y)
        if numerical_issues['has_issues']:
            edge_cases['detected_issues'].append('numerical_issues')
            edge_cases['numerical_info'] = numerical_issues
            if edge_cases['severity'] != 'critical':
                edge_cases['severity'] = 'major'
        
        return edge_cases
    
    def _detect_outliers(self, y: np.ndarray, yerr: np.ndarray) -> Dict[str, Any]:
        """Detect outliers using robust statistical methods."""
        # Use median and MAD for robust outlier detection
        median_y = np.median(y)
        mad = np.median(np.abs(y - median_y))
        
        # Normalized residuals
        if yerr is not None:
            normalized_residuals = (y - median_y) / yerr
        else:
            normalized_residuals = (y - median_y) / (1.4826 * mad)  # MAD to std conversion
        
        # Identify outliers
        outlier_mask = np.abs(normalized_residuals) > self.settings.outlier_threshold
        
        return {
            'outlier_indices': np.where(outlier_mask)[0].tolist(),
            'n_outliers': np.sum(outlier_mask),
            'fraction': np.sum(outlier_mask) / len(y),
            'outlier_residuals': normalized_residuals[outlier_mask].tolist()
        }
    
    def _assess_data_quality(self, y: np.ndarray, yerr: Optional[np.ndarray]) -> Dict[str, Any]:
        """Assess overall data quality."""
        if yerr is None:
            # Estimate noise from data
            noise_est = np.std(np.diff(y)) / np.sqrt(2)
            signal_est = np.ptp(y)
            snr = signal_est / noise_est if noise_est > 0 else np.inf
        else:
            signal_est = np.ptp(y)
            noise_est = np.median(yerr)
            snr = signal_est / noise_est if noise_est > 0 else np.inf
        
        return {
            'signal_to_noise': snr,
            'low_snr': snr < 5.0,
            'very_low_snr': snr < 2.0,
            'signal_estimate': signal_est,
            'noise_estimate': noise_est
        }
    
    def _check_systematic_trends(self, x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Check for systematic trends in the data."""
        try:
            from scipy import stats
            
            # Linear trend test
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            # Significant trend if p < 0.05 and |r| > 0.3
            has_trend = (p_value < 0.05) and (abs(r_value) > 0.3)
            
            return {
                'has_trend': has_trend,
                'slope': slope,
                'r_squared': r_value**2,
                'p_value': p_value,
                'trend_strength': 'strong' if abs(r_value) > 0.7 else 'moderate' if abs(r_value) > 0.3 else 'weak'
            }
        except Exception:
            return {'has_trend': False, 'error': 'Could not assess trends'}
    
    def _check_numerical_issues(self, x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Check for potential numerical issues."""
        issues = []
        
        # Check dynamic range
        x_range = np.ptp(x)
        y_range = np.ptp(y)
        
        if x_range < 1e-10:
            issues.append('zero_x_range')
        if y_range < 1e-10:
            issues.append('zero_y_range')
        
        # Check for extreme values
        if np.any(np.abs(x) > 1e10) or np.any(np.abs(y) > 1e10):
            issues.append('extreme_values')
        
        # Check for very small values that might cause underflow
        if np.any(np.abs(x[x != 0]) < 1e-10) or np.any(np.abs(y[y != 0]) < 1e-10):
            issues.append('potential_underflow')
        
        return {
            'has_issues': len(issues) > 0,
            'issues': issues,
            'x_range': x_range,
            'y_range': y_range
        }
    
    def _preprocess_data(self, x: np.ndarray, y: np.ndarray, 
                        yerr: Optional[np.ndarray], edge_case_info: Dict) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Preprocess data to handle edge cases."""
        x_proc, y_proc, yerr_proc = x.copy(), y.copy(), yerr.copy() if yerr is not None else None
        
        if not self.settings.handle_edge_cases:
            return x_proc, y_proc, yerr_proc
        
        # Handle outliers if detected
        if 'outliers_present' in edge_case_info['detected_issues']:
            outlier_info = edge_case_info.get('outlier_info', {})
            if outlier_info.get('fraction', 0) > 0.2:  # More than 20% outliers
                warnings.warn("High fraction of outliers detected. Consider manual inspection.")
            # For now, keep all data but flag for user awareness
        
        # Handle systematic trends
        if 'systematic_trend' in edge_case_info['detected_issues']:
            trend_info = edge_case_info.get('trend_info', {})
            if trend_info.get('trend_strength') == 'strong':
                warnings.warn("Strong systematic trend detected. Consider detrending.")
        
        # Handle numerical scaling issues
        if 'numerical_issues' in edge_case_info['detected_issues']:
            numerical_info = edge_case_info.get('numerical_info', {})
            if 'extreme_values' in numerical_info.get('issues', []):
                # Apply mild rescaling if needed
                x_scale = np.median(np.abs(x_proc[x_proc != 0])) if np.any(x_proc != 0) else 1.0
                y_scale = np.median(np.abs(y_proc[y_proc != 0])) if np.any(y_proc != 0) else 1.0
                
                if x_scale > 1e6 or x_scale < 1e-6:
                    warnings.warn("Extreme x-values detected. Consider rescaling for numerical stability.")
                if y_scale > 1e6 or y_scale < 1e-6:
                    warnings.warn("Extreme y-values detected. Consider rescaling for numerical stability.")
        
        return x_proc, y_proc, yerr_proc
    
    def _fit_all_models_robust(self, x: np.ndarray, y: np.ndarray,
                              yerr: Optional[np.ndarray], 
                              test_components: List[int]) -> Dict[str, Any]:
        """Enhanced model fitting with robust error handling."""
        fit_results = {}
        failed_models = []
        
        for n_comp in test_components:
            model_name = f"{n_comp}_component{'s' if n_comp != 1 else ''}"
            
            # Multiple fitting strategies for robustness
            strategies = self._get_fitting_strategies(n_comp)
            
            success = False
            for strategy_name, strategy_func in strategies.items():
                try:
                    result = strategy_func(x, y, yerr, n_comp)
                    
                    # Validate result quality
                    if self._validate_fit_result(result, x, y):
                        fit_results[model_name] = result
                        fit_results[model_name]['fitting_strategy'] = strategy_name
                        success = True
                        break
                
                except Exception as e:
                    continue  # Try next strategy
            
            if not success:
                failed_models.append(model_name)
                warnings.warn(f"All fitting strategies failed for {model_name}")
        
        if not fit_results:
            raise FittingError("All model fits failed with all strategies")
        
        # Record failures for analysis
        if failed_models:
            for model_name in fit_results:
                fit_results[model_name]['failed_models'] = failed_models
        
        return fit_results
    
    def _get_fitting_strategies(self, n_comp: int) -> Dict[str, Callable]:
        """Get different fitting strategies for robustness."""
        strategies = {}
        
        # Standard strategy
        strategies['standard'] = lambda x, y, yerr, n: self._fit_standard_strategy(x, y, yerr, n)
        
        # Conservative strategy (tighter bounds)
        strategies['conservative'] = lambda x, y, yerr, n: self._fit_conservative_strategy(x, y, yerr, n)
        
        # Relaxed strategy (looser bounds)
        strategies['relaxed'] = lambda x, y, yerr, n: self._fit_relaxed_strategy(x, y, yerr, n)
        
        # Multiple initial guesses strategy
        strategies['multiple_init'] = lambda x, y, yerr, n: self._fit_multiple_init_strategy(x, y, yerr, n)
        
        return strategies
    
    def _fit_standard_strategy(self, x: np.ndarray, y: np.ndarray,
                              yerr: Optional[np.ndarray], n_comp: int) -> Dict[str, Any]:
        """Standard fitting strategy."""
        if n_comp == 0:
            return self._fit_baseline_model(x, y, yerr)
        else:
            return self._fit_lorentzian_model(x, y, yerr, n_comp, 0)
    
    def _fit_conservative_strategy(self, x: np.ndarray, y: np.ndarray,
                                  yerr: Optional[np.ndarray], n_comp: int) -> Dict[str, Any]:
        """Conservative fitting with tighter parameter bounds."""
        if n_comp == 0:
            return self._fit_baseline_model(x, y, yerr)
        
        # Modify fitter settings for conservative approach
        conservative_fitter = LorentzianFitter(
            max_iterations=self.settings.max_iterations // 2,
            tolerance=self.settings.tolerance * 10
        )
        
        if n_comp == 1:
            params, param_errors, fit_info = conservative_fitter.fit_single(x, y, yerr)
        else:
            params, param_errors, fit_info = conservative_fitter.fit_multiple(x, y, n_comp, yerr)
        
        return {
            'params': params,
            'param_errors': param_errors,
            'fit_info': fit_info
        }
    
    def _fit_relaxed_strategy(self, x: np.ndarray, y: np.ndarray,
                             yerr: Optional[np.ndarray], n_comp: int) -> Dict[str, Any]:
        """Relaxed fitting with looser bounds and more iterations."""
        if n_comp == 0:
            return self._fit_baseline_model(x, y, yerr)
        
        # Modify fitter settings for relaxed approach
        relaxed_fitter = LorentzianFitter(
            max_iterations=self.settings.max_iterations * 2,
            tolerance=self.settings.tolerance / 10
        )
        
        if n_comp == 1:
            params, param_errors, fit_info = relaxed_fitter.fit_single(x, y, yerr)
        else:
            params, param_errors, fit_info = relaxed_fitter.fit_multiple(x, y, n_comp, yerr)
        
        return {
            'params': params,
            'param_errors': param_errors,
            'fit_info': fit_info
        }
    
    def _fit_multiple_init_strategy(self, x: np.ndarray, y: np.ndarray,
                                   yerr: Optional[np.ndarray], n_comp: int) -> Dict[str, Any]:
        """Strategy using multiple initial guesses and selecting best result."""
        if n_comp == 0:
            return self._fit_baseline_model(x, y, yerr)
        
        best_result = None
        best_chi2 = np.inf
        
        # Try multiple initial guesses
        for attempt in range(5):
            try:
                initial_guess = self._generate_perturbed_guess(x, y, n_comp, attempt)
                
                if n_comp == 1:
                    params, param_errors, fit_info = self.fitter.fit_single(x, y, yerr, initial_guess)
                else:
                    params, param_errors, fit_info = self.fitter.fit_multiple(x, y, n_comp, yerr, initial_guess)
                
                result = {
                    'params': params,
                    'param_errors': param_errors,
                    'fit_info': fit_info
                }
                
                # Select result with best chi-squared
                if fit_info['chi_squared'] < best_chi2:
                    best_chi2 = fit_info['chi_squared']
                    best_result = result
                    
            except Exception:
                continue
        
        if best_result is None:
            raise FittingError("All initial guesses failed")
        
        return best_result
    
    def _validate_fit_result(self, result: Dict[str, Any], x: np.ndarray, y: np.ndarray) -> bool:
        """Validate that a fit result is reasonable."""
        try:
            params = result['params']
            fit_info = result['fit_info']
            
            # Check parameter finiteness
            if not np.all(np.isfinite(params)):
                return False
            
            # Check chi-squared reasonableness
            if not np.isfinite(fit_info['chi_squared']) or fit_info['chi_squared'] < 0:
                return False
            
            # Check if fit is catastrophically bad
            if fit_info['reduced_chi_squared'] > 1000:
                return False
            
            # Check parameter ranges for physical reasonableness
            n_comp = fit_info.get('n_components', 0)
            if n_comp > 0:
                for i in range(n_comp):
                    idx = i * 3
                    if len(params) > idx + 2:
                        width = params[idx + 2]
                        if width <= 0 or width > 10 * np.ptp(x):  # Width too large
                            return False
            
            return True
            
        except Exception:
            return False
    
    def _enhance_error_estimates(self, x: np.ndarray, y: np.ndarray,
                                yerr: Optional[np.ndarray], 
                                fit_results: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance error estimates using robust methods."""
        enhanced_results = {}
        
        for model_name, result in fit_results.items():
            enhanced_result = result.copy()
            
            try:
                # Add bootstrap error estimates if enabled
                if self.settings.enable_bootstrap:
                    bootstrap_errors = self._bootstrap_errors(x, y, yerr, model_name, result)
                    enhanced_result['bootstrap_errors'] = bootstrap_errors
                
                # Add Monte Carlo error estimates if enabled
                if self.settings.enable_monte_carlo:
                    mc_errors = self._monte_carlo_errors(x, y, yerr, model_name, result)
                    enhanced_result['monte_carlo_errors'] = mc_errors
                
                # Calculate multiple confidence intervals
                for conf_level in self.settings.confidence_levels:
                    ci_key = f'confidence_interval_{int(conf_level*100)}'
                    ci = self._calculate_confidence_interval(result, conf_level)
                    enhanced_result[ci_key] = ci
                
                # Parameter significance assessment
                param_significance = self._assess_parameter_significance(result)
                enhanced_result['parameter_significance'] = param_significance
                
            except Exception as e:
                warnings.warn(f"Error enhancement failed for {model_name}: {e}")
                # Keep original result if enhancement fails
            
            enhanced_results[model_name] = enhanced_result
        
        return enhanced_results
    
    def _bootstrap_errors(self, x: np.ndarray, y: np.ndarray, yerr: np.ndarray,
                         model_name: str, fit_result: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate bootstrap error estimates."""
        n_comp = fit_result['fit_info'].get('n_components', 0)
        bootstrap_params = []
        
        for i in range(self.settings.bootstrap_samples):
            try:
                # Resample data
                indices = np.random.choice(len(x), size=len(x), replace=True)
                x_boot = x[indices]
                y_boot = y[indices]
                yerr_boot = yerr[indices] if yerr is not None else None
                
                # Fit resampled data
                if n_comp == 0:
                    boot_result = self._fit_baseline_model(x_boot, y_boot, yerr_boot)
                elif n_comp == 1:
                    params, _, _ = self.fitter.fit_single(x_boot, y_boot, yerr_boot)
                    boot_result = {'params': params}
                else:
                    params, _, _ = self.fitter.fit_multiple(x_boot, y_boot, n_comp, yerr_boot)
                    boot_result = {'params': params}
                
                bootstrap_params.append(boot_result['params'])
                
            except Exception:
                continue  # Skip failed bootstrap samples
        
        if bootstrap_params:
            bootstrap_params = np.array(bootstrap_params)
            return {
                'parameter_means': np.mean(bootstrap_params, axis=0),
                'parameter_stds': np.std(bootstrap_params, axis=0, ddof=1),
                'successful_samples': len(bootstrap_params),
                'success_rate': len(bootstrap_params) / self.settings.bootstrap_samples
            }
        else:
            return {'error': 'All bootstrap samples failed'}
    
    def _monte_carlo_errors(self, x: np.ndarray, y: np.ndarray, yerr: np.ndarray,
                           model_name: str, fit_result: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate Monte Carlo error estimates by adding noise to data."""
        if yerr is None:
            return {'error': 'Cannot perform Monte Carlo without error estimates'}
        
        n_comp = fit_result['fit_info'].get('n_components', 0)
        mc_params = []
        
        for i in range(self.settings.monte_carlo_samples):
            try:
                # Add random noise to data
                noise = np.random.normal(0, yerr)
                y_mc = y + noise
                
                # Fit noisy data
                if n_comp == 0:
                    mc_result = self._fit_baseline_model(x, y_mc, yerr)
                elif n_comp == 1:
                    params, _, _ = self.fitter.fit_single(x, y_mc, yerr)
                    mc_result = {'params': params}
                else:
                    params, _, _ = self.fitter.fit_multiple(x, y_mc, n_comp, yerr)
                    mc_result = {'params': params}
                
                mc_params.append(mc_result['params'])
                
            except Exception:
                continue  # Skip failed MC samples
        
        if mc_params:
            mc_params = np.array(mc_params)
            return {
                'parameter_means': np.mean(mc_params, axis=0),
                'parameter_stds': np.std(mc_params, axis=0, ddof=1),
                'successful_samples': len(mc_params),
                'success_rate': len(mc_params) / self.settings.monte_carlo_samples
            }
        else:
            return {'error': 'All Monte Carlo samples failed'}
    
    def _calculate_confidence_interval(self, fit_result: Dict[str, Any], 
                                     confidence_level: float) -> Dict[str, Any]:
        """Calculate confidence intervals at specified level."""
        from .metrics import calculate_confidence_intervals
        
        params = fit_result['params']
        param_errors = fit_result['param_errors']
        
        lower, upper = calculate_confidence_intervals(params, param_errors, confidence_level)
        
        return {
            'lower_bounds': lower,
            'upper_bounds': upper,
            'confidence_level': confidence_level
        }
    
    def _assess_parameter_significance(self, fit_result: Dict[str, Any]) -> Dict[str, Any]:
        """Assess statistical significance of fitted parameters."""
        params = fit_result['params']
        param_errors = fit_result['param_errors']
        
        # Calculate signal-to-noise ratios
        snr = np.abs(params) / param_errors
        
        # Determine significance
        significant = snr >= self.settings.min_parameter_snr
        
        return {
            'signal_to_noise_ratios': snr,
            'significant_parameters': significant,
            'n_significant': np.sum(significant),
            'significance_threshold': self.settings.min_parameter_snr
        }
    
    def _apply_selection_logic_robust(self, comparison_results: Dict[str, Any],
                                     edge_case_info: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced selection logic considering edge cases."""
        # Get base selection
        base_selection = self._apply_selection_logic(comparison_results)
        
        # Modify selection based on edge cases
        if edge_case_info['severity'] in ['major', 'critical']:
            # Apply more conservative selection for problematic data
            conservative_selection = self._apply_conservative_selection(
                comparison_results, edge_case_info, base_selection)
            return conservative_selection
        
        return base_selection
    
    def _apply_conservative_selection(self, comparison_results: Dict[str, Any],
                                     edge_case_info: Dict[str, Any],
                                     base_selection: Dict[str, Any]) -> Dict[str, Any]:
        """Apply conservative model selection for edge cases."""
        # Prefer simpler models for edge cases
        model_selection = comparison_results['model_selection']
        
        # Find the simplest model within reasonable evidence bounds
        delta_aic = model_selection['summary']['delta_aic']
        
        # Use stricter threshold for edge cases
        conservative_threshold = self.settings.moderate_evidence_threshold
        
        conservative_candidates = [model for model, delta in delta_aic.items() 
                                 if delta < conservative_threshold]
        
        if conservative_candidates:
            # Choose simplest model among conservative candidates
            fit_results = comparison_results['fit_results']
            simplest_model = min(conservative_candidates, 
                               key=lambda x: fit_results[x]['fit_info']['n_params'])
            
            base_selection['selected_model'] = simplest_model
            base_selection['selection_confidence'] = 'conservative'
            base_selection['selection_rationale'].append('Conservative selection due to edge cases')
            base_selection['edge_case_adjustment'] = True
        
        return base_selection
    
    def _validate_and_finalize_robust(self, selection_result: Dict[str, Any],
                                     x: np.ndarray, y: np.ndarray,
                                     yerr: Optional[np.ndarray],
                                     edge_case_info: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced validation and finalization with edge case reporting."""
        # Get base result
        base_result = self._validate_and_finalize(selection_result, x, y, yerr)
        
        # Add edge case information
        base_result['edge_case_analysis'] = edge_case_info
        
        # Enhanced validation considering edge cases
        enhanced_validation = self._enhanced_validation(base_result, edge_case_info)
        base_result['validation'].update(enhanced_validation)
        
        # Add reliability assessment
        reliability = self._assess_result_reliability(base_result, edge_case_info)
        base_result['reliability_assessment'] = reliability
        
        return base_result
    
    def _enhanced_validation(self, result: Dict[str, Any], 
                           edge_case_info: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced validation considering edge cases."""
        enhanced = {}
        
        # Adjust quality thresholds based on edge cases
        if edge_case_info['severity'] in ['major', 'critical']:
            enhanced['quality_adjusted_for_edge_cases'] = True
            enhanced['edge_case_severity'] = edge_case_info['severity']
            
            # More lenient thresholds for edge cases
            fit_info = result['best_fit']['fit_info']
            enhanced['acceptable_chi2_adjusted'] = 0.1 <= fit_info['reduced_chi_squared'] <= 5.0
            enhanced['acceptable_r_squared_adjusted'] = fit_info['r_squared'] > 0.5
        else:
            enhanced['quality_adjusted_for_edge_cases'] = False
        
        return enhanced
    
    def _assess_result_reliability(self, result: Dict[str, Any],
                                  edge_case_info: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall reliability of the fitting result."""
        reliability_factors = []
        reliability_score = 1.0
        
        # Edge case penalty
        if edge_case_info['severity'] == 'minor':
            reliability_score *= 0.9
            reliability_factors.append('minor_edge_cases')
        elif edge_case_info['severity'] == 'major':
            reliability_score *= 0.7
            reliability_factors.append('major_edge_cases')
        elif edge_case_info['severity'] == 'critical':
            reliability_score *= 0.4
            reliability_factors.append('critical_edge_cases')
        
        # Fit quality factors
        validation = result['validation']
        if not validation['fit_quality']['acceptable_chi2']:
            reliability_score *= 0.8
            reliability_factors.append('poor_chi_squared')
        
        if not validation['fit_quality']['good_r_squared']:
            reliability_score *= 0.9
            reliability_factors.append('low_r_squared')
        
        # Parameter significance
        best_fit = result['best_fit']
        if 'parameter_significance' in best_fit:
            param_sig = best_fit['parameter_significance']
            sig_fraction = param_sig['n_significant'] / len(param_sig['significant_parameters'])
            if sig_fraction < 0.8:
                reliability_score *= 0.8
                reliability_factors.append('poorly_constrained_parameters')
        
        # Model consensus
        selection = result['selection_summary']
        if selection['selection_confidence'] == 'low':
            reliability_score *= 0.7
            reliability_factors.append('low_selection_confidence')
        
        # Overall assessment
        if reliability_score >= 0.9:
            overall = 'high'
        elif reliability_score >= 0.7:
            overall = 'moderate'
        elif reliability_score >= 0.5:
            overall = 'low'
        else:
            overall = 'very_low'
        
        return {
            'reliability_score': reliability_score,
            'overall_reliability': overall,
            'reliability_factors': reliability_factors,
            'recommendations': self._generate_reliability_recommendations(reliability_factors, edge_case_info)
        }
    
    def _generate_reliability_recommendations(self, factors: List[str],
                                            edge_case_info: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on reliability assessment."""
        recommendations = []
        
        if 'critical_edge_cases' in factors:
            recommendations.extend(edge_case_info.get('recommendations', []))
            recommendations.append('Consider additional data collection')
        
        if 'major_edge_cases' in factors:
            recommendations.append('Results should be interpreted with caution')
        
        if 'poor_chi_squared' in factors:
            recommendations.append('Consider alternative models or additional components')
        
        if 'poorly_constrained_parameters' in factors:
            recommendations.append('Some parameters are poorly constrained - consider simpler model')
        
        if 'low_selection_confidence' in factors:
            recommendations.append('Model selection is ambiguous - consider multiple models')
        
        if not recommendations:
            recommendations.append('Results appear reliable')
        
        return recommendations
