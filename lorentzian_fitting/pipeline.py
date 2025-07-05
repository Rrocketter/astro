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
    
    # Selection strategy
    primary_criterion: str = "aic"  # "aic", "bic", or "aicc"
    require_statistical_significance: bool = True
    alpha_significance: float = 0.05


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
        
        # Determine component range to test
        test_components = self._determine_component_range(x, y)
        
        # Fit all models with robustness
        fit_results = self._fit_all_models(x, y, yerr, test_components)
        
        # Perform comprehensive model comparison
        comparison_results = self._perform_model_comparison(fit_results)
        
        # Apply intelligent selection logic
        selection_result = self._apply_selection_logic(comparison_results)
        
        # Validate final selection
        final_result = self._validate_and_finalize(selection_result, x, y, yerr)
        
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
    
    def _determine_component_range(self, x: np.ndarray, y: np.ndarray) -> List[int]:
        """Intelligently determine the range of components to test."""
        n_points = len(x)
        
        # Estimate maximum feasible components based on data
        # Rule of thumb: at least 3-5 points per parameter
        max_feasible = min(
            self.settings.max_components,
            (n_points - 1) // 4  # 4 parameters per component (3) + baseline (1)
        )
        
        # Always test baseline (0 components) and at least single component
        components = list(range(self.settings.min_components, max_feasible + 1))
        
        # Adaptive component testing based on data characteristics
        data_range = np.ptp(x)
        y_peaks = self._estimate_peak_count(x, y)
        
        # Adjust upper limit based on estimated peaks
        if y_peaks > 0:
            suggested_max = min(max_feasible, y_peaks + 1)
            components = list(range(self.settings.min_components, suggested_max + 1))
        
        return components
    
    def _estimate_peak_count(self, x: np.ndarray, y: np.ndarray) -> int:
        """Estimate the number of potential peaks in the data."""
        from scipy.signal import find_peaks
        
        try:
            # Remove baseline trend
            y_detrended = y - np.median(y)
            
            # Find peaks with reasonable prominence
            prominence = np.std(y_detrended) * 0.5
            peaks, _ = find_peaks(y_detrended, prominence=prominence)
            
            return min(len(peaks), self.settings.max_components)
        except Exception:
            # If peak finding fails, return conservative estimate
            return 2
    
    def _fit_all_models(self, x: np.ndarray, y: np.ndarray,
                       yerr: Optional[np.ndarray], 
                       test_components: List[int]) -> Dict[str, Any]:
        """Fit all models with robustness and retry logic."""
        fit_results = {}
        
        for n_comp in test_components:
            model_name = f"{n_comp}_component{'s' if n_comp != 1 else ''}"
            
            # Try fitting with multiple attempts for robustness
            for attempt in range(self.settings.n_retry_attempts):
                try:
                    if n_comp == 0:
                        result = self._fit_baseline_model(x, y, yerr)
                    else:
                        result = self._fit_lorentzian_model(x, y, yerr, n_comp, attempt)
                    
                    fit_results[model_name] = result
                    break  # Success, move to next model
                    
                except Exception as e:
                    if attempt == self.settings.n_retry_attempts - 1:
                        warnings.warn(f"Failed to fit {model_name} after {self.settings.n_retry_attempts} attempts: {e}")
                    else:
                        # Try with different initial conditions
                        continue
        
        if not fit_results:
            raise FittingError("All model fits failed")
        
        return fit_results
    
    def _fit_baseline_model(self, x: np.ndarray, y: np.ndarray,
                           yerr: Optional[np.ndarray]) -> Dict[str, Any]:
        """Fit baseline (constant) model."""
        baseline = np.mean(y)
        y_fit = np.full_like(y, baseline)
        residuals = y - y_fit
        
        if yerr is not None:
            chi2 = np.sum((residuals / yerr) ** 2)
        else:
            chi2 = np.sum(residuals ** 2) / np.var(residuals) if np.var(residuals) > 0 else 0
        
        n_params = 1
        dof = len(y) - n_params
        
        # Import metrics functions
        from .metrics import calculate_aic, calculate_bic, calculate_goodness_of_fit
        
        result = {
            'params': np.array([baseline]),
            'param_errors': np.array([np.std(residuals) / np.sqrt(len(y))]),
            'fit_info': {
                'fitted_curve': y_fit,
                'residuals': residuals,
                'chi_squared': chi2,
                'reduced_chi_squared': chi2 / dof if dof > 0 else np.inf,
                'degrees_of_freedom': dof,
                'aic': calculate_aic(chi2, n_params, len(y)),
                'bic': calculate_bic(chi2, n_params, len(y)),
                'n_params': n_params,
                'n_points': len(y),
                'n_components': 0
            }
        }
        
        # Add goodness of fit metrics
        gof_metrics = calculate_goodness_of_fit(y, y_fit, yerr)
        result['fit_info'].update(gof_metrics)
        
        return result
    
    def _fit_lorentzian_model(self, x: np.ndarray, y: np.ndarray,
                             yerr: Optional[np.ndarray], n_comp: int,
                             attempt: int) -> Dict[str, Any]:
        """Fit Lorentzian model with specified number of components."""
        
        # Generate different initial guesses for retry attempts
        initial_guess = None
        if attempt > 0:
            initial_guess = self._generate_perturbed_guess(x, y, n_comp, attempt)
        
        if n_comp == 1:
            params, param_errors, fit_info = self.fitter.fit_single(
                x, y, yerr, initial_guess
            )
        else:
            params, param_errors, fit_info = self.fitter.fit_multiple(
                x, y, n_comp, yerr, initial_guess
            )
        
        return {
            'params': params,
            'param_errors': param_errors,
            'fit_info': fit_info
        }
    
    def _generate_perturbed_guess(self, x: np.ndarray, y: np.ndarray,
                                 n_comp: int, attempt: int) -> List[float]:
        """Generate perturbed initial guess for retry attempts."""
        from .models import generate_initial_guess
        
        # Get base guess
        base_guess = generate_initial_guess(x, y, n_comp)
        
        # Add random perturbations scaled by attempt number
        perturbation_factor = 0.2 * attempt
        np.random.seed(42 + attempt)  # Reproducible perturbations
        
        perturbed_guess = []
        for i, param in enumerate(base_guess):
            if i % 3 == 0:  # Amplitude
                perturbation = np.random.normal(0, abs(param) * perturbation_factor)
            elif i % 3 == 1:  # Center
                x_range = np.ptp(x)
                perturbation = np.random.normal(0, x_range * perturbation_factor)
            elif i % 3 == 2:  # Width
                perturbation = np.random.normal(0, abs(param) * perturbation_factor)
            else:  # Baseline
                y_range = np.ptp(y)
                perturbation = np.random.normal(0, y_range * perturbation_factor)
            
            perturbed_guess.append(param + perturbation)
        
        return perturbed_guess
    
    def _perform_model_comparison(self, fit_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive model comparison."""
        # Extract fit statistics for comparison
        statistics = {}
        for model_name, result in fit_results.items():
            statistics[model_name] = result['fit_info']
        
        # Use existing model selection framework
        from .metrics import model_selection_criteria
        selection_results = model_selection_criteria(statistics)
        
        # Add custom pipeline-specific analyses
        pipeline_analysis = self._perform_pipeline_analysis(fit_results, selection_results)
        
        return {
            'fit_results': fit_results,
            'model_selection': selection_results,
            'pipeline_analysis': pipeline_analysis
        }
    
    def _perform_pipeline_analysis(self, fit_results: Dict[str, Any],
                                  selection_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform pipeline-specific analysis."""
        analysis = {}
        
        # Extract information criteria
        ic_values = selection_results['summary']
        
        # Apply threshold-based decision making
        analysis['threshold_decisions'] = self._apply_threshold_decisions(ic_values)
        
        # Assess model stability
        analysis['stability_assessment'] = self._assess_model_stability(fit_results)
        
        # Check for overfitting indicators
        analysis['overfitting_check'] = self._check_overfitting(fit_results)
        
        return analysis
    
    def _apply_threshold_decisions(self, ic_values: Dict[str, Dict]) -> Dict[str, Any]:
        """Apply threshold-based decisions for model selection."""
        decisions = {}
        
        # Get delta values
        delta_aic = ic_values.get('delta_aic', {})
        delta_bic = ic_values.get('delta_bic', {})
        
        # Find models within thresholds
        aic_candidates = [model for model, delta in delta_aic.items() 
                         if delta < self.settings.delta_aic_threshold]
        bic_candidates = [model for model, delta in delta_bic.items() 
                         if delta < self.settings.delta_bic_threshold]
        
        decisions['aic_candidates'] = aic_candidates
        decisions['bic_candidates'] = bic_candidates
        decisions['consensus_candidates'] = list(set(aic_candidates) & set(bic_candidates))
        
        # Apply evidence strength categories
        for model, delta in delta_aic.items():
            if delta < 2:
                strength = "substantial"
            elif delta < self.settings.moderate_evidence_threshold:
                strength = "considerable"
            elif delta < self.settings.strong_evidence_threshold:
                strength = "strong"
            elif delta < self.settings.delta_aic_threshold:
                strength = "very_strong"
            else:
                strength = "decisive"
            
            decisions[f"{model}_evidence_against"] = strength
        
        return decisions
    
    def _assess_model_stability(self, fit_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the stability of fitted models."""
        stability = {}
        
        for model_name, result in fit_results.items():
            params = result['params']
            param_errors = result['param_errors']
            
            # Calculate relative uncertainties
            rel_uncertainties = []
            for i, (param, error) in enumerate(zip(params, param_errors)):
                if abs(param) > 1e-10:
                    rel_uncertainties.append(abs(error / param))
                else:
                    rel_uncertainties.append(np.inf)
            
            # Assess stability metrics
            max_rel_uncertainty = max(rel_uncertainties)
            mean_rel_uncertainty = np.mean([u for u in rel_uncertainties if np.isfinite(u)])
            
            stability[model_name] = {
                'max_relative_uncertainty': max_rel_uncertainty,
                'mean_relative_uncertainty': mean_rel_uncertainty,
                'is_stable': max_rel_uncertainty < 1.0,  # Parameters known to better than 100%
                'is_well_constrained': max_rel_uncertainty < 0.5  # Parameters known to better than 50%
            }
        
        return stability
    
    def _check_overfitting(self, fit_results: Dict[str, Any]) -> Dict[str, Any]:
        """Check for indicators of overfitting."""
        overfitting = {}
        
        # Sort models by complexity
        models_by_complexity = sorted(fit_results.keys(), 
                                    key=lambda x: fit_results[x]['fit_info'].get('n_params', 0))
        
        # Check for suspiciously good fits
        for model_name, result in fit_results.items():
            fit_info = result['fit_info']
            reduced_chi2 = fit_info.get('reduced_chi_squared', np.inf)
            
            # Flags for potential overfitting
            overfitting[model_name] = {
                'extremely_low_chi2': reduced_chi2 < 0.1,
                'perfect_fit_warning': reduced_chi2 < 0.01,
                'high_parameter_uncertainty': False  # Will be set based on stability
            }
        
        return overfitting
    
    def _apply_selection_logic(self, comparison_results: Dict[str, Any]) -> Dict[str, Any]:
        """Apply intelligent selection logic based on all criteria."""
        selection = comparison_results['model_selection']
        pipeline_analysis = comparison_results['pipeline_analysis']
        
        # Get primary criterion
        primary_ic = self.settings.primary_criterion
        best_model = selection[f'best_{primary_ic}_model']
        
        # Check if best model meets quality criteria
        threshold_decisions = pipeline_analysis['threshold_decisions']
        stability = pipeline_analysis['stability_assessment']
        
        # Apply selection logic
        final_selection = self._select_final_model(
            best_model, threshold_decisions, stability, comparison_results
        )
        
        return {
            'selected_model': final_selection['model'],
            'selection_confidence': final_selection['confidence'],
            'selection_rationale': final_selection['rationale'],
            'alternative_models': final_selection['alternatives'],
            'comparison_results': comparison_results
        }
    
    def _select_final_model(self, best_model: str, threshold_decisions: Dict,
                           stability: Dict, comparison_results: Dict) -> Dict[str, Any]:
        """Select the final model using comprehensive criteria."""
        
        # Start with the best model according to primary criterion
        candidate = best_model
        confidence = "high"
        rationale = [f"Best {self.settings.primary_criterion.upper()} model"]
        
        # Check stability
        if not stability[candidate]['is_stable']:
            # Look for stable alternatives
            stable_candidates = [model for model, stab in stability.items() 
                               if stab['is_stable']]
            
            if stable_candidates:
                # Find the simplest stable model within thresholds
                consensus = threshold_decisions['consensus_candidates']
                stable_consensus = list(set(stable_candidates) & set(consensus))
                
                if stable_consensus:
                    # Choose simplest stable model in consensus
                    candidate = min(stable_consensus, 
                                  key=lambda x: comparison_results['fit_results'][x]['fit_info']['n_params'])
                    confidence = "moderate"
                    rationale.append("Selected for stability")
                else:
                    confidence = "low"
                    rationale.append("Best model unstable, no stable consensus")
        
        # Find alternative models for reporting
        alternatives = []
        ic_values = comparison_results['model_selection']['summary']
        delta_aic = ic_values.get('delta_aic', {})
        
        for model, delta in delta_aic.items():
            if model != candidate and delta < self.settings.moderate_evidence_threshold:
                alternatives.append({
                    'model': model,
                    'delta_aic': delta,
                    'evidence_strength': threshold_decisions[f"{model}_evidence_against"]
                })
        
        return {
            'model': candidate,
            'confidence': confidence,
            'rationale': rationale,
            'alternatives': alternatives
        }
    
    def _validate_and_finalize(self, selection_result: Dict[str, Any],
                              x: np.ndarray, y: np.ndarray,
                              yerr: Optional[np.ndarray]) -> Dict[str, Any]:
        """Validate and finalize the selection result."""
        
        selected_model = selection_result['selected_model']
        comparison_results = selection_result['comparison_results']
        
        # Get the final fit result
        final_fit = comparison_results['fit_results'][selected_model]
        
        # Perform final validation checks
        validation = self._final_validation(final_fit, x, y, yerr)
        
        # Compile comprehensive result
        result = {
            'best_fit': final_fit,
            'model_name': selected_model,
            'n_components': final_fit['fit_info'].get('n_components', 0),
            'selection_summary': selection_result,
            'all_models': comparison_results,
            'validation': validation,
            'pipeline_settings': self.settings
        }
        
        return result
    
    def _final_validation(self, final_fit: Dict[str, Any], x: np.ndarray,
                         y: np.ndarray, yerr: Optional[np.ndarray]) -> Dict[str, Any]:
        """Perform final validation of the selected fit."""
        
        fit_info = final_fit['fit_info']
        residuals = fit_info['residuals']
        
        validation = {
            'fit_quality': {
                'reduced_chi_squared': fit_info['reduced_chi_squared'],
                'acceptable_chi2': 0.5 <= fit_info['reduced_chi_squared'] <= 2.0,
                'r_squared': fit_info['r_squared'],
                'good_r_squared': fit_info['r_squared'] > 0.8
            },
            'residual_analysis': {
                'mean_residual': np.mean(residuals),
                'std_residual': np.std(residuals),
                'residuals_centered': abs(np.mean(residuals)) < 2 * np.std(residuals) / np.sqrt(len(residuals))
            },
            'parameter_quality': {
                'all_finite': np.all(np.isfinite(final_fit['params'])),
                'positive_widths': True  # Will check based on model type
            }
        }
        
        # Overall assessment
        validation['overall_quality'] = (
            validation['fit_quality']['acceptable_chi2'] and
            validation['fit_quality']['good_r_squared'] and
            validation['residual_analysis']['residuals_centered'] and
            validation['parameter_quality']['all_finite']
        )
        
        return validation
