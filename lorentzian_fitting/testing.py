"""
plotting and testing script for the Lorentzian fitting package.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append('/Users/rahulgupta/Developer/astro')

from lorentzian_fitting.models import single_lorentzian, multiple_lorentzian
from lorentzian_fitting.fitting import LorentzianFitter
from lorentzian_fitting.comparison import ModelComparison, automated_model_selection
from lorentzian_fitting.metrics import model_selection_criteria
from lorentzian_fitting.pipeline import AutomatedPipeline, PipelineSettings

def create_test_data():
    """Create synthetic test data with known parameters."""
    np.random.seed(42)  # For reproducible results
    
    # Test case 1: Single Lorentzian
    x1 = np.linspace(-10, 10, 100)
    y1_true = single_lorentzian(x1, amplitude=5.0, center=0.0, width=2.0, baseline=1.0)
    noise1 = np.random.normal(0, 0.2, len(x1))
    y1 = y1_true + noise1
    yerr1 = np.full_like(y1, 0.2)
    
    # Test case 2: Two Lorentzian components
    x2 = np.linspace(-15, 15, 150)
    y2_comp1 = single_lorentzian(x2, 3.0, -5.0, 2.0, 0.0)
    y2_comp2 = single_lorentzian(x2, 4.0, 5.0, 1.5, 0.0)
    y2_true = y2_comp1 + y2_comp2 + 0.5  # baseline
    noise2 = np.random.normal(0, 0.15, len(x2))
    y2 = y2_true + noise2
    yerr2 = np.full_like(y2, 0.15)
    
    return (x1, y1, yerr1), (x2, y2, yerr2)

def test_step1_fitting():
    """Test Step 1: Basic Lorentzian fitting."""
    print("=" * 60)
    print("TESTING STEP 1: LORENTZIAN FITTING")
    print("=" * 60)
    
    (x1, y1, yerr1), (x2, y2, yerr2) = create_test_data()
    fitter = LorentzianFitter()
    
    # Test single component fitting
    print("\n1. Single Lorentzian Fitting:")
    print("-" * 30)
    
    try:
        params1, param_errors1, fit_info1 = fitter.fit_single(x1, y1, yerr1)
        
        print(f"True parameters: [5.0, 0.0, 2.0, 1.0]")
        print(f"Fitted parameters:")
        print(f"  Amplitude: {params1[0]:.3f} ± {param_errors1[0]:.3f}")
        print(f"  Center: {params1[1]:.3f} ± {param_errors1[1]:.3f}")
        print(f"  Width: {params1[2]:.3f} ± {param_errors1[2]:.3f}")
        print(f"  Baseline: {params1[3]:.3f} ± {param_errors1[3]:.3f}")
        print(f"Reduced χ²: {fit_info1['reduced_chi_squared']:.3f}")
        print(f"R²: {fit_info1['r_squared']:.3f}")
        
        # Plot results
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.errorbar(x1, y1, yerr=yerr1, fmt='o', alpha=0.7, label='Data')
        plt.plot(x1, fit_info1['fitted_curve'], 'r-', linewidth=2, label='Fitted curve')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Single Lorentzian Fit')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Residuals plot
        plt.subplot(1, 2, 2)
        plt.errorbar(x1, fit_info1['residuals'], yerr=yerr1, fmt='o', alpha=0.7)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        plt.xlabel('x')
        plt.ylabel('Residuals')
        plt.title('Residuals')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Single fit failed: {e}")
        return False
    
    # Test multiple component fitting
    print("\n2. Multiple Lorentzian Fitting:")
    print("-" * 30)
    
    try:
        params2, param_errors2, fit_info2 = fitter.fit_multiple(x2, y2, n_components=2, yerr=yerr2)
        
        print(f"True parameters: [3.0, -5.0, 2.0, 4.0, 5.0, 1.5, 0.5]")
        print(f"Fitted parameters:")
        for i in range(2):
            idx = i * 3
            print(f"Component {i+1}:")
            print(f"  Amplitude: {params2[idx]:.3f} ± {param_errors2[idx]:.3f}")
            print(f"  Center: {params2[idx+1]:.3f} ± {param_errors2[idx+1]:.3f}")
            print(f"  Width: {params2[idx+2]:.3f} ± {param_errors2[idx+2]:.3f}")
        print(f"Baseline: {params2[-1]:.3f} ± {param_errors2[-1]:.3f}")
        print(f"Reduced χ²: {fit_info2['reduced_chi_squared']:.3f}")
        print(f"R²: {fit_info2['r_squared']:.3f}")
        
        # Plot results
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.errorbar(x2, y2, yerr=yerr2, fmt='o', alpha=0.7, label='Data')
        plt.plot(x2, fit_info2['fitted_curve'], 'r-', linewidth=2, label='Total fit')
        
        # Plot individual components
        for i in range(2):
            idx = i * 3
            y_component = single_lorentzian(x2, params2[idx], params2[idx+1], params2[idx+2], 0.0)
            plt.plot(x2, y_component + params2[-1], '--', alpha=0.7, label=f'Component {i+1}')
        
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Multiple Lorentzian Fit')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Residuals plot
        plt.subplot(1, 2, 2)
        plt.errorbar(x2, fit_info2['residuals'], yerr=yerr2, fmt='o', alpha=0.7)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        plt.xlabel('x')
        plt.ylabel('Residuals')
        plt.title('Residuals')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Multiple fit failed: {e}")
        return False
    
    return True

def test_step2_metrics():
    """Test Step 2: Fit quality metrics and model comparison."""
    print("\n" + "=" * 60)
    print("TESTING STEP 2: FIT QUALITY METRICS")
    print("=" * 60)
    
    (x1, y1, yerr1), (x2, y2, yerr2) = create_test_data()
    
    # Test model comparison on single Lorentzian data
    print("\n1. Model Comparison (Single Lorentzian Data):")
    print("-" * 45)
    
    try:
        comparator = ModelComparison(max_components=3)
        results1 = comparator.compare_models(x1, y1, yerr1, test_components=[0, 1, 2])
        
        # Print model selection results
        selection = results1['model_selection']
        print(f"Best AIC model: {selection['best_aic_model']}")
        print(f"Best BIC model: {selection['best_bic_model']}")
        
        print("\nModel Statistics:")
        for model, stats in results1['statistics'].items():
            print(f"{model:15} | AIC: {stats['aic']:6.1f} | BIC: {stats['bic']:6.1f} | χ²: {stats['reduced_chi_squared']:5.2f}")
        
        print("\nΔAIC values:")
        for model, delta in selection['summary']['delta_aic'].items():
            print(f"{model:15} | ΔAIC: {delta:6.1f}")
        
        print("\nRecommendations:")
        for key, recommendation in results1['recommendations'].items():
            print(f"{key}: {recommendation}")
        
    except Exception as e:
        print(f"Model comparison failed: {e}")
        return False
    
    # Test automated model selection
    print("\n2. Automated Model Selection (Two Lorentzian Data):")
    print("-" * 50)
    
    try:
        best_n, results2 = automated_model_selection(x2, y2, yerr2, max_components=4)
        
        print(f"Automatically selected: {best_n} components")
        print(f"True number of components: 2")
        
        # Create comparison plot
        plt.figure(figsize=(15, 5))
        
        # Plot AIC comparison
        plt.subplot(1, 3, 1)
        models = list(results2['model_selection']['summary']['aic_values'].keys())
        aic_values = list(results2['model_selection']['summary']['aic_values'].values())
        plt.bar(range(len(models)), aic_values)
        plt.xlabel('Model')
        plt.ylabel('AIC')
        plt.title('AIC Comparison')
        plt.xticks(range(len(models)), [m.replace('_', '\n') for m in models], rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Plot BIC comparison
        plt.subplot(1, 3, 2)
        bic_values = list(results2['model_selection']['summary']['bic_values'].values())
        plt.bar(range(len(models)), bic_values)
        plt.xlabel('Model')
        plt.ylabel('BIC')
        plt.title('BIC Comparison')
        plt.xticks(range(len(models)), [m.replace('_', '\n') for m in models], rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Plot Akaike weights
        plt.subplot(1, 3, 3)
        weights = list(results2['model_selection']['summary']['akaike_weights'].values())
        plt.bar(range(len(models)), weights)
        plt.xlabel('Model')
        plt.ylabel('Akaike Weight')
        plt.title('Model Weights')
        plt.xticks(range(len(models)), [m.replace('_', '\n') for m in models], rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print statistical tests
        print("\nStatistical Tests:")
        for test_name, test_result in results2['statistical_tests'].items():
            lr = test_result['likelihood_ratio']
            f_test = test_result['f_test']
            print(f"{test_name}:")
            print(f"  LR test: stat={lr['statistic']:.2f}, p={lr['p_value']:.4f}, sig={lr['significant']}")
            print(f"  F test:  stat={f_test['statistic']:.2f}, p={f_test['p_value']:.4f}, sig={f_test['significant']}")
        
    except Exception as e:
        print(f"Automated selection failed: {e}")
        return False
    
    return True

def test_step3_pipeline():
    """Test Step 3: Automated fitting pipeline."""
    print("\n" + "=" * 60)
    print("TESTING STEP 3: AUTOMATED PIPELINE")
    print("=" * 60)
    
    (x1, y1, yerr1), (x2, y2, yerr2) = create_test_data()
    
    # Test automated pipeline
    print("\n1. Automated Pipeline (Single Lorentzian Data):")
    print("-" * 45)
    
    try:
        # Configure pipeline settings
        settings = PipelineSettings(
            max_components=3,
            delta_aic_threshold=20.0,
            n_retry_attempts=2
        )
        
        pipeline = AutomatedPipeline(settings)
        result1 = pipeline.fit_with_model_selection(x1, y1, yerr1)
        
        print(f"Selected model: {result1['model_name']}")
        print(f"Number of components: {result1['n_components']}")
        print(f"Selection confidence: {result1['selection_summary']['selection_confidence']}")
        print(f"Selection rationale: {', '.join(result1['selection_summary']['selection_rationale'])}")
        
        # Print fit quality
        fit_quality = result1['validation']['fit_quality']
        print(f"\nFit Quality:")
        print(f"  Reduced χ²: {fit_quality['reduced_chi_squared']:.3f}")
        print(f"  R²: {fit_quality['r_squared']:.3f}")
        print(f"  Overall quality: {'Good' if result1['validation']['overall_quality'] else 'Poor'}")
        
        # Show alternatives
        alternatives = result1['selection_summary']['alternative_models']
        if alternatives:
            print(f"\nAlternative models:")
            for alt in alternatives:
                print(f"  {alt['model']}: ΔAIC = {alt['delta_aic']:.1f} ({alt['evidence_strength']} evidence against)")
        
    except Exception as e:
        print(f"Automated pipeline failed: {e}")
        return False
    
    # Test on more complex data
    print("\n2. Automated Pipeline (Two Lorentzian Data):")
    print("-" * 45)
    
    try:
        result2 = pipeline.fit_with_model_selection(x2, y2, yerr2)
        
        print(f"Selected model: {result2['model_name']}")
        print(f"Number of components: {result2['n_components']}")
        print(f"True number of components: 2")
        print(f"Selection confidence: {result2['selection_summary']['selection_confidence']}")
        
        # Create pipeline comparison plot
        plt.figure(figsize=(15, 10))
        
        # Plot the best fit
        plt.subplot(2, 2, 1)
        best_fit = result2['best_fit']
        plt.errorbar(x2, y2, yerr=yerr2, fmt='o', alpha=0.7, label='Data')
        plt.plot(x2, best_fit['fit_info']['fitted_curve'], 'r-', linewidth=2, label='Best fit')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Best Fit: {result2["model_name"]}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot model comparison
        plt.subplot(2, 2, 2)
        all_models = result2['all_models']['model_selection']['summary']
        models = list(all_models['aic_values'].keys())
        aic_values = list(all_models['aic_values'].values())
        
        colors = ['green' if model == result2['model_name'] else 'lightblue' for model in models]
        plt.bar(range(len(models)), aic_values, color=colors)
        plt.xlabel('Model')
        plt.ylabel('AIC')
        plt.title('Model Comparison (Green = Selected)')
        plt.xticks(range(len(models)), [m.replace('_', '\n') for m in models], rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Plot threshold analysis
        plt.subplot(2, 2, 3)
        delta_aic = list(all_models['delta_aic'].values())
        plt.bar(range(len(models)), delta_aic, color=colors)
        plt.axhline(y=4, color='orange', linestyle='--', label='Moderate evidence')
        plt.axhline(y=10, color='red', linestyle='--', label='Strong evidence')
        plt.axhline(y=20, color='darkred', linestyle='--', label='Decisive evidence')
        plt.xlabel('Model')
        plt.ylabel('ΔAIC')
        plt.title('Evidence Thresholds')
        plt.xticks(range(len(models)), [m.replace('_', '\n') for m in models], rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot stability assessment
        plt.subplot(2, 2, 4)
        stability = result2['all_models']['pipeline_analysis']['stability_assessment']
        stability_scores = [stability[model]['mean_relative_uncertainty'] for model in models]
        plt.bar(range(len(models)), stability_scores, color=colors)
        plt.axhline(y=0.5, color='orange', linestyle='--', label='Well constrained')
        plt.axhline(y=1.0, color='red', linestyle='--', label='Stable threshold')
        plt.xlabel('Model')
        plt.ylabel('Mean Relative Uncertainty')
        plt.title('Parameter Stability')
        plt.xticks(range(len(models)), [m.replace('_', '\n') for m in models], rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Complex pipeline test failed: {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("Testing Lorentzian Fitting Package - Steps 1, 2 & 3")
    print("=" * 60)
    
    success = True
    
    # Test Step 1
    if not test_step1_fitting():
        success = False
        print("❌ Step 1 failed!")
    else:
        print("✅ Step 1 passed!")
    
    # Test Step 2
    if not test_step2_metrics():
        success = False
        print("❌ Step 2 failed!")
    else:
        print("✅ Step 2 passed!")
    
    # Test Step 3
    if not test_step3_pipeline():
        success = False
        print("❌ Step 3 failed!")
    else:
        print("✅ Step 3 passed!")
    
    if success:
        print("\n🎉 All tests passed! Pipeline ready for use.")
    else:
        print("\n⚠️  Some tests failed.")
    
    return success

if __name__ == "__main__":
    main()
