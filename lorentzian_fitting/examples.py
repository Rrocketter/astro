"""
Usage Examples

This module provides practical examples of how to use the Lorentzian fitting
package for common astronomical data analysis tasks.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

from .fitting import LorentzianFitter
from .models import single_lorentzian
from .errors import FittingError

def generate_sample_data(noise_level: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate sample data with noise for testing."""
    x = np.linspace(-10, 10, 100)
    
    # True parameters: amplitude=5, center=0, width=2, baseline=1
    y_true = single_lorentzian(x, 5.0, 0.0, 2.0, 1.0)
    
    # Add noise
    noise = np.random.normal(0, noise_level, len(x))
    y_data = y_true + noise
    yerr = np.full_like(y_data, noise_level)
    
    return x, y_data, yerr

def example_single_fit():
    """Example of fitting a single Lorentzian."""
    print("=== Single Lorentzian Fitting Example ===")
    
    # Generate sample data
    x, y, yerr = generate_sample_data(noise_level=0.2)
    
    # Create fitter
    fitter = LorentzianFitter()
    
    try:
        # Fit the data
        params, param_errors, fit_info = fitter.fit_single(x, y, yerr)
        
        print(f"Fitted parameters:")
        print(f"  Amplitude: {params[0]:.3f} ± {param_errors[0]:.3f}")
        print(f"  Center: {params[1]:.3f} ± {param_errors[1]:.3f}")
        print(f"  Width: {params[2]:.3f} ± {param_errors[2]:.3f}")
        print(f"  Baseline: {params[3]:.3f} ± {param_errors[3]:.3f}")
        print(f"Reduced χ²: {fit_info['reduced_chi_squared']:.3f}")
        
        # Plot results
        plt.figure(figsize=(10, 6))
        plt.errorbar(x, y, yerr=yerr, fmt='o', alpha=0.7, label='Data')
        plt.plot(x, fit_info['fitted_curve'], 'r-', label='Fitted curve')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Single Lorentzian Fit Example')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
    except FittingError as e:
        print(f"Fitting failed: {e.message}")

def example_multiple_fit():
    """Example of fitting multiple Lorentzian components."""
    print("\n=== Multiple Lorentzian Fitting Example ===")
    
    # Generate data with two components
    x = np.linspace(-15, 15, 150)
    y1 = single_lorentzian(x, 3.0, -5.0, 2.0, 0.0)
    y2 = single_lorentzian(x, 4.0, 5.0, 1.5, 0.0)
    y_true = y1 + y2 + 0.5  # baseline = 0.5
    
    # Add noise
    noise = np.random.normal(0, 0.15, len(x))
    y_data = y_true + noise
    yerr = np.full_like(y_data, 0.15)
    
    # Create fitter
    fitter = LorentzianFitter()
    
    try:
        # Fit with 2 components
        params, param_errors, fit_info = fitter.fit_multiple(x, y_data, n_components=2, yerr=yerr)
        
        print(f"Fitted parameters for 2 components:")
        for i in range(2):
            idx = i * 3
            print(f"Component {i+1}:")
            print(f"  Amplitude: {params[idx]:.3f} ± {param_errors[idx]:.3f}")
            print(f"  Center: {params[idx+1]:.3f} ± {param_errors[idx+1]:.3f}")
            print(f"  Width: {params[idx+2]:.3f} ± {param_errors[idx+2]:.3f}")
        print(f"Baseline: {params[-1]:.3f} ± {param_errors[-1]:.3f}")
        print(f"Reduced χ²: {fit_info['reduced_chi_squared']:.3f}")
        
        # Plot results
        plt.figure(figsize=(12, 6))
        plt.errorbar(x, y_data, yerr=yerr, fmt='o', alpha=0.7, label='Data')
        plt.plot(x, fit_info['fitted_curve'], 'r-', linewidth=2, label='Total fit')
        
        # Plot individual components
        for i in range(2):
            idx = i * 3
            y_component = single_lorentzian(x, params[idx], params[idx+1], params[idx+2], 0.0)
            plt.plot(x, y_component + params[-1], '--', alpha=0.7, label=f'Component {i+1}')
        
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Multiple Lorentzian Fit Example')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
    except FittingError as e:
        print(f"Multi-component fitting failed: {e.message}")

def run_all_examples():
    """Run all examples."""
    np.random.seed(42)  # For reproducible results
    example_single_fit()
    example_multiple_fit()

if __name__ == "__main__":
    run_all_examples()