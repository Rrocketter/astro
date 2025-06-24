from stingray import Lightcurve
from stingray.powerspectrum import AveragedPowerspectrum
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats

# Read file and generate power spectrum (same as data_load.py)
with fits.open('1130360105/xti/event_cl/ni1130360105_0mpu7_cl.evt.gz') as hdul:
    data = hdul[1].data
    times = data['TIME']

# Create light curve with larger time bins for efficiency
dt = 0.1  # Increased from 0.01 to reduce computation time
time_bins = np.arange(times.min(), times.max(), dt)
counts, _ = np.histogram(times, bins=time_bins)
time_bin_centers = (time_bins[:-1] + time_bins[1:]) / 2

print(f"Created light curve with {len(counts)} time bins")

lc = Lightcurve(time_bin_centers, counts, dt=dt)

# Generate power spectrum with smaller segment size for efficiency
ps = AveragedPowerspectrum(lc, segment_size=8.0)  # Reduced from 16.0

print(f"Power spectrum has {len(ps.freq)} frequency bins")

# Remove zero frequencies and powers for log transformation
valid_mask = (ps.freq > 0) & (ps.power > 0)
freq_valid = ps.freq[valid_mask]
power_valid = ps.power[valid_mask]

# Transform to log space
log_freq = np.log10(freq_valid)
log_power = np.log10(power_valid)

# Define polynomial models
def polynomial_2nd(log_f, a, b, c):
    """2nd order polynomial: log(P) = a*log(f)^2 + b*log(f) + c"""
    return a * log_f**2 + b * log_f + c

def polynomial_3rd(log_f, a, b, c, d):
    """3rd order polynomial: log(P) = a*log(f)^3 + b*log(f)^2 + c*log(f) + d"""
    return a * log_f**3 + b * log_f**2 + c * log_f + d

def linear_model(log_f, slope, intercept):
    """Linear model for comparison: log(P) = slope*log(f) + intercept"""
    return slope * log_f + intercept

# Fit different polynomial orders
models = {
    'Linear': (linear_model, 2),
    '2nd Order': (polynomial_2nd, 3),
    '3rd Order': (polynomial_3rd, 4)
}

results = {}

for model_name, (model_func, n_params) in models.items():
    try:
        popt, pcov = curve_fit(model_func, log_freq, log_power)
        param_errors = np.sqrt(np.diag(pcov))
        
        # Calculate predictions and R-squared
        log_power_pred = model_func(log_freq, *popt)
        ss_res = np.sum((log_power - log_power_pred) ** 2)
        ss_tot = np.sum((log_power - np.mean(log_power)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Calculate AIC and BIC for model comparison
        n = len(log_freq)
        mse = ss_res / n
        aic = n * np.log(mse) + 2 * n_params
        bic = n * np.log(mse) + n_params * np.log(n)
        
        results[model_name] = {
            'params': popt,
            'param_errors': param_errors,
            'predictions': log_power_pred,
            'r_squared': r_squared,
            'aic': aic,
            'bic': bic,
            'mse': mse,
            'model_func': model_func
        }
        
    except Exception as e:
        print(f"Failed to fit {model_name} model: {e}")
        continue

# Print comparison results
print("Polynomial Fitting Results (log-log space):")
print("=" * 60)
print(f"{'Model':<12} {'R²':<8} {'AIC':<10} {'BIC':<10} {'MSE':<12}")
print("-" * 60)

best_model = None
best_aic = float('inf')

for model_name, result in results.items():
    print(f"{model_name:<12} {result['r_squared']:<8.4f} {result['aic']:<10.2f} {result['bic']:<10.2f} {result['mse']:<12.6f}")
    if result['aic'] < best_aic:
        best_aic = result['aic']
        best_model = model_name

print(f"\nBest model (lowest AIC): {best_model}")
print("\nDetailed Results:")
print("=" * 60)

for model_name, result in results.items():
    print(f"\n{model_name} Model:")
    params = result['params']
    errors = result['param_errors']
    
    if model_name == 'Linear':
        print(f"  Slope: {params[0]:.6f} ± {errors[0]:.6f}")
        print(f"  Intercept: {params[1]:.6f} ± {errors[1]:.6f}")
        print(f"  Power-law model: P(f) = {10**params[1]:.3e} * f^{params[0]:.3f}")
    elif model_name == '2nd Order':
        print(f"  a (quadratic): {params[0]:.6f} ± {errors[0]:.6f}")
        print(f"  b (linear): {params[1]:.6f} ± {errors[1]:.6f}")
        print(f"  c (constant): {params[2]:.6f} ± {errors[2]:.6f}")
    elif model_name == '3rd Order':
        print(f"  a (cubic): {params[0]:.6f} ± {errors[0]:.6f}")
        print(f"  b (quadratic): {params[1]:.6f} ± {errors[1]:.6f}")
        print(f"  c (linear): {params[2]:.6f} ± {errors[2]:.6f}")
        print(f"  d (constant): {params[3]:.6f} ± {errors[3]:.6f}")
    
    print(f"  R²: {result['r_squared']:.6f}")

# Use the best model for further analysis
best_result = results[best_model]
best_params = best_result['params']
best_pred = best_result['predictions']

# Create fit for plotting (using best model)
freq_fit = np.logspace(np.log10(freq_valid.min()), np.log10(freq_valid.max()), 100)
log_freq_fit = np.log10(freq_fit)
log_power_fit = best_result['model_func'](log_freq_fit, *best_params)
power_fit = 10**log_power_fit

# Plot results
plt.figure(figsize=(15, 10))

# Subplot 1: Log-log plot with all fits
plt.subplot(2, 3, 1)
plt.loglog(freq_valid, power_valid, 'b.', alpha=0.6, markersize=2, label='Data')

colors = ['red', 'green', 'orange']
for i, (model_name, result) in enumerate(results.items()):
    freq_plot = np.logspace(np.log10(freq_valid.min()), np.log10(freq_valid.max()), 100)
    log_freq_plot = np.log10(freq_plot)
    log_power_plot = result['model_func'](log_freq_plot, *result['params'])
    power_plot = 10**log_power_plot
    
    style = '--' if model_name != best_model else '-'
    width = 1 if model_name != best_model else 2
    plt.loglog(freq_plot, power_plot, color=colors[i], linestyle=style, linewidth=width, 
               label=f'{model_name} (R²={result["r_squared"]:.3f})')

plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.title('PDS with Polynomial Fits')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 2: Residuals for best model
plt.subplot(2, 3, 2)
residuals = log_power - best_pred
plt.scatter(log_freq, residuals, alpha=0.6, s=1)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('log10(Frequency)')
plt.ylabel('Residuals (log10(Power))')
plt.title(f'Residuals from {best_model} Fit')
plt.grid(True, alpha=0.3)

# Subplot 3: Best fit in log space
plt.subplot(2, 3, 3)
plt.scatter(log_freq, log_power, alpha=0.6, s=1, label='Data')
plt.plot(log_freq, best_pred, 'r-', linewidth=2, label=f'{best_model} fit')
plt.xlabel('log10(Frequency)')
plt.ylabel('log10(Power)')
plt.title(f'{best_model} Fit in Log-Log Space')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 4: Frequency-weighted PDS with best fit
plt.subplot(2, 3, 4)
freq_weighted_power = freq_valid * power_valid
freq_weighted_fit = freq_fit * power_fit
plt.loglog(freq_valid, freq_weighted_power, 'b.', alpha=0.6, markersize=2, label='Data')
plt.loglog(freq_fit, freq_weighted_fit, 'r-', linewidth=2, label=f'{best_model} model')
plt.xlabel('Frequency (Hz)')
plt.ylabel(r'$\nu P(\nu)$')
plt.title('Frequency-weighted PDS')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 5: Model comparison (AIC values)
plt.subplot(2, 3, 5)
model_names = list(results.keys())
aic_values = [results[name]['aic'] for name in model_names]
colors_bar = ['red', 'green', 'orange']
bars = plt.bar(model_names, aic_values, color=colors_bar, alpha=0.7)
plt.ylabel('AIC Value')
plt.title('Model Comparison (Lower AIC = Better)')
plt.xticks(rotation=45)
# Highlight best model
best_idx = model_names.index(best_model)
bars[best_idx].set_edgecolor('black')
bars[best_idx].set_linewidth(3)
plt.grid(True, alpha=0.3)

# Subplot 6: R-squared comparison
plt.subplot(2, 3, 6)
r2_values = [results[name]['r_squared'] for name in model_names]
bars = plt.bar(model_names, r2_values, color=colors_bar, alpha=0.7)
plt.ylabel('R² Value')
plt.title('R² Comparison (Higher = Better)')
plt.xticks(rotation=45)
plt.ylim([min(r2_values) - 0.01, 1.0])
# Highlight best model
bars[best_idx].set_edgecolor('black')
bars[best_idx].set_linewidth(3)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('polynomial_continuum_fit.jpg', dpi=300, bbox_inches='tight')
plt.show()

# Save fit parameters to file
with open('polynomial_fit_results.txt', 'w') as f:
    f.write("Polynomial Continuum Fit Results\n")
    f.write("================================\n\n")
    f.write(f"Best model (lowest AIC): {best_model}\n\n")
    
    f.write("Model Comparison:\n")
    f.write("-" * 50 + "\n")
    f.write(f"{'Model':<12} {'R²':<8} {'AIC':<10} {'BIC':<10} {'MSE':<12}\n")
    f.write("-" * 50 + "\n")
    
    for model_name, result in results.items():
        f.write(f"{model_name:<12} {result['r_squared']:<8.4f} {result['aic']:<10.2f} {result['bic']:<10.2f} {result['mse']:<12.6f}\n")
    
    f.write("\nDetailed Results:\n")
    f.write("=" * 50 + "\n")
    
    for model_name, result in results.items():
        f.write(f"\n{model_name} Model:\n")
        params = result['params']
        errors = result['param_errors']
        
        if model_name == 'Linear':
            f.write(f"  Slope (power-law index): {params[0]:.6f} ± {errors[0]:.6f}\n")
            f.write(f"  Intercept: {params[1]:.6f} ± {errors[1]:.6f}\n")
            f.write(f"  Power-law model: P(f) = {10**params[1]:.6e} * f^{params[0]:.6f}\n")
        elif model_name == '2nd Order':
            f.write(f"  a (quadratic coeff): {params[0]:.6f} ± {errors[0]:.6f}\n")
            f.write(f"  b (linear coeff): {params[1]:.6f} ± {errors[1]:.6f}\n")
            f.write(f"  c (constant): {params[2]:.6f} ± {errors[2]:.6f}\n")
            f.write(f"  Model: log10(P) = {params[0]:.3e}*log10(f)² + {params[1]:.3f}*log10(f) + {params[2]:.3f}\n")
        elif model_name == '3rd Order':
            f.write(f"  a (cubic coeff): {params[0]:.6f} ± {errors[0]:.6f}\n")
            f.write(f"  b (quadratic coeff): {params[1]:.6f} ± {errors[1]:.6f}\n")
            f.write(f"  c (linear coeff): {params[2]:.6f} ± {errors[2]:.6f}\n")
            f.write(f"  d (constant): {params[3]:.6f} ± {errors[3]:.6f}\n")
            f.write(f"  Model: log10(P) = {params[0]:.3e}*log10(f)³ + {params[1]:.3e}*log10(f)² + {params[2]:.3f}*log10(f) + {params[3]:.3f}\n")
        
        f.write(f"  R²: {result['r_squared']:.6f}\n")
        f.write(f"  AIC: {result['aic']:.3f}\n")
        f.write(f"  BIC: {result['bic']:.3f}\n")
        f.write(f"  MSE: {result['mse']:.6e}\n")
    
    f.write(f"\nNumber of data points: {len(freq_valid)}\n")
    f.write(f"\nNote: AIC (Akaike Information Criterion) and BIC (Bayesian Information Criterion)\n")
    f.write(f"are used for model selection. Lower values indicate better models.\n")

print(f"\nResults saved to 'polynomial_fit_results.txt'")
print(f"Plot saved to 'polynomial_continuum_fit.jpg'")