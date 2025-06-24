from stingray import Lightcurve
from stingray.powerspectrum import AveragedPowerspectrum
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Read file and generate power spectrum (same as polynomial fitting)
with fits.open('1130360105/xti/event_cl/ni1130360105_0mpu7_cl.evt.gz') as hdul:
    data = hdul[1].data
    times = data['TIME']

# Create light curve with same parameters as polynomial fitting
dt = 0.1
time_bins = np.arange(times.min(), times.max(), dt)
counts, _ = np.histogram(times, bins=time_bins)
time_bin_centers = (time_bins[:-1] + time_bins[1:]) / 2

print(f"Created light curve with {len(counts)} time bins")

lc = Lightcurve(time_bin_centers, counts, dt=dt)

# Generate power spectrum
ps = AveragedPowerspectrum(lc, segment_size=8.0)

print(f"Power spectrum has {len(ps.freq)} frequency bins")

# Remove zero frequencies and powers for log transformation
valid_mask = (ps.freq > 0) & (ps.power > 0)
freq_valid = ps.freq[valid_mask]
power_valid = ps.power[valid_mask]

# Transform to log space
log_freq = np.log10(freq_valid)
log_power = np.log10(power_valid)

# Define the best continuum model from polynomial fitting (3rd order)
def polynomial_3rd(log_f, a, b, c, d):
    """3rd order polynomial: log(P) = a*log(f)^3 + b*log(f)^2 + c*log(f) + d"""
    return a * log_f**3 + b * log_f**2 + c * log_f + d

# Fit the 3rd order polynomial to get the continuum
popt_continuum, _ = curve_fit(polynomial_3rd, log_freq, log_power)
log_power_continuum = polynomial_3rd(log_freq, *popt_continuum)

# Calculate residuals (data - continuum)
residuals = log_power - log_power_continuum

# Convert back to linear space for peak finding
power_continuum = 10**log_power_continuum
residuals_linear = power_valid - power_continuum

print(f"Continuum model parameters: {popt_continuum}")
print(f"RMS of residuals: {np.std(residuals):.4f} (log space)")
print(f"RMS of residuals: {np.std(residuals_linear):.6f} (linear space)")

# Define Gaussian and Lorentzian peak models
def gaussian(freq, amplitude, center, width):
    """Gaussian peak: A * exp(-0.5 * ((f - f0) / sigma)^2)"""
    return amplitude * np.exp(-0.5 * ((freq - center) / width)**2)

def lorentzian(freq, amplitude, center, width):
    """Lorentzian peak: A * (gamma^2 / ((f - f0)^2 + gamma^2))"""
    return amplitude * (width**2 / ((freq - center)**2 + width**2))

def double_gaussian(freq, amp1, center1, width1, amp2, center2, width2):
    """Sum of two Gaussian peaks"""
    return gaussian(freq, amp1, center1, width1) + gaussian(freq, amp2, center2, width2)

def triple_gaussian(freq, amp1, center1, width1, amp2, center2, width2, amp3, center3, width3):
    """Sum of three Gaussian peaks"""
    return (gaussian(freq, amp1, center1, width1) + 
            gaussian(freq, amp2, center2, width2) + 
            gaussian(freq, amp3, center3, width3))

# Find peaks in the residuals
# Use both log and linear space for peak detection
peaks_log, properties_log = find_peaks(residuals, height=0.05, distance=2)
peaks_linear, properties_linear = find_peaks(residuals_linear, height=np.std(residuals_linear), distance=2)

print(f"\nPeak Detection Results:")
print(f"Peaks found in log residuals: {len(peaks_log)} at indices {peaks_log}")
print(f"Peaks found in linear residuals: {len(peaks_linear)} at indices {peaks_linear}")

if len(peaks_log) > 0:
    peak_freqs_log = freq_valid[peaks_log]
    peak_powers_log = power_valid[peaks_log]
    print(f"Peak frequencies (log method): {peak_freqs_log}")
    print(f"Peak powers (log method): {peak_powers_log}")

if len(peaks_linear) > 0:
    peak_freqs_linear = freq_valid[peaks_linear]
    peak_powers_linear = power_valid[peaks_linear]
    print(f"Peak frequencies (linear method): {peak_freqs_linear}")
    print(f"Peak powers (linear method): {peak_powers_linear}")

# Combine and deduplicate peaks
all_peak_indices = np.unique(np.concatenate([peaks_log, peaks_linear]) if len(peaks_log) > 0 and len(peaks_linear) > 0 
                            else peaks_log if len(peaks_log) > 0 else peaks_linear if len(peaks_linear) > 0 else [])

if len(all_peak_indices) > 0:
    peak_frequencies = freq_valid[all_peak_indices]
    peak_residuals = residuals_linear[all_peak_indices]
    
    print(f"\nCombined peak analysis:")
    print(f"Number of significant peaks: {len(peak_frequencies)}")
    print(f"Peak frequencies: {peak_frequencies}")
    print(f"Peak residual amplitudes: {peak_residuals}")
    
    # Try to fit peaks with different models
    peak_models = {}
    
    if len(peak_frequencies) >= 1:
        # Single Gaussian fit
        try:
            # Initial guess for single peak
            amp_guess = np.max(peak_residuals)
            center_guess = peak_frequencies[np.argmax(peak_residuals)]
            width_guess = (freq_valid.max() - freq_valid.min()) / 20
            
            popt_single, pcov_single = curve_fit(
                gaussian, freq_valid, residuals_linear,
                p0=[amp_guess, center_guess, width_guess],
                bounds=([0, freq_valid.min(), 0], 
                       [np.inf, freq_valid.max(), freq_valid.max() - freq_valid.min()])
            )
            
            residuals_after_single = residuals_linear - gaussian(freq_valid, *popt_single)
            rms_single = np.std(residuals_after_single)
            
            peak_models['Single Gaussian'] = {
                'params': popt_single,
                'param_errors': np.sqrt(np.diag(pcov_single)),
                'residuals': residuals_after_single,
                'rms': rms_single,
                'model_func': gaussian
            }
            
            print(f"\nSingle Gaussian fit:")
            print(f"  Amplitude: {popt_single[0]:.6f} ± {np.sqrt(pcov_single[0,0]):.6f}")
            print(f"  Center frequency: {popt_single[1]:.6f} ± {np.sqrt(pcov_single[1,1]):.6f} Hz")
            print(f"  Width: {popt_single[2]:.6f} ± {np.sqrt(pcov_single[2,2]):.6f} Hz")
            print(f"  RMS after fit: {rms_single:.6f}")
            
        except Exception as e:
            print(f"Single Gaussian fit failed: {e}")
    
    if len(peak_frequencies) >= 2:
        # Double Gaussian fit
        try:
            # Initial guess for double peaks
            sorted_indices = np.argsort(peak_residuals)[::-1]  # Sort by amplitude
            amp1_guess = peak_residuals[sorted_indices[0]]
            center1_guess = peak_frequencies[sorted_indices[0]]
            amp2_guess = peak_residuals[sorted_indices[1]] if len(sorted_indices) > 1 else amp1_guess * 0.5
            center2_guess = peak_frequencies[sorted_indices[1]] if len(sorted_indices) > 1 else center1_guess * 1.5
            width_guess = (freq_valid.max() - freq_valid.min()) / 30
            
            popt_double, pcov_double = curve_fit(
                double_gaussian, freq_valid, residuals_linear,
                p0=[amp1_guess, center1_guess, width_guess, amp2_guess, center2_guess, width_guess],
                bounds=([0, freq_valid.min(), 0, 0, freq_valid.min(), 0],
                       [np.inf, freq_valid.max(), freq_valid.max()-freq_valid.min(), 
                        np.inf, freq_valid.max(), freq_valid.max()-freq_valid.min()])
            )
            
            residuals_after_double = residuals_linear - double_gaussian(freq_valid, *popt_double)
            rms_double = np.std(residuals_after_double)
            
            peak_models['Double Gaussian'] = {
                'params': popt_double,
                'param_errors': np.sqrt(np.diag(pcov_double)),
                'residuals': residuals_after_double,
                'rms': rms_double,
                'model_func': double_gaussian
            }
            
            print(f"\nDouble Gaussian fit:")
            print(f"  Peak 1 - Amp: {popt_double[0]:.6f}, Center: {popt_double[1]:.6f} Hz, Width: {popt_double[2]:.6f}")
            print(f"  Peak 2 - Amp: {popt_double[3]:.6f}, Center: {popt_double[4]:.6f} Hz, Width: {popt_double[5]:.6f}")
            print(f"  RMS after fit: {rms_double:.6f}")
            
        except Exception as e:
            print(f"Double Gaussian fit failed: {e}")

else:
    print("\nNo significant peaks detected in residuals.")
    peak_models = {}

# Plotting
plt.figure(figsize=(16, 12))

# Subplot 1: Original PDS with continuum
plt.subplot(3, 3, 1)
plt.loglog(freq_valid, power_valid, 'b.', alpha=0.7, markersize=4, label='Data')
plt.loglog(freq_valid, power_continuum, 'r-', linewidth=2, label='3rd Order Continuum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.title('PDS with Continuum Model')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 2: Residuals in linear space
plt.subplot(3, 3, 2)
plt.semilogx(freq_valid, residuals_linear, 'g.', alpha=0.7, markersize=4, label='Residuals')
plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
plt.axhline(y=np.std(residuals_linear), color='r', linestyle='--', alpha=0.5, label='±1σ')
plt.axhline(y=-np.std(residuals_linear), color='r', linestyle='--', alpha=0.5)
if len(all_peak_indices) > 0:
    plt.scatter(peak_frequencies, peak_residuals, color='red', s=50, marker='^', 
                label=f'{len(peak_frequencies)} peaks', zorder=5)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Residual Power')
plt.title('Residuals (Data - Continuum)')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 3: Residuals in log space
plt.subplot(3, 3, 3)
plt.semilogx(freq_valid, residuals, 'purple', alpha=0.7, marker='.', markersize=4, label='Log residuals')
plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
plt.axhline(y=np.std(residuals), color='r', linestyle='--', alpha=0.5, label='±1σ')
plt.axhline(y=-np.std(residuals), color='r', linestyle='--', alpha=0.5)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Log Residuals')
plt.title('Residuals in Log Space')
plt.legend()
plt.grid(True, alpha=0.3)

# Peak fitting plots
if len(peak_models) > 0:
    subplot_idx = 4
    freq_fine = np.logspace(np.log10(freq_valid.min()), np.log10(freq_valid.max()), 1000)
    
    for model_name, model_result in peak_models.items():
        if subplot_idx <= 6:  # Only plot first 3 models
            plt.subplot(3, 3, subplot_idx)
            
            # Plot residuals and fit
            plt.semilogx(freq_valid, residuals_linear, 'g.', alpha=0.7, markersize=4, 
                        label='Residuals')
            
            # Plot the fitted model
            model_prediction = model_result['model_func'](freq_fine, *model_result['params'])
            plt.semilogx(freq_fine, model_prediction, 'r-', linewidth=2, 
                        label=f'{model_name} fit')
            
            # Plot individual peaks if it's a multi-peak model
            if 'Double' in model_name:
                params = model_result['params']
                peak1 = gaussian(freq_fine, params[0], params[1], params[2])
                peak2 = gaussian(freq_fine, params[3], params[4], params[5])
                plt.semilogx(freq_fine, peak1, '--', alpha=0.7, label='Peak 1')
                plt.semilogx(freq_fine, peak2, '--', alpha=0.7, label='Peak 2')
            
            plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Residual Power')
            plt.title(f'{model_name}\nRMS: {model_result["rms"]:.6f}')
            plt.legend(fontsize=8)
            plt.grid(True, alpha=0.3)
            
            subplot_idx += 1

# Subplot 7: Peak significance analysis
plt.subplot(3, 3, 7)
if len(all_peak_indices) > 0:
    # Calculate significance of each peak
    noise_level = np.std(residuals_linear)
    significance = peak_residuals / noise_level
    
    bars = plt.bar(range(len(peak_frequencies)), significance, alpha=0.7, color='orange')
    plt.axhline(y=3, color='r', linestyle='--', label='3σ threshold')
    plt.axhline(y=5, color='r', linestyle='-', label='5σ threshold')
    
    # Label bars with frequency values
    for i, (freq, sig) in enumerate(zip(peak_frequencies, significance)):
        plt.text(i, sig + 0.1, f'{freq:.3f} Hz', rotation=45, ha='left', fontsize=8)
    
    plt.xlabel('Peak Index')
    plt.ylabel('Significance (σ)')
    plt.title('Peak Significance Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
else:
    plt.text(0.5, 0.5, 'No peaks detected', ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('Peak Significance Analysis')

# Subplot 8: Model comparison if multiple models exist
plt.subplot(3, 3, 8)
if len(peak_models) > 1:
    model_names = list(peak_models.keys())
    rms_values = [peak_models[name]['rms'] for name in model_names]
    
    bars = plt.bar(model_names, rms_values, alpha=0.7, color=['blue', 'green', 'orange'][:len(model_names)])
    plt.ylabel('RMS of Residuals')
    plt.title('Peak Model Comparison')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Highlight best model (lowest RMS)
    best_idx = np.argmin(rms_values)
    bars[best_idx].set_edgecolor('black')
    bars[best_idx].set_linewidth(3)
else:
    plt.text(0.5, 0.5, 'Only one model\navailable', ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('Peak Model Comparison')

# Subplot 9: Summary statistics
plt.subplot(3, 3, 9)
plt.axis('off')
summary_text = f"""Peak Fitting Summary
==================
Total frequency bins: {len(freq_valid)}
Frequency range: {freq_valid.min():.3f} - {freq_valid.max():.3f} Hz

Continuum (3rd order polynomial):
RMS residuals: {np.std(residuals_linear):.6f}

Peak Detection:
Peaks found: {len(all_peak_indices) if len(all_peak_indices) > 0 else 0}
"""

if len(all_peak_indices) > 0:
    summary_text += f"""
Peak frequencies: {', '.join([f'{f:.3f}' for f in peak_frequencies])} Hz
Max significance: {np.max(peak_residuals/np.std(residuals_linear)):.1f}σ
"""

if len(peak_models) > 0:
    best_model = min(peak_models.keys(), key=lambda k: peak_models[k]['rms'])
    summary_text += f"""
Best peak model: {best_model}
Best RMS: {peak_models[best_model]['rms']:.6f}
"""

plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes, fontsize=10, 
         verticalalignment='top', fontfamily='monospace')

plt.tight_layout()
plt.savefig('residual_peak_fitting.jpg', dpi=300, bbox_inches='tight')
plt.show()

# Save detailed results
with open('residual_peak_results.txt', 'w') as f:
    f.write("Residual Peak Fitting Results\n")
    f.write("============================\n\n")
    
    f.write("Continuum Model (3rd Order Polynomial):\n")
    f.write(f"Parameters: {popt_continuum}\n")
    f.write(f"RMS of residuals: {np.std(residuals_linear):.6f} (linear space)\n")
    f.write(f"RMS of residuals: {np.std(residuals):.6f} (log space)\n\n")
    
    f.write("Peak Detection:\n")
    f.write(f"Number of peaks found: {len(all_peak_indices) if len(all_peak_indices) > 0 else 0}\n")
    
    if len(all_peak_indices) > 0:
        f.write(f"Peak frequencies: {peak_frequencies}\n")
        f.write(f"Peak amplitudes: {peak_residuals}\n")
        noise_level = np.std(residuals_linear)
        significance = peak_residuals / noise_level
        f.write(f"Peak significance (σ): {significance}\n\n")
        
        for i, (freq, amp, sig) in enumerate(zip(peak_frequencies, peak_residuals, significance)):
            f.write(f"Peak {i+1}: {freq:.6f} Hz, amplitude = {amp:.6f}, significance = {sig:.1f}σ\n")
    
    f.write("\nPeak Model Fits:\n")
    f.write("=" * 40 + "\n")
    
    if len(peak_models) > 0:
        for model_name, result in peak_models.items():
            f.write(f"\n{model_name}:\n")
            f.write(f"RMS of residuals after fit: {result['rms']:.6f}\n")
            params = result['params']
            errors = result['param_errors']
            
            if 'Single' in model_name:
                f.write(f"Amplitude: {params[0]:.6f} ± {errors[0]:.6f}\n")
                f.write(f"Center frequency: {params[1]:.6f} ± {errors[1]:.6f} Hz\n")
                f.write(f"Width (σ): {params[2]:.6f} ± {errors[2]:.6f} Hz\n")
                f.write(f"FWHM: {2.355 * params[2]:.6f} Hz\n")
                
            elif 'Double' in model_name:
                f.write(f"Peak 1 - Amp: {params[0]:.6f} ± {errors[0]:.6f}\n")
                f.write(f"Peak 1 - Center: {params[1]:.6f} ± {errors[1]:.6f} Hz\n")
                f.write(f"Peak 1 - Width: {params[2]:.6f} ± {errors[2]:.6f} Hz\n")
                f.write(f"Peak 2 - Amp: {params[3]:.6f} ± {errors[3]:.6f}\n")
                f.write(f"Peak 2 - Center: {params[4]:.6f} ± {errors[4]:.6f} Hz\n")
                f.write(f"Peak 2 - Width: {params[5]:.6f} ± {errors[5]:.6f} Hz\n")
        
        best_model = min(peak_models.keys(), key=lambda k: peak_models[k]['rms'])
        f.write(f"\nBest model: {best_model} (lowest RMS)\n")
    else:
        f.write("No significant peaks detected or fitted.\n")
    
    f.write(f"\nAnalysis completed on: June 24, 2025\n")

print(f"\nAnalysis complete!")
print(f"Results saved to 'residual_peak_results.txt'")
print(f"Plot saved to 'residual_peak_fitting.jpg'")
