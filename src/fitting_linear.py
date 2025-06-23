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

# Create light curve
dt = 0.01
time_bins = np.arange(times.min(), times.max(), dt)
counts, _ = np.histogram(times, bins=time_bins)
time_bin_centers = (time_bins[:-1] + time_bins[1:]) / 2

lc = Lightcurve(time_bin_centers, counts, dt=dt)

# Generate power spectrum
ps = AveragedPowerspectrum(lc, segment_size=16.0)

# Remove zero frequencies and powers for log transformation
valid_mask = (ps.freq > 0) & (ps.power > 0)
freq_valid = ps.freq[valid_mask]
power_valid = ps.power[valid_mask]

# Transform to log space
log_freq = np.log10(freq_valid)
log_power = np.log10(power_valid)

# Define linear model: log(P) = m * log(f) + b
def linear_model(log_f, slope, intercept):
    return slope * log_f + intercept

# Fit linear model to log-log data
popt, pcov = curve_fit(linear_model, log_freq, log_power)
slope, intercept = popt
slope_err, intercept_err = np.sqrt(np.diag(pcov))

# Calculate R-squared
log_power_pred = linear_model(log_freq, slope, intercept)
ss_res = np.sum((log_power - log_power_pred) ** 2)
ss_tot = np.sum((log_power - np.mean(log_power)) ** 2)
r_squared = 1 - (ss_res / ss_tot)

# Calculate Pearson correlation coefficient
correlation, p_value = stats.pearsonr(log_freq, log_power)

# Print results
print("Linear Fit Results (log-log space):")
print(f"Slope (power-law index): {slope:.3f} ± {slope_err:.3f}")
print(f"Intercept: {intercept:.3f} ± {intercept_err:.3f}")
print(f"R-squared: {r_squared:.3f}")
print(f"Pearson correlation: {correlation:.3f} (p-value: {p_value:.2e})")
print()
print("Power-law model: P(f) = A * f^α")
print(f"Where α (power-law index) = {slope:.3f}")
print(f"And A (normalization) = 10^{intercept:.3f} = {10**intercept:.2e}")

# Create fit for plotting
freq_fit = np.logspace(np.log10(freq_valid.min()), np.log10(freq_valid.max()), 100)
power_fit = (10**intercept) * (freq_fit**slope)

# Plot results
plt.figure(figsize=(12, 8))

# Subplot 1: Log-log plot with fit
plt.subplot(2, 2, 1)
plt.loglog(freq_valid, power_valid, 'b.', alpha=0.6, markersize=2, label='Data')
plt.loglog(freq_fit, power_fit, 'r-', linewidth=2, label=f'Linear fit (α={slope:.2f})')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.title('PDS with Linear Continuum Fit')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 2: Residuals in log space
plt.subplot(2, 2, 2)
residuals = log_power - log_power_pred
plt.scatter(log_freq, residuals, alpha=0.6, s=1)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('log10(Frequency)')
plt.ylabel('Residuals (log10(Power))')
plt.title('Residuals from Linear Fit')
plt.grid(True, alpha=0.3)

# Subplot 3: Linear relationship in log space
plt.subplot(2, 2, 3)
plt.scatter(log_freq, log_power, alpha=0.6, s=1, label='Data')
plt.plot(log_freq, log_power_pred, 'r-', linewidth=2, label=f'y = {slope:.2f}x + {intercept:.2f}')
plt.xlabel('log10(Frequency)')
plt.ylabel('log10(Power)')
plt.title('Linear Fit in Log-Log Space')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 4: Frequency-weighted PDS with fit
plt.subplot(2, 2, 4)
freq_weighted_power = freq_valid * power_valid
freq_weighted_fit = freq_fit * power_fit
plt.loglog(freq_valid, freq_weighted_power, 'b.', alpha=0.6, markersize=2, label='Data')
plt.loglog(freq_fit, freq_weighted_fit, 'r-', linewidth=2, label='Continuum model')
plt.xlabel('Frequency (Hz)')
plt.ylabel(r'$\nu P(\nu)$')
plt.title('Frequency-weighted PDS')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('linear_continuum_fit.jpg', dpi=300, bbox_inches='tight')
plt.show()

# Save fit parameters to file
with open('continuum_fit_results.txt', 'w') as f:
    f.write("Linear Continuum Fit Results\n")
    f.write("============================\n\n")
    f.write(f"Power-law model: P(f) = A * f^α\n")
    f.write(f"Power-law index (α): {slope:.6f} ± {slope_err:.6f}\n")
    f.write(f"Normalization (A): {10**intercept:.6e}\n")
    f.write(f"Log normalization: {intercept:.6f} ± {intercept_err:.6f}\n")
    f.write(f"R-squared: {r_squared:.6f}\n")
    f.write(f"Pearson correlation: {correlation:.6f}\n")
    f.write(f"P-value: {p_value:.6e}\n")
    f.write(f"Number of data points: {len(freq_valid)}\n")

print(f"\nResults saved to 'continuum_fit_results.txt'")
print(f"Plot saved to 'linear_continuum_fit.jpg'")