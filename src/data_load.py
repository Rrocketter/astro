from stingray import Lightcurve
from stingray.powerspectrum import AveragedPowerspectrum
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

# Read file
with fits.open('1130360105/xti/event_cl/ni1130360105_0mpu7_cl.evt.gz') as hdul:
    data = hdul[1].data
    times = data['TIME']

# light curve
dt = 0.01
time_bins = np.arange(times.min(), times.max(), dt)
counts, _ = np.histogram(times, bins=time_bins)
time_bin_centers = (time_bins[:-1] + time_bins[1:]) / 2

lc = Lightcurve(time_bin_centers, counts, dt=dt)

# plot and save light curve
plt.figure()
plt.plot(lc.time, lc.counts, color='black')
plt.xlabel("Time (s)")
plt.ylabel("Counts")
plt.title("Light Curve of MAXI J1535-571")
plt.savefig("lightcurve.jpg", dpi=300)
plt.close()

# pds
ps = AveragedPowerspectrum(lc, segment_size=16.0)

# plot and save standard pds
plt.figure()
plt.loglog(ps.freq, ps.power, color='blue')
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")
plt.title("PDS of MAXI J1535-571")
plt.savefig("pds.jpg", dpi=300)
plt.close()

# plot and save frequency-weighted pds
plt.figure()
plt.loglog(ps.freq, ps.freq * ps.power, color='red')
plt.xlabel("Frequency (Hz)")
plt.ylabel(r"$\nu P(\nu)$")
plt.title("Frequency-weighted PDS of MAXI J1535-571")
plt.savefig("frequency_weighted_pds.jpg", dpi=300)
plt.close()