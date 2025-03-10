import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, filtfilt

# --------------------- Parameters ---------------------
fs = 1000                   # Sampling frequency (Hz)
duration = 1.0              # Signal duration in seconds
t = np.linspace(0, duration, int(fs * duration), endpoint=False)

lowcut = 1.0                # Low cutoff frequency (Hz)
highcut = 10.0              # High cutoff frequency (Hz)
f_center = (lowcut + highcut) / 2.0  # Center frequency for phase correction

delay = 0.05                # Delay in seconds for microphone 2 (50 ms)
delay_samples = int(delay * fs)

# --------------------- Bandpass Filter Function ---------------------
def bandpass_filter(data, lowcut, highcut, fs, order=3):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# --------------------- Generate Synthetic Random Signals ---------------------
np.random.seed(0)  # For reproducibility
mic1_white = np.random.randn(len(t))
mic2_white = np.random.randn(len(t))

# Filter white noise to get band-limited noise in the desired band
mic1 = bandpass_filter(mic1_white, lowcut, highcut, fs)
mic2 = bandpass_filter(mic2_white, lowcut, highcut, fs)

# Simulate that mic2 is a delayed version of mic1 by shifting mic1
mic2 = np.concatenate((np.zeros(delay_samples), mic1[:-delay_samples]))

# --------------------- Compute Analytic Signals ---------------------
analytic_mic1 = hilbert(mic1)
analytic_mic2 = hilbert(mic2)

# Extract envelopes and phases
envelope1 = np.abs(analytic_mic1)
phase1 = np.unwrap(np.angle(analytic_mic1))

envelope2 = np.abs(analytic_mic2)
phase2 = np.unwrap(np.angle(analytic_mic2))

# --------------------- Visualize Individual Signals ---------------------
plt.figure(figsize=(12, 10))

# Original filtered signals
plt.subplot(3, 1, 1)
plt.plot(t, mic1, label='Microphone 1')
plt.plot(t, mic2, label='Microphone 2', linestyle='--')
plt.title('Filtered Random Signals')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()

# Envelopes from analytic signals
plt.subplot(3, 1, 2)
plt.plot(t, envelope1, label='Envelope 1')
plt.plot(t, envelope2, label='Envelope 2', linestyle='--')
plt.title('Envelopes from Hilbert Transform')
plt.xlabel('Time [s]')
plt.ylabel('Envelope')
plt.legend()

# Phases from analytic signals
plt.subplot(3, 1, 3)
plt.plot(t, phase1, label='Phase 1')
plt.plot(t, phase2, label='Phase 2', linestyle='--')
plt.title('Phases from Hilbert Transform')
plt.xlabel('Time [s]')
plt.ylabel('Phase [radians]')
plt.legend()

plt.tight_layout()
plt.show()

# --------------------- Beamforming with Phase Correction ---------------------
# For a sinusoidal signal at frequency f_center, a delay produces a phase shift:
#   delta_phi = 2*pi * f_center * delay
# Although the signals are random, we use f_center to simulate phase correction.
phase_correction = np.exp(1j * 2 * np.pi * f_center * delay)
aligned_mic2 = analytic_mic2 * phase_correction

# Sum the analytic signals (beamforming)
beamformed_signal = analytic_mic1 + aligned_mic2

# Compute envelope and phase of the beamformed signal
beamformed_envelope = np.abs(beamformed_signal)
beamformed_phase = np.unwrap(np.angle(beamformed_signal))

# --------------------- Visualize Beamformed Signal ---------------------
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(t, np.real(beamformed_signal), label='Real Part of Beamformed Signal')
plt.title('Beamformed Signal (Real Part)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t, beamformed_envelope, label='Beamformed Envelope', color='orange')
plt.title('Envelope of Beamformed Signal')
plt.xlabel('Time [s]')
plt.ylabel('Envelope')
plt.legend()

plt.tight_layout()
plt.show()
