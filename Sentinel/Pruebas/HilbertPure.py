import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

# -------------------------- Parameters --------------------------
fs = 1000               # Sampling rate (Hz)
duration = 1.0          # Signal duration in seconds
t = np.linspace(0, duration, int(fs * duration), endpoint=False)

f0 = 5                  # Signal frequency in Hz
delay = 0.025            # Delay in seconds for microphone 2 (50 ms)

# -------------------------- Synthetic Signals --------------------------
# Microphone 1: pure cosine
mic1 = np.cos(2 * np.pi * f0 * t)
# Microphone 2: same cosine but delayed by "delay" seconds
mic2 = np.cos(2 * np.pi * f0 * (t - delay))

# -------------------------- Hilbert Transform --------------------------
# Compute analytic signals for each microphone
analytic_mic1 = hilbert(mic1)
analytic_mic2 = hilbert(mic2)

# Extract envelope and phase for each channel
envelope1 = np.abs(analytic_mic1)
phase1 = np.unwrap(np.angle(analytic_mic1))

envelope2 = np.abs(analytic_mic2)
phase2 = np.unwrap(np.angle(analytic_mic2))

# -------------------------- Visualization of Individual Signals --------------------------
plt.figure(figsize=(12, 10))

# Plot original signals
plt.subplot(3, 1, 1)
plt.plot(t, mic1, label='Microphone 1')
plt.plot(t, mic2, label='Microphone 2', linestyle='--')
plt.title('Original Synthetic Signals')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()

# Plot envelopes obtained from analytic signals
plt.subplot(3, 1, 2)
plt.plot(t, envelope1, label='Envelope 1')
plt.plot(t, envelope2, label='Envelope 2', linestyle='--')
plt.title('Envelopes from Analytic Signals')
plt.xlabel('Time [s]')
plt.ylabel('Envelope')
plt.legend()

# Plot phases obtained from analytic signals
plt.subplot(3, 1, 3)
plt.plot(t, phase1, label='Phase 1')
plt.plot(t, phase2, label='Phase 2', linestyle='--')
plt.title('Phases from Analytic Signals')
plt.xlabel('Time [s]')
plt.ylabel('Phase [radians]')
plt.legend()

plt.tight_layout()
plt.show()

# -------------------------- Beamforming using Hilbert Method --------------------------
# The idea: compensate for the delay in mic2 by rotating its phase.
# For a sinusoid of frequency f0, a delay "delay" seconds causes a phase shift of:
#   Δφ = 2π f0 * delay
# To align mic2 with mic1, we multiply its analytic signal by exp(j*2π*f0*delay)
phase_correction = np.exp(1j * 2 * np.pi * f0 * delay)
aligned_mic2 = analytic_mic2 * phase_correction

# Sum the analytic signals (beamforming)
beamformed_signal = analytic_mic1 + aligned_mic2

# Calculate envelope and phase of the beamformed signal
beamformed_envelope = np.abs(beamformed_signal)
beamformed_phase = np.unwrap(np.angle(beamformed_signal))

# -------------------------- Visualization of Beamformed Signal --------------------------
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(t, np.real(beamformed_signal), label='Beamformed Signal (Real Part)')
plt.title('Beamformed Signal (Real Part)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t, beamformed_envelope, label='Beamformed Envelope', color='orange')
plt.title('Beamformed Signal Envelope')
plt.xlabel('Time [s]')
plt.ylabel('Envelope')
plt.legend()

plt.tight_layout()
plt.show()
