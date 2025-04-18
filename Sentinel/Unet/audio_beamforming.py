# audio_beamforming.py
import numpy as np
from numba import njit, prange

# Beamforming related functions

@njit
def shift_signal(signal, delay_samples):
    # Shift the signal in time domain by delay_samples
    num_samples = signal.shape[0]
    shifted_signal = np.zeros_like(signal)

    if delay_samples > 0:
        # Delay: shift forward, pad at start
        if delay_samples < num_samples:
            shifted_signal[delay_samples:] = signal[:-delay_samples]
    elif delay_samples < 0:
        # Advance: shift backward, pad at end
        delay_samples = -delay_samples
        if delay_samples < num_samples:
            shifted_signal[:-delay_samples] = signal[delay_samples:]
    else:
        # No delay
        shifted_signal = signal.copy()

    return shifted_signal

@njit(parallel=True)
def beamform_time(signal_data, delay_samples):
    # Beamform time-domain signals given delay_samples for each mic and direction
    num_samples, num_mics = signal_data.shape
    num_mics_, num_az, num_el = delay_samples.shape
    energy = np.zeros((num_az, num_el))

    for az_idx in prange(num_az):
        for el_idx in range(num_el):
            output_signal = np.zeros(num_samples)
            for mic_idx in range(num_mics):
                delay = delay_samples[mic_idx, az_idx, el_idx]
                shifted_signal = shift_signal(signal_data[:, mic_idx], delay)
                output_signal += shifted_signal

            output_signal /= num_mics
            energy[az_idx, el_idx] = np.sum(output_signal ** 2)
    return energy
