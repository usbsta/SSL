# acoustic_utils_iso9613.py
"""
Acoustic utilities for ISO 9613 atmospheric absorption, SPL calibration, noise mixing,
and bandpass filtering for 1-octave or 1/3-octave band simulation.

All comments are in English.
"""

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt, resample, resample_poly

# ---------------------------------------------------------------------------
# 1) ISO 9613-1 Atmospheric Absorption (full frequency support)
# ---------------------------------------------------------------------------
def atmospheric_absorption_iso9613(frequencies, temperature=20.0, humidity=70.0, pressure=101.325):
    """
    Calculate atmospheric absorption coefficient α(f) [dB/m] for each frequency using ISO 9613-1.

    Args:
        frequencies (np.ndarray): Array of frequencies in Hz.
        temperature (float): Temperature in Celsius.
        humidity (float): Relative humidity in %.
        pressure (float): Atmospheric pressure in kPa.

    Returns:
        alpha (np.ndarray): Absorption coefficients in dB/m for each frequency.
    """
    T_kelvin = temperature + 273.15
    T_ref = 293.15  # 20°C in Kelvin

    # Saturation vapor pressure (hPa)
    saturation_pressure = 6.1078 * 10 ** ((7.5 * temperature) / (temperature + 237.3))
    # Convert to Pa
    saturation_pressure = saturation_pressure * 100.0

    # Actual vapor pressure (Pa)
    H = humidity * saturation_pressure / 100.0

    # dimensionless humidity ratio (Pa / kPa)
    h = H / (pressure * 1000.0)

    p0 = 101.325  # reference pressure [kPa]

    # Relaxation frequencies
    frO = (pressure / p0) * (24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h))
    frN = (pressure / p0) * (T_kelvin / T_ref) ** (-0.5) * (9.0 + 280.0 * h * np.exp(-4.17 * ((T_kelvin / T_ref) ** (-1.0/3.0) - 1.0)))

    alpha = []
    for f in frequencies:
        termO = 0.01275 * np.exp(-2239.1 / T_kelvin) / (frO + (f ** 2) / frO)
        termN = 0.1068 * np.exp(-3352.0 / T_kelvin) / (frN + (f ** 2) / frN)
        alpha_f = 8.686 * (f ** 2) * (
            1.84e-11 * (p0 / pressure) * (T_kelvin / T_ref) ** 0.5
            + termO + termN
        )
        alpha.append(alpha_f)

    return np.array(alpha)

# ---------------------------------------------------------------------------
# 2) Single-frequency amplitude factor (Geometric + Atmospheric) from ISO 9613
# ---------------------------------------------------------------------------
def iso9613_attenuation_factor(distance, frequency, temperature=20.0, humidity=70.0, pressure=101.325):
    """
    Compute total attenuation factor (amplitude) at a given distance and frequency,
    combining geometric spreading and atmospheric absorption from ISO 9613.

    Args:
        distance (float): Distance in meters.
        frequency (float): Frequency in Hz.
        temperature (float): Temperature in Celsius.
        humidity (float): Relative humidity in %.
        pressure (float): kPa.

    Returns:
        attenuation_factor (float): Amplitude multiplier.
    """
    if distance <= 0.0:
        return 1.0

    import math
    import warnings
    if distance < 1e-6:
        warnings.warn("Distance is extremely small, returning factor 1.0.")
        return 1.0

    # Geometric attenuation (20 * log10(d)) in dB
    A_geom = 20.0 * np.log10(distance)

    # Atmospheric absorption at this frequency in dB/m
    alpha_db_per_m = atmospheric_absorption_iso9613(
        np.array([frequency]),
        temperature=temperature,
        humidity=humidity,
        pressure=pressure
    )[0]
    A_atm = alpha_db_per_m * distance

    # Combine in dB
    A_total_db = A_geom + A_atm

    # Convert to amplitude ratio
    attenuation_factor = 10.0 ** (-A_total_db / 20.0)
    return attenuation_factor

# ---------------------------------------------------------------------------
# 3) SPL Calibration (based on pistófono reference)
# ---------------------------------------------------------------------------
def calibrate_signal_to_spl(signal, reference_signal, target_spl_db=94.0):
    """
    Calibrate 'signal' amplitude based on known SPL reference in 'reference_signal'.
    We assume 'reference_signal' is recorded at target_spl_db (e.g. 94 dB SPL).

    Args:
        signal (np.ndarray): Signal to calibrate.
        reference_signal (np.ndarray): Reference signal at known SPL.
        target_spl_db (float): SPL of reference in dB.

    Returns:
        calibrated_signal (np.ndarray): Signal scaled to physical SPL.
        original_spl_signal (float): Estimated SPL of 'signal' before calibration.
    """
    rms_ref = np.sqrt(np.mean(reference_signal ** 2))
    rms_sig = np.sqrt(np.mean(signal ** 2))

    # SPL of 'signal' if 'reference_signal' is at target_spl_db.
    original_spl_signal = target_spl_db + 20.0 * np.log10((rms_sig + 1e-12)/(rms_ref + 1e-12))

    # We want 'signal' to match the same reference SPL
    gain_factor = 10.0 ** ((target_spl_db - original_spl_signal)/20.0)
    calibrated_signal = signal * gain_factor

    return calibrated_signal, original_spl_signal

# ---------------------------------------------------------------------------
# 4) Mix noise with a desired SNR
# ---------------------------------------------------------------------------
def mix_signals_with_snr(signal, noise, snr_db=None):
    """
    Mix signal and noise to achieve desired SNR in dB.

    If snr_db is None, the real ratio is preserved.

    signal, noise: 1D or 2D (channels x samples)

    Returns:
        mixed (np.ndarray)
    """
    if signal.ndim == 1:
        signal = signal[np.newaxis, :]
    if noise.ndim == 1:
        noise = noise[np.newaxis, :]

    n_channels = signal.shape[0]
    mixed = np.zeros_like(signal)

    for c in range(n_channels):
        sig_c = signal[c]
        noi_c = noise[c]
        rms_sig = np.sqrt(np.mean(sig_c**2))
        rms_noi = np.sqrt(np.mean(noi_c**2))
        if snr_db is None:
            # Keep real ratio
            mixed[c] = sig_c + noi_c
        else:
            desired_rms_noi = rms_sig / (10.0**(snr_db/20.0))
            scale_factor = desired_rms_noi / (rms_noi + 1e-12)
            noi_scaled = noi_c * scale_factor
            mixed[c] = sig_c + noi_scaled

    if mixed.shape[0] == 1:
        return mixed[0]
    return mixed

# ---------------------------------------------------------------------------
# 5) Bandpass filters for octave or 1/3-octave
# ---------------------------------------------------------------------------
def butter_bandpass(lowcut, highcut, fs, order=4):
    """
    Create a Butterworth bandpass filter in SOS form.
    lowcut, highcut: cutoff frequencies in Hz
    fs: sample rate (Hz)
    """
    from scipy.signal import butter, sosfilt
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    return sos

def filter_octave_band(signal, center_freq, fs, order=4):
    """
    Filter signal in a 1-octave band around center_freq (f0).
    band edges: [f0/sqrt(2), f0*sqrt(2)]
    """
    import math
    lowcut = center_freq / math.sqrt(2)
    highcut = center_freq * math.sqrt(2)
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    filtered = sosfilt(sos, signal)
    return filtered
