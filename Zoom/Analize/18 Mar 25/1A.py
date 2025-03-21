import numpy as np
import matplotlib.pyplot as plt
import wave
import time

# Import external beamforming functions
from Utilities.functions import (
    initialize_microphone_positions_24,
    calculate_delays_for_direction,
    apply_beamforming,
    apply_bandpass_filter
)

# -------------------------------------
# Parameters and Precomputations
# -------------------------------------
RATE = 48000  # Sampling rate in Hz
CHUNK = int(0.1 * RATE)  # Process 100 ms per chunk
LOWCUT = 400.0  # Lower cutoff frequency in Hz
HIGHCUT = 3000.0  # Upper cutoff frequency in Hz
FILTER_ORDER = 5  # Filter order for Butterworth filter
c = 343  # Speed of sound in air (m/s)
skip_seconds = 72

azimuth_range = np.arange(-180, 181, 5)
elevation_range = np.arange(0, 91, 5)

mic_positions = initialize_microphone_positions_24()
CHANNELS = mic_positions.shape[0]

wav_filenames = [
    '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/22 Nov 24/1/20241122_120459_device_1_sync_part1.wav',
    '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/22 Nov 24/1/20241122_120501_device_2_sync_part1.wav',
    '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/22 Nov 24/1/20241122_120502_device_3_sync_part1.wav',
    '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/22 Nov 24/1/20241122_120503_device_4_sync_part1.wav'
]