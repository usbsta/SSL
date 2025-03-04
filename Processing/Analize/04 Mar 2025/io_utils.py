# io_utils.py
import numpy as np
import wave
from scipy.signal import butter, filtfilt
import pandas as pd
from pyproj import Transformer

def butter_bandpass(lowcut, highcut, rate, order=5):
    # Design a bandpass filter
    nyquist = 0.5 * rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(signal_data, lowcut, highcut, rate, order=5):
    # Apply the bandpass filter to the signal_data along all channels
    b, a = butter_bandpass(lowcut, highcut, rate, order=order)
    filtered_signal = filtfilt(b, a, signal_data, axis=0)
    return filtered_signal

def read_wav_block(wav_file, chunk_size, channels):
    data = wav_file.readframes(chunk_size)
    if len(data) == 0:
        return None
    signal_data = np.frombuffer(data, dtype=np.int32)
    return np.reshape(signal_data, (-1, channels))

def skip_wav_seconds(wav_file, seconds, rate):
    frames_to_skip = int(seconds * rate)
    wav_file.setpos(frames_to_skip)

def calculate_time(time_idx, chunk_size, rate):
    # Calculate current time in seconds
    time_seconds = (time_idx * chunk_size) / rate
    return time_seconds

def load_flight_data(csv_path, altitude_col, latitude_col, longitude_col, time_col):
    # Load flight data (csv)
    flight_data = pd.read_csv(csv_path, skiprows=0, delimiter=',', low_memory=False)
    flight_data = flight_data[[latitude_col, longitude_col, altitude_col, time_col]].dropna()
    return flight_data


def initialize_beamforming_params(azimuth_range, elevation_range, c, RATE):
    """
    Initialize microphone positions, direction vectors, and calculate delay_samples.
    """
    # Microphone positions (example as given)
    a = [0, -60, -180, -300]

    # progressive distance configuration
    h = [0.45, 0.25, 0.11]
    r = [0.07, 0.02, 0.18]

    # Define microphone positions
    mic_positions = np.array([
        [r[0] * np.cos(np.radians(a[0])), r[0] * np.sin(np.radians(a[0])), h[0]], # mic 1
        [r[1] * np.cos(np.radians(a[2])), r[1] * np.sin(np.radians(a[2])), h[1]], # mic 2
        [r[0] * np.cos(np.radians(a[1])), r[0] * np.sin(np.radians(a[1])), h[0]], # mic 3
        [r[2] * np.cos(np.radians(a[1])), r[2] * np.sin(np.radians(a[1])), h[2]], # mic 4
        [r[0] * np.cos(np.radians(a[2])), r[0] * np.sin(np.radians(a[2])), h[0]], # mic 5
        [r[2] * np.cos(np.radians(a[2])), r[2] * np.sin(np.radians(a[2])), h[2]], # mic 6
        [r[0] * np.cos(np.radians(a[3])), r[0] * np.sin(np.radians(a[3])), h[0]], # mic 7
        [r[2] * np.cos(np.radians(a[3])), r[2] * np.sin(np.radians(a[3])), h[2]]  # mic 8
    ])

    azimuth_rad = np.radians(azimuth_range)
    elevation_rad = np.radians(elevation_range)
    num_az = len(azimuth_rad)
    num_el = len(elevation_rad)
    num_mics = mic_positions.shape[0]

    az_rad_grid, el_rad_grid = np.meshgrid(azimuth_rad, elevation_rad, indexing='ij')
    direction_vectors = np.empty((num_az, num_el, 3), dtype=np.float64)
    direction_vectors[:, :, 0] = np.cos(el_rad_grid) * np.cos(az_rad_grid)
    direction_vectors[:, :, 1] = np.cos(el_rad_grid) * np.sin(az_rad_grid)
    direction_vectors[:, :, 2] = np.sin(el_rad_grid)

    mic_positions_expanded = mic_positions[:, np.newaxis, np.newaxis, :]
    direction_vectors_expanded = direction_vectors[np.newaxis, :, :, :]
    delays = np.sum(mic_positions_expanded * direction_vectors_expanded, axis=3) / c
    delay_samples = np.round(delays * RATE).astype(np.int32)

    return mic_positions, delay_samples, num_mics

def open_wav_files(wav_filenames):
    # Open all WAV files
    wav_files = [wave.open(filename, 'rb') for filename in wav_filenames]
    return wav_files
