import os
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import argparse
import random

def process_audio_data(data, sample_rate, num_bins=20, freq_range=(200, 4000), window_duration=0.1, debug=False):
    if debug:
        print("Processing audio data.")

    # Check if data is stereo
    if data.ndim > 1:
        data = data.mean(axis=1)  # Convert to mono by averaging channels

    window_samples = int(sample_rate * window_duration)
    num_windows = len(data) // window_samples

    if num_windows == 0:
        return []

    # Truncate data to fit an integer number of windows
    data = data[:num_windows * window_samples]
    data_windows = data.reshape((num_windows, window_samples))

    # Initialize list to hold binned FFT magnitudes per window
    binned_ffts = []
    freqs = np.fft.rfftfreq(window_samples, d=1/sample_rate)

    # Create frequency bins
    bin_edges = np.linspace(freq_range[0], freq_range[1], num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Identify indices of frequencies within the specified range
    freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    freqs_in_range = freqs[freq_mask]

    for idx, window_data in enumerate(data_windows):
        # Normalize audio data
        window_data = window_data.astype(np.float32)
        if np.max(np.abs(window_data)) > 0:
            window_data = window_data / np.max(np.abs(window_data))

        # Compute FFT
        fft_result = np.fft.rfft(window_data)
        fft_magnitude = np.abs(fft_result)

        # Apply frequency mask
        fft_magnitude_in_range = fft_magnitude[freq_mask]

        # Bin the FFT magnitudes
        bin_indices = np.digitize(freqs_in_range, bins=bin_edges) - 1  # Adjust indices to start from 0
        binned_magnitude = np.zeros(num_bins)
        for i in range(len(fft_magnitude_in_range)):
            bin_idx = bin_indices[i]
            if 0 <= bin_idx < num_bins:
                binned_magnitude[bin_idx] += fft_magnitude_in_range[i]

        binned_ffts.append(binned_magnitude)

        if debug:
            # Plotting the time-domain signal
            time_axis = np.linspace(0, window_duration, window_samples)
            plt.figure(figsize=(15, 4))
            plt.plot(time_axis, window_data)
            plt.title(f'Window {idx+1} - Time Domain Signal')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')

            # Plotting the FFT magnitude
            plt.figure(figsize=(15, 4))
            plt.plot(freqs_in_range, fft_magnitude_in_range)
            plt.title(f'Window {idx+1} - FFT Magnitude Spectrum')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Magnitude')

            # Plotting the binned FFT magnitudes
            plt.figure(figsize=(15, 4))
            plt.plot(bin_centers, binned_magnitude)
            plt.title(f'Window {idx+1} - Binned FFT Magnitudes')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Magnitude')

            plt.show()

    return binned_ffts

def process_audio_file(filename, num_bins=20, freq_range=(200, 4000), window_duration=0.1, debug=False):
    if debug:
        print(f"Reading file: {filename}")

    sample_rate, data = wav.read(filename)
    return process_audio_data(data, sample_rate, num_bins, freq_range, window_duration, debug)

def process_class_folder(class_folder, num_bins=20, freq_range=(200, 4000), window_duration=0.1):
    fft_magnitudes = {}
    for filename in os.listdir(class_folder):
        if filename.endswith('.wav'):
            filepath = os.path.join(class_folder, filename)
            binned_ffts = process_audio_file(filepath, num_bins, freq_range, window_duration)
            fft_magnitudes[filename] = binned_ffts
    return fft_magnitudes

def main():
    parser = argparse.ArgumentParser(description='Process audio files and extract FFT features.')
    parser.add_argument('main_folder', type=str, help='Path to the main folder containing class subfolders.')
    parser.add_argument('--num_bins', type=int, default=100, help='Number of frequency bins (default: 20)')
    parser.add_argument('--freq_range', type=float, nargs=2, default=[300, 2300], help='Frequency range in Hz (default: 200 4000)')
    parser.add_argument('--window_duration', type=float, default=0.1, help='Window duration in seconds (default: 0.1)')
    args = parser.parse_args()

    main_folder = args.main_folder
    num_bins = args.num_bins
    freq_range = tuple(args.freq_range)
    window_duration = args.window_duration

    classes = [d for d in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, d))]
    class_fft_magnitudes = {}

    for class_name in classes:
        class_folder = os.path.join(main_folder, class_name)
        fft_magnitudes = process_class_folder(class_folder, num_bins, freq_range, window_duration)
        class_fft_magnitudes[class_name] = fft_magnitudes

    # Prepare bins for plotting
    bin_edges = np.linspace(freq_range[0], freq_range[1], num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2



# Allow external scripts to import functions
if __name__ == '__main__':
    main()
