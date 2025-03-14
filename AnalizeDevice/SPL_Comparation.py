"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav


def calculate_rms(signal):

    return np.sqrt(np.mean(signal ** 2))


# -------------------------
# Load the WAV file
# -------------------------
# Replace 'audio_file.wav' with your recording file path
fs, data = wav.read('measure.wav')

# If the audio is stereo, take only the first channel
if data.ndim > 1:
    data = data[:, 0]

# Normalize data to float values between -1 and 1 (if necessary)
if data.dtype != np.float32 and data.dtype != np.float64:
    data = data / np.max(np.abs(data))

# -------------------------
# Define calibration segments
# -------------------------
# Set the start and end times (in seconds) for the calibration segments.
# Update these values based on your recording where the calibration tones are present.
calib_94_start = 7.6  # start time for 94 dBSPL segment
calib_94_end = 13.4  # end time for 94 dBSPL segment

calib_114_start = 14.5  # start time for 114 dBSPL segment
calib_114_end = 18.0  # end time for 114 dBSPL segment

# Convert time to sample indices
idx_94_start = int(calib_94_start * fs)
idx_94_end = int(calib_94_end * fs)
idx_114_start = int(calib_114_start * fs)
idx_114_end = int(calib_114_end * fs)

# Extract calibration segments
segment_94 = data[idx_94_start:idx_94_end]
segment_114 = data[idx_114_start:idx_114_end]

# -------------------------
# Calculate RMS for calibration segments
# -------------------------
rms_94 = calculate_rms(segment_94)
rms_114 = calculate_rms(segment_114)

# Known SPL values for the calibration segments
spl_94 = 94.0
spl_114 = 114.0

# -------------------------
# Calibrate: Fit a linear model in the log domain
# The model is: SPL = a * log10(rms) + b
# -------------------------
X = np.log10(np.array([rms_94, rms_114]))
Y = np.array([spl_94, spl_114])
a, b = np.polyfit(X, Y, 1)
print(f"Calibration parameters: a = {a:.4f}, b = {b:.4f}")

# -------------------------
# Compute SPL over time using a moving window
# -------------------------
window_duration = 0.1  # window duration in seconds (e.g., 100 ms)
window_size = int(window_duration * fs)
num_windows = int((len(data) - window_size) / window_size)

spl_values = []
time_axis = []

# Process each window
for i in range(num_windows):
    start_idx = i * window_size
    end_idx = start_idx + window_size
    window = data[start_idx:end_idx]

    # Compute RMS of the window
    window_rms = calculate_rms(window)

    # Calculate SPL for the window using the calibrated model
    # Adding a small epsilon to avoid log10(0)
    epsilon = 1e-12
    spl = a * np.log10(window_rms + epsilon) + b
    spl_values.append(spl)
    time_axis.append(start_idx / fs)

# -------------------------
# Plot the SPL time series
# -------------------------
plt.figure(figsize=(10, 5))
plt.plot(time_axis, spl_values, label='Estimated SPL (dB)')
plt.xlabel('Time (s)')
plt.ylabel('SPL (dB)')
plt.title('Time-varying SPL Estimation')
plt.legend()
plt.grid(True)
plt.show()


"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

def calculate_rms(signal):
    """
    Calculate the Root Mean Square (RMS) of a given signal.
    """
    return np.sqrt(np.mean(signal ** 2))

def compute_calibration_params(data, fs, calib_94_interval, calib_114_interval, spl_94=94.0, spl_114=114.0):
    """
    Compute the calibration parameters (a and b) for the model:
        SPL = a * log10(RMS) + b
    using two calibration segments defined by their start and end times (in seconds).
    """
    start_94, end_94 = calib_94_interval
    start_114, end_114 = calib_114_interval

    idx_94_start = int(start_94 * fs)
    idx_94_end = int(end_94 * fs)
    idx_114_start = int(start_114 * fs)
    idx_114_end = int(end_114 * fs)

    segment_94 = data[idx_94_start:idx_94_end]
    segment_114 = data[idx_114_start:idx_114_end]

    rms_94 = calculate_rms(segment_94)
    rms_114 = calculate_rms(segment_114)

    # Fit a linear model in the logarithmic domain: SPL = a * log10(RMS) + b
    X = np.log10(np.array([rms_94, rms_114]))
    Y = np.array([spl_94, spl_114])
    a, b = np.polyfit(X, Y, 1)
    return a, b

def estimate_spl(data, fs, a, b, window_duration=0.1):
    """
    Estimate the SPL (Sound Pressure Level) over time using a moving window approach.
    """
    window_size = int(window_duration * fs)
    num_windows = int((len(data) - window_size) / window_size)
    spl_values = []
    time_axis = []
    epsilon = 1e-12  # Small constant to avoid log(0)

    for i in range(num_windows):
        start_idx = i * window_size
        end_idx = start_idx + window_size
        window = data[start_idx:end_idx]
        window_rms = calculate_rms(window)
        spl = a * np.log10(window_rms + epsilon) + b
        spl_values.append(spl)
        time_axis.append(start_idx / fs)

    return np.array(time_axis), np.array(spl_values)

def process_audio_file(filename, calib_94_interval, calib_114_interval, window_duration=0.1):
    """
    Load an audio file, compute its calibration parameters and estimate its SPL over time.
    For mems.wav, only the 94 dB SPL calibration segment is used (with a fixed slope of 20)
    and the minimum SPL value is set to 0 dB.
    """
    fs, data = wav.read(filename)

    # If the audio file is stereo, take only the first channel.
    if data.ndim > 1:
        data = data[:, 0]

    # Normalize the data to floating point values between -1 and 1 if necessary.
    if data.dtype not in [np.float32, np.float64]:
        data = data / np.max(np.abs(data))

    # Check if the file is mems.wav to use only the 94 dB calibration
    if 'mems' in filename.lower():
        # Use only the 94 dB SPL calibration segment.
        start_94, end_94 = calib_94_interval
        idx_94_start = int(start_94 * fs)
        idx_94_end = int(end_94 * fs)
        segment_94 = data[idx_94_start:idx_94_end]
        rms_94 = calculate_rms(segment_94)
        # Assume a fixed slope (commonly 20 for voltage to SPL conversion)
        a = 20.0
        # Compute offset so that the calibration point corresponds to 94 dB SPL
        b = 94.0 - a * np.log10(rms_94)
        print(f"File: {filename} -> Using only 94 dB calibration. a = {a:.4f}, b = {b:.4f}")
    else:
        # For measure.wav, use both calibration segments.
        a, b = compute_calibration_params(data, fs, calib_94_interval, calib_114_interval)
        print(f"File: {filename} -> Calibration parameters: a = {a:.4f}, b = {b:.4f}")

    # Estimate the SPL over time using a moving window.
    time_axis, spl_values = estimate_spl(data, fs, a, b, window_duration)

    # For mems.wav, ensure the minimum SPL value is 0 dB.
    if 'mems' in filename.lower():
        spl_values = np.maximum(spl_values, 0)

    return time_axis, spl_values

def main():
    # Filenames for the two audio files (update these with the correct paths)
    file1 = 'measure.wav'
    file2 = 'mems.wav'
    file3 = 'rode.wav'
    file4 = 'ultra.wav'

    # Define calibration intervals for each file (in seconds)
    # For measure.wav, both calibration segments are used.
    calib_94_interval_file1 = (7.6, 13.4)    # 94 dB SPL segment for measure.wav
    calib_114_interval_file1 = (14.5, 18.0)    # 114 dB SPL segment for measure.wav

    # For mems.wav, only the 94 dB SPL segment is used.
    calib_94_interval_file2 = (274.3, 278.0)   # 94 dB SPL segment for mems.wav
    calib_114_interval_file2 = (196.0, 200.0)    # This interval will be ignored for mems.wav

    calib_94_interval_file3 = (37.8, 40.8)   # 94 dB SPL segment for mems.wav
    calib_114_interval_file3 = (41.6, 44.3)    # This interval will be ignored for mems.wav

    calib_94_interval_file4 = (26.0, 28.0)   # 94 dB SPL segment for mems.wav
    calib_114_interval_file4 = (29.5, 31.5)    # This interval will be ignored for mems.wav

    # Process both audio files.
    time1, spl1 = process_audio_file(file1, calib_94_interval_file1, calib_114_interval_file1)
    time2, spl2 = process_audio_file(file2, calib_94_interval_file2, calib_114_interval_file2)
    time3, spl3 = process_audio_file(file3, calib_94_interval_file3, calib_114_interval_file3)
    time4, spl4 = process_audio_file(file4, calib_94_interval_file4, calib_114_interval_file4)

    # Plot the SPL time series for both audio files.
    plt.figure(figsize=(12, 6))
    plt.plot(time1, spl1, label=f'SPL of {file1}')
    plt.plot(time2, spl2, label=f'SPL of {file2}')
    plt.plot(time3, spl3, label=f'SPL of {file3}')
    plt.plot(time4, spl4, label=f'SPL of {file4}')

    plt.xlabel('Time (s)')
    plt.ylabel('SPL (dB)')
    plt.title('Time-varying SPL Estimation for Two Audio Files')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
