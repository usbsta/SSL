import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav


def calculate_rms(signal):
    """
    Calculate the Root Mean Square (RMS) of the given signal.
    """
    return np.sqrt(np.mean(signal ** 2))


def compute_spl(data, fs, a, b, window_duration=0.1):
    """
    Compute the Sound Pressure Level (SPL) time series using the calibrated model:
        SPL = a * log10(RMS) + b
    over a moving window approach.

    Parameters:
        data (numpy array): Audio signal.
        fs (int): Sampling frequency.
        a (float): Calibration slope.
        b (float): Calibration offset.
        window_duration (float): Duration of each window in seconds.

    Returns:
        time_axis (numpy array): Time axis in seconds.
        spl_values (numpy array): Estimated SPL values (in dB).
    """
    window_size = int(window_duration * fs)
    num_windows = int((len(data) - window_size) / window_size)
    spl_values = []
    time_axis = []
    epsilon = 1e-12  # Small constant to avoid log10(0)

    for i in range(num_windows):
        start_idx = i * window_size
        end_idx = start_idx + window_size
        window = data[start_idx:end_idx]
        window_rms = calculate_rms(window)
        # Calculate SPL using the calibrated model
        spl = a * np.log10(window_rms + epsilon) + b
        spl_values.append(spl)
        time_axis.append(start_idx / fs)

    return np.array(time_axis), np.array(spl_values)


def adjust_spl_for_distance(spl_values, distance, ref_distance=1.0):
    """
    Adjust SPL values based on the measurement distance using the inverse-square law.

    The adjustment is performed as:
        SPL_adjusted = SPL_measured + 20 * log10(distance / ref_distance)
    where ref_distance is the reference distance (1 meter in this case).
    """
    return spl_values + 20 * np.log10(distance / ref_distance)


def calculate_leq(spl_values):
    """
    Calculate the equivalent continuous sound level (LEQ) from the SPL values.

    Formula:
        LEQ = 10 * log10(mean(10^(SPL/10)))
    """
    leq = 10 * np.log10(np.mean(10 ** (spl_values / 10)))
    return leq


def main():
    # Filename for the new recording made with the "measure" microphone
    filename = 'zoom/250312_004.WAV'  # Update with the correct path if needed

    # Read the audio file
    fs, data = wav.read(filename)

    # If the audio file is stereo, take only the first channel
    if data.ndim > 1:
        data = data[:, 0]

    # Normalize data to float values between -1 and 1 if necessary
    if data.dtype not in [np.float32, np.float64]:
        data = data / np.max(np.abs(data))

    # -------------------------------------------------------------------------
    # Calibration parameters for the "measure" microphone obtained previously.
    # Replace these placeholder values with your actual calibration parameters.
    # -------------------------------------------------------------------------
    a = 20.4067  # Example calibration slope (replace with actual value)
    b = 129.6568  # Example calibration offset (replace with actual value)

    # -------------------------------------------------------------------------
    # Measurement distance: distance between the sound source and the microphone.
    # In this case, it was 1 meter.
    # -------------------------------------------------------------------------
    distance = 1.0  # in meters

    # Define window duration for SPL analysis (in seconds)
    window_duration = 0.1  # 100 ms windows

    # Compute the SPL time series using the calibrated model
    time_axis, spl_values = compute_spl(data, fs, a, b, window_duration)

    # Adjust SPL values for the measurement distance using the inverse-square law
    spl_adjusted = adjust_spl_for_distance(spl_values, distance)

    # Calculate acoustic metrics using the adjusted SPL values
    leq = calculate_leq(spl_adjusted)
    lmax = np.max(spl_adjusted)
    l90 = np.percentile(spl_adjusted, 10)  # L90: 10th percentile (sound level exceeded 90% of the time)

    # Print the computed acoustic metrics and measurement distance
    print(f"Measurement distance: {distance} m")
    print(f"LEQ: {leq:.2f} dB")
    print(f"Lmax: {lmax:.2f} dB")
    print(f"L90: {l90:.2f} dB")

    # Plot the time-varying adjusted SPL time series
    plt.figure(figsize=(10, 5))
    plt.plot(time_axis, spl_adjusted, label='Adjusted SPL (dB)')
    plt.xlabel('Time (s)')
    plt.ylabel('SPL (dB)')
    plt.title('Time-varying Adjusted SPL Estimation')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
