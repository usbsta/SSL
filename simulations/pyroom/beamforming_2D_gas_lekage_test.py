import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf  # pip install pysoundfile


def delay_and_sum_beamforming(audio_chunk, mic_positions, fs, angles_deg, c=343.0):
    """
    Applies delay-and-sum beamforming on a single chunk of audio data
    from multiple microphones.

    Parameters:
    - audio_chunk: NumPy array of shape (num_samples, num_mics).
    - mic_positions: List of tuples [(x1, y1), (x2, y2), ...] in meters.
    - fs: Sampling frequency in Hz.
    - angles_deg: List or array of angles (in degrees) to scan.
    - c: Speed of sound in m/s.

    Returns:
    - energies: The computed energy for each angle in angles_deg.
    """
    num_samples, num_mics = audio_chunk.shape
    energies = []

    angles_rad = np.radians(angles_deg)

    for angle in angles_rad:
        # Direction unit vector (assuming 2D plane)
        direction = np.array([np.cos(angle), np.sin(angle)])
        beamformed = np.zeros(num_samples)

        for m in range(num_mics):
            mic_pos = np.array(mic_positions[m])
            # Distance in direction of arrival
            delay_distance = np.dot(mic_pos, direction)
            # Convert distance to time, then time to integer sample delay
            time_delay = delay_distance / c
            sample_delay = int(np.round(time_delay * fs))

            # Create a shifted signal array
            shifted = np.zeros(num_samples)

            # --- Handle shifting carefully to avoid invalid slice shapes ---
            if sample_delay == 0:
                # No shift
                shifted[:] = audio_chunk[:, m]
            elif sample_delay > 0:
                # Shift signal forward by sample_delay samples
                if sample_delay < num_samples:
                    shifted[sample_delay:] = audio_chunk[:num_samples - sample_delay, m]
                # If sample_delay >= num_samples, entire shifted remains zeros
            else:
                # sample_delay < 0
                # Negative shift: shift signal backward by abs(sample_delay)
                neg_delay = abs(sample_delay)
                if neg_delay < num_samples:
                    shifted[:num_samples - neg_delay] = audio_chunk[neg_delay:, m]
                # If abs(sample_delay) >= num_samples, entire shifted remains zeros

            # Sum up each aligned channel
            beamformed += shifted

        # Compute energy of the beamformed signal
        energy = np.sum(beamformed ** 2)
        energies.append(energy)

    return np.array(energies)


def main():
    # === Parameters ===
    wav_file = "C:/Users/30068385/OneDrive - Western Sydney University/ICNS/PhD/simulations/pyroom/offline_file_number_0_master_device.wav"
    mic_positions = [
        (0.0, 0.0),
        (0.055, 0.0),
        (0.065, 0.01),
        (0.065, 0.02),
        (0.055, 0.03),
        (0.0, 0.03),
        (-0.01, 0.02),
        (-0.01, 0.01)
    ]
    angles_deg = np.arange(0, 360, 10)  # Angle resolution (0–358 with step of 2)
    window_size_sec = 0.2  # 100 ms
    overlap_size_sec = 0.05  # 50 ms
    c = 343.0  # Speed of sound in m/s

    # === Read audio file ===
    audio_data, fs = sf.read(wav_file)  # audio_data shape: (num_samples, num_mics)
    num_samples, num_mics = audio_data.shape
    print("Audio samples:", num_samples, " Number of mics:", num_mics)

    # Convert window size and overlap to samples
    window_size = int(window_size_sec * fs)
    overlap_size = int(overlap_size_sec * fs)
    hop_size = window_size - overlap_size

    # We will collect the beamforming energies for each time frame
    all_energies = []  # Will be a list of arrays, each array = energies over angles
    time_axis = []  # Will store one time value (the midpoint of the window) per frame

    # === Sliding window processing ===
    start_index = 0
    while (start_index + window_size) <= num_samples:
        end_index = start_index + window_size
        chunk = audio_data[start_index:end_index, :]  # shape: (window_size, num_mics)

        # Perform delay-and-sum beamforming for all angles
        energies = delay_and_sum_beamforming(chunk, mic_positions, fs, angles_deg, c=c)

        # Store for plotting
        midpoint_time = (start_index + window_size / 2.0) / fs
        time_axis.append(midpoint_time)
        all_energies.append(energies)

        start_index += hop_size

    # Convert all_energies into a 2D NumPy array of shape (num_frames, num_angles)
    all_energies = np.array(all_energies)  # shape = (T, A)
    time_axis = np.array(time_axis)  # shape = (T,)

    # === Plot as a heatmap ===
    # By default, imshow expects the shape (num_rows, num_cols) where:
    #  - rows map to y (angles)
    #  - cols map to x (time)
    #
    # Our array is (T, A), i.e. (time, angle). We want:
    #  - x-axis = time
    #  - y-axis = angle
    #
    # So we can either transpose the array or provide "origin='lower'" with shape logic.
    # We'll transpose so that index 0 of the array is angle=0 at the bottom row.

    # We can define an extent that maps pixel coordinates to actual time/angle ranges:
    # extent = [time_axis[0], time_axis[-1], angles_deg[0], angles_deg[-1]]
    # We'll do a small trick to ensure the upper bound is not cut off visually by imshow.

    # If there's only one time frame, we guard against time_axis[-1] == time_axis[0].
    if len(time_axis) > 1:
        t_min, t_max = time_axis[0], time_axis[-1]
    else:
        # If there's only 1 frame, just set an arbitrary small range
        t_min, t_max = time_axis[0], time_axis[0] + 0.001

    # Similarly for angles
    a_min, a_max = angles_deg[0], angles_deg[-1]

    # Create the 2D heatmap
    plt.figure()
    plt.imshow(all_energies.T,
               origin='lower',
               aspect='auto',
               extent=[t_min, t_max, a_min, a_max])
    plt.colorbar(label='Energy')
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (deg)")
    plt.title("Delay-and-Sum Beamforming Energy (Heatmap)")
    plt.show()


if __name__ == "__main__":
    main()
