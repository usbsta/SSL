import numpy as np
import matplotlib.pyplot as plt
import wave
import time

# Import external beamforming functions
from Utilities.functions import (
    microphone_positions_8_medium,
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

# Define beamforming grid for azimuth and elevation angles
azimuth_range = np.arange(-180, 181, 4)  # Azimuth from -180° to 180° in 4° steps
elevation_range = np.arange(0, 91, 4)  # Elevation from 0° to 90° in 4° steps

# Initialize microphone positions and determine the number of channels
mic_positions = microphone_positions_8_medium()

CHANNELS = mic_positions.shape[0]  # Number of microphones based on geometry


# Precompute delay samples for each (azimuth, elevation) pair
precomputed_delays = np.empty((len(azimuth_range), len(elevation_range), CHANNELS), dtype=np.int32)
for i, az in enumerate(azimuth_range):
    for j, el in enumerate(elevation_range):
        precomputed_delays[i, j, :] = calculate_delays_for_direction(mic_positions, az, el, RATE, c)


# -------------------------------------
# Function to Process a WAV File in Chunks
# -------------------------------------
def process_audio_file(wav_filename):
    """
    Process the input WAV file in chunks and perform beamforming to compute an energy map.
    The energy map is displayed in a simulated real-time manner.
    """
    # Open the WAV file for reading
    wf = wave.open(wav_filename, 'rb')

    # Verify that the WAV file parameters match the expected values
    if wf.getnchannels() != CHANNELS:
        print(f"Error: Expected {CHANNELS} channels, but found {wf.getnchannels()} in {wav_filename}")
        wf.close()
        return
    if wf.getsampwidth() != 2:  # 16-bit audio expected
        print("Error: WAV file sample width is not 16-bit.")
        wf.close()
        return
    if wf.getframerate() != RATE:
        print("Error: WAV file sampling rate does not match the expected RATE.")
        wf.close()
        return

    # Set up an interactive plot for real-time visualization
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 3))
    heatmap = ax.imshow(np.zeros((len(azimuth_range), len(elevation_range))).T,
                        extent=[azimuth_range[0], azimuth_range[-1], elevation_range[0], elevation_range[-1]],
                        origin='lower', aspect='auto', cmap='inferno')
    fig.colorbar(heatmap, ax=ax, label='Energy')
    ax.set_xlabel('Azimuth (degrees)')
    ax.set_ylabel('Elevation (degrees)')
    ax.set_title('Beamforming Energy Map (Simulated Real-time)')

    # Marker for maximum energy and CSV trajectory line
    max_energy_marker, = ax.plot([], [], 'ro', label='Max Energy')
    plt.legend()
    plt.grid(True)
    fig.canvas.draw()
    fig.canvas.flush_events()

    # Process the audio file chunk by chunk
    while True:
        # Read a chunk of audio data from the file
        data = wf.readframes(CHUNK)
        if len(data) < CHUNK * CHANNELS * 2:  # Incomplete chunk indicates end of file
            break

        # Convert binary data to a NumPy array and reshape to (CHUNK, CHANNELS)
        audio_chunk = np.frombuffer(data, dtype=np.int16).reshape((-1, CHANNELS))

        # Apply bandpass filter to each channel
        filtered_chunk = apply_bandpass_filter(audio_chunk, LOWCUT, HIGHCUT, RATE, order=FILTER_ORDER)

        # Compute the beamforming energy map
        energy_map = np.zeros((len(azimuth_range), len(elevation_range)))
        for i in range(len(azimuth_range)):
            for j in range(len(elevation_range)):
                # Apply beamforming using the precomputed delays for current (azimuth, elevation)
                beamformed_signal = apply_beamforming(filtered_chunk, precomputed_delays[i, j, :])
                # Calculate the energy of the beamformed signal
                energy_map[i, j] = np.sum(beamformed_signal ** 2)

        # Update the heatmap display
        max_idx = np.unravel_index(np.argmax(energy_map), energy_map.shape)
        estimated_azimuth = azimuth_range[max_idx[0]]
        estimated_elevation = elevation_range[max_idx[1]]

        heatmap.set_data(energy_map.T)
        heatmap.set_clim(vmin=np.min(energy_map), vmax=np.max(energy_map))
        max_energy_marker.set_data([estimated_azimuth], [estimated_elevation])
        fig.canvas.draw()
        fig.canvas.flush_events()

        # Delay to simulate real-time processing (approx. 100 ms per chunk)
        time.sleep(0.1)

    plt.ioff()
    plt.show()
    wf.close()


# -------------------------------------
# Main Execution: List of WAV Files to Process
# -------------------------------------
wav_filenames = [
    #'/Users/30068385/OneDrive - Western Sydney University/FlightRecord/DJI Inspire 1/CSV/03 Mar 25/1/20250303_133939_File0_Master_device.wav'
    'C:/Users/a30068385/OneDrive - Western Sydney University/ICNS/PhD/simulations/pyroom/offline_file_number_0_master_device.wav'
]

for wav_file in wav_filenames:
    print(f"Processing file: {wav_file}")
    process_audio_file(wav_file)
