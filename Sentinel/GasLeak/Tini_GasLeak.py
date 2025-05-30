import numpy as np
import matplotlib.pyplot as plt
import wave
import time

# Import external beamforming functions
from Utilities.functions import (
    microphone_positions_8_medium,
    calculate_delays_for_direction,
    apply_beamforming,
    apply_bandpass_filter,
)

# ---------- Constants ----------
RATE          = 192_000                 # Hz
CHUNK         = int(0.1 * RATE)         # 100 ms per chunk
LOWCUT        = 6_000.0                 # Hz
HIGHCUT       = 30_000.0                # Hz
FILTER_ORDER  = 5                       # Butterworth order
c             = 343                     # m s-1

azimuth_range    = np.arange(-180, 181, 4)   # °
elevation_range  = np.arange(0,   91, 4)     # °

mic_positions = [
    (0.0, 0.0, 0.02),         #11
    (0.0, 0.01, 0.02),        #12
    (0.0, -0.015, 0.0),       #13
    (0.0, 0.025, 0.0),        #14
    (0.005, 0.005, 0.02),     #21
    (-0.005, 0.005, 0.02),    #22
    (0.02, 0.005, 0.0),       #23
    (-0.02, 0.005, 0.0)       #24
]


'''
mic_positions = [
    (0.0, 0.0, 0.02),       #11
    (0.005, 0.005, 0.02),   #12
    (0.0, 0.01, 0.02),      #13
    (-0.005, 0.005, 0.02),  #14
    (0.0, -0.015, 0),       #21
    (0.02, 0.005, 0),       #22
    (0.0, 0.025, 0),        #23
    (-0.02, 0.005, 0)       #24
]
'''

#CHANNELS = mic_positions.shape[0]  # Number of microphones based on geometry
CHANNELS = 8

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


    # -------------------------------------
    # Process the audio file chunk by chunk
    # -------------------------------------
    frame_idx = 0                       # <-- add this before the loop
    start_file = time.perf_counter()    # <-- optional: overall timer

    # Process the audio file chunk by chunk
    while True:
        # Read a chunk of audio data from the file
        t0 = time.perf_counter()  # <-- start timer for this chunk
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
        #heatmap.set_clim(vmin=np.min(energy_map), vmax=np.max(energy_map))
        #np.percentile(smoothed_energy, 1)
        #heatmap.set_clim(3e7, vmax=np.max(energy_map))
        heatmap.set_clim(np.percentile(energy_map, 50), vmax=np.max(energy_map))
        #heatmap.set_clim(1e7, 1e9)
        max_energy_marker.set_data([estimated_azimuth], [estimated_elevation])
        fig.canvas.draw()
        fig.canvas.flush_events()


        # Delay to simulate real-time processing (approx. 100 ms per chunk)
        #time.sleep(0.1)

        # ---------- timing & progress print ----------
        frame_idx += 1
        processing_time = time.perf_counter() - t0      # s
        audio_time     = frame_idx * CHUNK / RATE       # s of audio processed
        print(f"audio time = {audio_time:7.2f} s ")

        # Delay to simulate real-time playback (100 ms)
        time.sleep(0.1 - processing_time
                   if processing_time < 0.1 else 0)

    plt.ioff()
    plt.show()
    wf.close()


# -------------------------------------
# Main Execution: List of WAV Files to Process
# -------------------------------------
wav_filenames = [
    #'/Users/30068385/OneDrive - Western Sydney University/FlightRecord/DJI Inspire 1/CSV/03 Mar 25/1/20250303_133939_File0_Master_device.wav'
    'C:/Users/30068385/OneDrive - Western Sydney University/recordings/GasLeake/robotF2.wav'
]

for wav_file in wav_filenames:
    print(f"Processing file: {wav_file}")
    process_audio_file(wav_file)
