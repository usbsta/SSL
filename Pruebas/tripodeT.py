import socket
import numpy as np
import matplotlib.pyplot as plt
import time

# ---------------------------------------------------
# Import functions from the external module 'functions.py'
# ---------------------------------------------------
from Pruebas.functions import (
    initialize_microphone_positions,
    calculate_delays_for_direction,
    apply_beamforming,
    apply_bandpass_filter
)

# ---------------------------------------------------
# Main Parameters and Precomputation
# ---------------------------------------------------
RATE = 48000                    # Sampling rate in Hz
CHUNK = int(0.1 * RATE)         # 100 ms per chunk

# Initialize microphone positions using the imported function
mic_positions = initialize_microphone_positions()
CHANNELS = mic_positions.shape[0]    # Number of microphones based on geometry
BYTES_PER_SAMPLE = 2                 # 16-bit integer (2 bytes per sample)
BYTES_PER_CHUNK = CHUNK * CHANNELS * BYTES_PER_SAMPLE

# Bandpass filter parameters
LOWCUT = 400.0                # Lower cutoff frequency in Hz
HIGHCUT = 18000.0             # Upper cutoff frequency in Hz
FILTER_ORDER = 5              # Filter order for Butterworth

c = 343                       # Speed of sound in air (m/s)

# Define the beamforming grid
azimuth_range = np.arange(-180, 181, 4)  # Azimuth from -180° to 180° in 4° steps
elevation_range = np.arange(0, 91, 4)      # Elevation from 0° to 90° in 4° steps

# Precompute delay samples for each (azimuth, elevation) pair
precomputed_delays = np.empty((len(azimuth_range), len(elevation_range), CHANNELS), dtype=np.int32)
for i, az in enumerate(azimuth_range):
    for j, el in enumerate(elevation_range):
        precomputed_delays[i, j, :] = calculate_delays_for_direction(mic_positions, az, el, RATE, c)

# ---------------------------------------------------
# Socket Setup for Receiving Audio Data
# ---------------------------------------------------
IP_SERVER = '127.0.0.1'
PORT_SERVER = 5001
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((IP_SERVER, PORT_SERVER))
print(f"Connected to audio server at {IP_SERVER}:{PORT_SERVER}")

# ---------------------------------------------------
# Real-Time Processing and Visualization
# ---------------------------------------------------
plt.ion()
fig, ax = plt.subplots(figsize=(12, 3))
heatmap = ax.imshow(np.zeros((len(azimuth_range), len(elevation_range))).T,
                    extent=[azimuth_range[0], azimuth_range[-1],
                            elevation_range[0], elevation_range[-1]],
                    origin='lower', aspect='auto', cmap='jet')
fig.colorbar(heatmap, ax=ax, label='Energy')
ax.set_xlabel('Azimuth (degrees)')
ax.set_ylabel('Elevation (degrees)')
ax.set_title('Real-time Beamforming Energy Map')

try:
    while True:
        # ---------------------------------------------------
        # Receive data until a full chunk is obtained
        # ---------------------------------------------------
        chunk_data = b''
        while len(chunk_data) < BYTES_PER_CHUNK:
            packet = sock.recv(BYTES_PER_CHUNK - len(chunk_data))
            if not packet:
                break
            chunk_data += packet

        if len(chunk_data) != BYTES_PER_CHUNK:
            print("Incomplete chunk received, exiting...")
            break

        # Convert binary data to NumPy array and reshape to (CHUNK, CHANNELS)
        audio_chunk = np.frombuffer(chunk_data, dtype=np.int16).reshape((-1, CHANNELS))

        # ---------------------------------------------------
        # Apply bandpass filter on each channel using the imported function
        # ---------------------------------------------------
        filtered_chunk = apply_bandpass_filter(audio_chunk, LOWCUT, HIGHCUT, RATE, order=FILTER_ORDER)

        # ---------------------------------------------------
        # Compute the beamforming energy map using precomputed delays
        # ---------------------------------------------------
        energy_map = np.zeros((len(azimuth_range), len(elevation_range)))
        for i in range(len(azimuth_range)):
            for j in range(len(elevation_range)):
                # Use precomputed delay samples for current (azimuth, elevation) pair
                beamformed_signal = apply_beamforming(filtered_chunk, precomputed_delays[i, j, :])
                #beamformed_signal = apply_beamforming(audio_chunk, precomputed_delays[i, j, :])
                energy_map[i, j] = np.sum(beamformed_signal ** 2)

        # ---------------------------------------------------
        # Update the heatmap display
        # ---------------------------------------------------
        heatmap.set_data(energy_map.T)
        heatmap.set_clim(vmin=np.min(energy_map), vmax=np.max(energy_map))
        fig.canvas.draw()
        fig.canvas.flush_events()

        # Optional delay to control update rate
        time.sleep(0.01)

except KeyboardInterrupt:
    print("Processing interrupted by user.")

finally:
    sock.close()
    print("Socket closed.")
    plt.ioff()
    plt.show()
