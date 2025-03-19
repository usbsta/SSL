#!/usr/bin/env python
import socket
import numpy as np
import matplotlib.pyplot as plt
import time
import struct
from datetime import datetime

# Import beamforming-related functions from your external module.
# These functions must be defined in your "functions.py" file.
from Sentinel.Pruebas.functions import (
    initialize_microphone_positions,
    calculate_delays_for_direction,
    apply_bandpass_filter,
    apply_beamforming
)

# =============================================================================
# Configuration Parameters
# =============================================================================
RATE = 48000                          # Audio sampling rate (Hz)
CHUNK_DURATION = 0.1                  # Chunk duration in seconds (100 ms)
CHUNK = int(CHUNK_DURATION * RATE)    # Number of samples per chunk
BYTES_PER_SAMPLE = 2                  # 16-bit audio → 2 bytes per sample

# =============================================================================
# Initialize Microphone Geometry and Precompute Delays
# =============================================================================
mic_positions = initialize_microphone_positions()  # Your function for 8 mics
CHANNELS = mic_positions.shape[0]                  # Number of microphones
c = 343.0                                          # Speed of sound in m/s

# Define the beamforming grid (azimuth and elevation angles)
azimuth_range = np.arange(-180, 181, 4)            # From -180° to 180° in 4° steps
elevation_range = np.arange(0, 91, 4)                # From 0° to 90° in 4° steps
num_azimuth = len(azimuth_range)
num_elevation = len(elevation_range)

# Precompute delay samples for each direction (using your function)
precomputed_delays = np.empty((num_azimuth, num_elevation, CHANNELS), dtype=np.int32)
for i, az in enumerate(azimuth_range):
    for j, el in enumerate(elevation_range):
        precomputed_delays[i, j, :] = calculate_delays_for_direction(mic_positions, az, el, RATE, c)

BYTES_PER_CHUNK = CHUNK * CHANNELS * BYTES_PER_SAMPLE

# =============================================================================
# Setup Socket for Audio Reception
# =============================================================================
IP_SERVER = '127.0.0.1'
PORT_SERVER = 5001
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((IP_SERVER, PORT_SERVER))
print(f"Connected to audio server at {IP_SERVER}:{PORT_SERVER}")

# =============================================================================
# Setup WAV File for Recording
# =============================================================================
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
wav_filename = f"{current_time}_recording.wav"

def write_wav_header(file, num_channels, sample_rate, bits_per_sample):
    """Write a placeholder WAV header to the file."""
    byte_rate = sample_rate * num_channels * bits_per_sample // 8
    block_align = num_channels * bits_per_sample // 8
    file.write(b"RIFF")
    file.write(struct.pack("<I", 0))  # Placeholder for chunk size
    file.write(b"WAVE")
    file.write(b"fmt ")
    file.write(struct.pack("<I", 16))  # PCM subchunk size
    file.write(struct.pack("<H", 1))   # Audio format (PCM)
    file.write(struct.pack("<H", num_channels))
    file.write(struct.pack("<I", sample_rate))
    file.write(struct.pack("<I", byte_rate))
    file.write(struct.pack("<H", block_align))
    file.write(struct.pack("<H", bits_per_sample))
    file.write(b"data")
    file.write(struct.pack("<I", 0))  # Placeholder for data chunk size

wav_file = open(wav_filename, "wb")
write_wav_header(wav_file, CHANNELS, RATE, 16)
data_size = 0

# =============================================================================
# Setup Real-Time Visualization (Beamforming Energy Map)
# =============================================================================
plt.ion()
fig, ax = plt.subplots(figsize=(12, 3))
# Initialize heatmap with zeros; note the transposition for correct orientation.
heatmap = ax.imshow(np.zeros((num_azimuth, num_elevation)).T,
                    extent=[azimuth_range[0], azimuth_range[-1],
                            elevation_range[0], elevation_range[-1]],
                    origin='lower', aspect='auto', cmap='jet')
fig.colorbar(heatmap, ax=ax, label='Energy')
ax.set_xlabel('Azimuth (degrees)')
ax.set_ylabel('Elevation (degrees)')
ax.set_title('Real-time Beamforming Energy Map')
plt.show()

# =============================================================================
# Main Processing Loop: Record Audio & Perform Beamforming in Real Time
# =============================================================================
try:
    while True:
        # ----------------------------
        # Receive a complete audio chunk from the socket
        # ----------------------------
        chunk_data = b""
        while len(chunk_data) < BYTES_PER_CHUNK:
            packet = sock.recv(BYTES_PER_CHUNK - len(chunk_data))
            if not packet:
                break
            chunk_data += packet
        if len(chunk_data) != BYTES_PER_CHUNK:
            print("Incomplete chunk received, exiting...")
            break

        # ----------------------------
        # Write raw audio data to WAV file
        # ----------------------------
        wav_file.write(chunk_data)
        data_size += len(chunk_data)

        # ----------------------------
        # Convert chunk to NumPy array and reshape to (CHUNK, CHANNELS)
        # ----------------------------
        audio_chunk = np.frombuffer(chunk_data, dtype=np.int16).reshape((-1, CHANNELS))

        # ----------------------------
        # Apply bandpass filter (using your function from functions.py)
        # ----------------------------
        filtered_chunk = apply_bandpass_filter(audio_chunk, 400.0, 18000.0, RATE, order=5)

        # ----------------------------
        # Compute the beamforming energy map
        # For each (azimuth, elevation) pair, apply delay-and-sum beamforming.
        # ----------------------------
        energy_map = np.zeros((num_azimuth, num_elevation))
        for i in range(num_azimuth):
            for j in range(num_elevation):
                # Use precomputed delays for the current direction
                beamformed_signal = apply_beamforming(filtered_chunk, precomputed_delays[i, j, :])
                energy_map[i, j] = np.sum(beamformed_signal ** 2)

        # ----------------------------
        # Update the heatmap display
        # ----------------------------
        heatmap.set_data(energy_map.T)
        heatmap.set_clim(vmin=np.min(energy_map), vmax=np.max(energy_map))
        fig.canvas.draw()
        fig.canvas.flush_events()

        # Optional sleep to control update rate
        time.sleep(0.01)

except KeyboardInterrupt:
    print("Processing interrupted by user.")

finally:
    # Close socket
    sock.close()
    print("Socket closed.")

    # Update the WAV header with the correct data size
    wav_file.seek(4)
    wav_file.write(struct.pack("<I", 36 + data_size))
    wav_file.seek(40)
    wav_file.write(struct.pack("<I", data_size))
    wav_file.close()
    print(f"Audio recorded and saved as {wav_filename}")

    plt.ioff()
    plt.show()
