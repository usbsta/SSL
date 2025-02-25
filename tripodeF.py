import socket
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import time

# ----------------------------
# Audio Processing Parameters
# ----------------------------
RATE = 48000                    # Sampling rate in Hz
CHUNK = int(0.1 * RATE)         # 100 ms per chunk
CHANNELS = 8                    # 8 microphones
BYTES_PER_SAMPLE = 2            # 16-bit integer => 2 bytes per sample
BYTES_PER_CHUNK = CHUNK * CHANNELS * BYTES_PER_SAMPLE

# ----------------------------
# Bandpass Filter Parameters
# ----------------------------
LOWCUT = 400.0                  # Lower cutoff frequency in Hz
HIGHCUT = 18000.0               # Upper cutoff frequency in Hz
FILTER_ORDER = 5                # Order of the Butterworth filter

# ----------------------------
# Microphone Geometry (Using your provided notation)
# ----------------------------
a = [0, -120, -240]             # Angles in degrees
h = [1.12, 1.02, 0.87, 0.68, 0.47, 0.02]
r = [0.1, 0.16, 0.23, 0.29, 0.43, 0.63]

mic_positions = np.array([
    [r[0] * np.cos(np.radians(a[0])), r[0] * np.sin(np.radians(a[0])), h[0]],
    [r[0] * np.cos(np.radians(a[1])), r[0] * np.sin(np.radians(a[1])), h[0]],
    [r[0] * np.cos(np.radians(a[2])), r[0] * np.sin(np.radians(a[2])), h[0]],
    [r[1] * np.cos(np.radians(a[0])), r[1] * np.sin(np.radians(a[0])), h[1]],
    [r[1] * np.cos(np.radians(a[1])), r[1] * np.sin(np.radians(a[1])), h[1]],
    [r[1] * np.cos(np.radians(a[2])), r[1] * np.sin(np.radians(a[2])), h[1]],
    [r[2] * np.cos(np.radians(a[0])), r[2] * np.sin(np.radians(a[0])), h[2]],
    [r[2] * np.cos(np.radians(a[1])), r[2] * np.sin(np.radians(a[1])), h[2]]
])
# Note: The 0° azimuth position corresponds to the first row: [r[0], 0, h[0]].

# Speed of sound in air (m/s)
c = 343

# ----------------------------
# Filter Design Functions
# ----------------------------
def butter_bandpass(lowcut, highcut, fs, order=5):
    """Design a Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band')

def apply_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Apply a zero-phase Butterworth bandpass filter along the time axis."""
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return filtfilt(b, a, data, axis=0)

# ----------------------------
# Precomputation of Phase Shift Factors
# ----------------------------
# Define the search grid (in degrees)
azimuth_range = np.arange(-180, 181, 4)   # From -180° to 180° in 4° steps
elevation_range = np.arange(0, 91, 2)       # From 0° to 90° in 2° steps

# Create a list of candidate directions (unit vectors)
grid_dirs = []
for az in azimuth_range:
    for el in elevation_range:
        az_rad = np.radians(az)
        el_rad = np.radians(el)
        d = np.array([np.cos(el_rad) * np.cos(az_rad),
                      np.cos(el_rad) * np.sin(az_rad),
                      np.sin(el_rad)])
        grid_dirs.append(d)
grid_dirs = np.array(grid_dirs)           # Shape: (num_dirs, 3)
num_dirs = grid_dirs.shape[0]

# Calculate delays for each candidate direction and each microphone:
# delays[d, i] = dot(mic_positions[i], direction_d) / c
delays = (grid_dirs @ mic_positions.T) / c  # Shape: (num_dirs, CHANNELS)

# Precompute FFT frequency bins for CHUNK samples
N = CHUNK
freqs = np.fft.rfftfreq(N, d=1/RATE)       # Shape: (N_fft,)
N_fft = len(freqs)

# Precompute the phase shift factors for each candidate direction, frequency bin, and microphone.
# Desired shape: (num_dirs, N_fft, CHANNELS)
phase_factors = np.exp(-1j * 2 * np.pi * freqs[None, :, None] * delays[:, None, :])
# Now: phase_factors[d, f, i] = exp(-j2π f * delay[d, i])

# ----------------------------
# Frequency-Domain Beamforming Function (Vectorized)
# ----------------------------
def beamform_freq_vectorized(signal_data, phase_factors):
    """
    Compute the beamforming energy in the frequency domain using vectorized operations.

    Parameters:
      signal_data: 2D array of shape (N, CHANNELS) with the filtered signal.
      phase_factors: Precomputed array of shape (num_dirs, N_fft, CHANNELS).

    Returns:
      energy: 1D array of length num_dirs with the beamforming energy for each candidate direction.
    """
    # Compute FFT along the time axis for each channel
    Y = np.fft.rfft(signal_data, axis=0)    # Shape: (N_fft, CHANNELS)
    # Multiply FFT data by the precomputed phase factors for all candidate directions
    # Expand Y to shape (1, N_fft, CHANNELS) for broadcasting
    Y_weighted = phase_factors * Y[None, :, :]  # Shape: (num_dirs, N_fft, CHANNELS)
    # Sum over the microphone channels
    Y_sum = np.sum(Y_weighted, axis=2)           # Shape: (num_dirs, N_fft)
    # Compute energy for each candidate direction: sum of squared magnitudes over frequency bins
    energy = np.sum(np.abs(Y_sum)**2, axis=1)      # Shape: (num_dirs,)
    return energy

# ----------------------------
# Socket Setup to Receive Audio Data
# ----------------------------
IP_SERVER = '127.0.0.1'
PORT_SERVER = 5001

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((IP_SERVER, PORT_SERVER))
print(f"Connected to audio server at {IP_SERVER}:{PORT_SERVER}")

# ----------------------------
# Real-Time Processing and Visualization
# ----------------------------
plt.ion()
fig, ax = plt.subplots(figsize=(12, 3))
# Initialize a heatmap with the shape of the search grid
energy_map_display = np.zeros((len(azimuth_range), len(elevation_range)))
heatmap = ax.imshow(energy_map_display.T,
                    extent=[azimuth_range[0], azimuth_range[-1],
                            elevation_range[0], elevation_range[-1]],
                    origin='lower', aspect='auto', cmap='inferno')
fig.colorbar(heatmap, ax=ax, label='Energy')
ax.set_xlabel('Azimuth (degrees)')
ax.set_ylabel('Elevation (degrees)')
ax.set_title('Real-time Frequency-Domain Beamforming Energy Map')

try:
    while True:
        # Receive data until a full chunk is obtained
        chunk_data = b''
        while len(chunk_data) < BYTES_PER_CHUNK:
            packet = sock.recv(BYTES_PER_CHUNK - len(chunk_data))
            if not packet:
                break
            chunk_data += packet

        if len(chunk_data) != BYTES_PER_CHUNK:
            print("Incomplete chunk received, exiting...")
            break

        # Convert binary data to a NumPy array and reshape to (CHUNK, CHANNELS)
        audio_chunk = np.frombuffer(chunk_data, dtype=np.int16)
        audio_chunk = audio_chunk.reshape((-1, CHANNELS))

        # Apply the bandpass filter to each channel
        filtered_chunk = apply_bandpass_filter(audio_chunk, LOWCUT, HIGHCUT, RATE, order=FILTER_ORDER)

        # Compute the beamforming energy using the precomputed phase factors
        energy_values = beamform_freq_vectorized(filtered_chunk, phase_factors)  # Shape: (num_dirs,)

        # Reshape the energy values into the grid shape (azimuth_range, elevation_range)
        energy_map = energy_values.reshape(len(azimuth_range), len(elevation_range))

        # Update the heatmap display
        heatmap.set_data(energy_map.T)
        heatmap.set_clim(vmin=np.min(energy_map), vmax=np.max(energy_map))
        fig.canvas.draw()
        fig.canvas.flush_events()

        # Short delay to control the update rate
        time.sleep(0.01)

except KeyboardInterrupt:
    print("Processing interrupted by user.")

finally:
    sock.close()
    print("Socket closed.")
    plt.ioff()
    plt.show()
