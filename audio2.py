import socket
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import time

# ----------------------------
# Audio Processing Parameters
# ----------------------------
RATE = 48000  # Sampling rate in Hz
CHUNK = int(0.1 * RATE)  # 100 ms per chunk
CHANNELS = 8  # 8 microphones
BYTES_PER_SAMPLE = 4  # 32-bit integer = 4 bytes
BYTES_PER_CHUNK = CHUNK * CHANNELS * BYTES_PER_SAMPLE

# ----------------------------
# Bandpass Filter Parameters
# ----------------------------
LOWCUT = 400.0  # Lower cutoff frequency in Hz
HIGHCUT = 18000.0  # Upper cutoff frequency in Hz
FILTER_ORDER = 5  # Order of the Butterworth filter


"""
# Define a circular microphone array with 8 microphones placed on a circle of radius 0.1 m at z=0
num_mics = CHANNELS
radius = 0.1
angles = np.linspace(0, 2 * np.pi, num_mics, endpoint=False)
mic_positions = np.zeros((num_mics, 3))
for i in range(num_mics):
    mic_positions[i, 0] = radius * np.cos(angles[i])  # x-coordinate
    mic_positions[i, 1] = radius * np.sin(angles[i])  # y-coordinate
    mic_positions[i, 2] = 0  # z-coordinate
"""

# ----------------------------
# Original Microphone Geometry
# ----------------------------
# Provided arrays (using "config 2 augmented")
a = [0, -120, -240]  # Angles in degrees
# a2 is not used here as we only want 8 microphones
h = [1.12, 1.02, 0.87, 0.68, 0.47, 0.02]
r = [0.1, 0.16, 0.23, 0.29, 0.43, 0.63]

# Build the mic positions exactly as in your original code but only take the first 8 positions.
# The original positions (for 18 mics) are defined in blocks;
# here we select:
#   Mic 1: [r[0]*cos(0°),    r[0]*sin(0°),    h[0]]
#   Mic 2: [r[0]*cos(-120°), r[0]*sin(-120°), h[0]]
#   Mic 3: [r[0]*cos(-240°), r[0]*sin(-240°), h[0]]
#   Mic 4: [r[1]*cos(0°),    r[1]*sin(0°),    h[1]]
#   Mic 5: [r[1]*cos(-120°), r[1]*sin(-120°), h[1]]
#   Mic 6: [r[1]*cos(-240°), r[1]*sin(-240°), h[1]]
#   Mic 7: [r[2]*cos(0°),    r[2]*sin(0°),    h[2]]
#   Mic 8: [r[2]*cos(-120°), r[2]*sin(-120°), h[2]]
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
    """Apply a zero-phase Butterworth bandpass filter."""
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return filtfilt(b, a, data, axis=0)


# ----------------------------
# Beamforming Function
# ----------------------------
def beamform_time(signal_data, mic_positions, az_range, el_range, fs, c):
    """
    Compute the beamforming energy map.

    Parameters:
      signal_data: 2D NumPy array of shape (num_samples, num_channels)
      mic_positions: Array of microphone positions (num_channels x 3)
      az_range: Array of azimuth angles (in degrees)
      el_range: Array of elevation angles (in degrees)
      fs: Sampling rate
      c: Speed of sound

    Returns:
      energy: 2D NumPy array with energy values for each (azimuth, elevation) pair.
    """
    num_samples = signal_data.shape[0]
    energy = np.zeros((len(az_range), len(el_range)))

    for i, az in enumerate(az_range):
        az_rad = np.radians(az)
        for j, el in enumerate(el_range):
            el_rad = np.radians(el)
            # Compute 3D unit direction vector for (az, el)
            direction = np.array([
                np.cos(el_rad) * np.cos(az_rad),
                np.cos(el_rad) * np.sin(az_rad),
                np.sin(el_rad)
            ])
            # Compute delays for each mic (in seconds)
            delays = np.dot(mic_positions, direction) / c
            combined = np.zeros(num_samples)
            # Apply circular shift to each channel by the appropriate delay (in samples)
            for mic in range(signal_data.shape[1]):
                delay_samples = int(np.round(delays[mic] * fs))
                combined += np.roll(signal_data[:, mic], delay_samples)
            # Normalize by number of microphones and compute energy
            combined /= signal_data.shape[1]
            energy[i, j] = np.sum(combined ** 2)
    return energy


# Define beamforming grid
azimuth_range = np.arange(-180, 181, 5)  # -180° to 180° in 5° steps
elevation_range = np.arange(0, 91, 5)  # 0° to 90° in 5° steps

# ----------------------------
# Socket Setup to Receive Audio Data
# ----------------------------
# Adjust IP and PORT to match your audio sending code
IP = '127.0.0.1'
PORT = 5001

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((IP, PORT))
print(f"Connected to audio server at {IP}:{PORT}")

# ----------------------------
# Real-Time Processing and Visualization
# ----------------------------
plt.ion()
fig, ax = plt.subplots(figsize=(8, 6))
heatmap = ax.imshow(np.zeros((len(azimuth_range), len(elevation_range))).T,
                    extent=[azimuth_range[0], azimuth_range[-1],
                            elevation_range[0], elevation_range[-1]],
                    origin='lower', aspect='auto', cmap='viridis')
fig.colorbar(heatmap, ax=ax, label='Energy')
ax.set_xlabel('Azimuth (degrees)')
ax.set_ylabel('Elevation (degrees)')
ax.set_title('Real-time Beamforming Energy Map')

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

        # Convert binary data to NumPy array and reshape to (CHUNK, CHANNELS)
        audio_chunk = np.frombuffer(chunk_data, dtype=np.int32)
        audio_chunk = audio_chunk.reshape((-1, CHANNELS))

        # Apply the bandpass filter to each channel
        filtered_chunk = apply_bandpass_filter(audio_chunk, LOWCUT, HIGHCUT, RATE, order=FILTER_ORDER)

        # Compute the beamforming energy map using the filtered data
        energy_map = beamform_time(filtered_chunk, mic_positions, azimuth_range, elevation_range, RATE, c)

        # Update the heatmap display
        heatmap.set_data(energy_map.T)
        heatmap.set_clim(vmin=np.min(energy_map), vmax=np.max(energy_map))
        fig.canvas.draw()
        fig.canvas.flush_events()

        # Small delay to control the update rate
        time.sleep(0.01)

except KeyboardInterrupt:
    print("Processing interrupted by user.")

finally:
    sock.close()
    print("Socket closed.")
    plt.ioff()
    plt.show()
