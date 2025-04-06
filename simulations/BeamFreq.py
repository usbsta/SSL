import numpy as np
import matplotlib.pyplot as plt

# Constants
c = 343.0  # Speed of sound in m/s
fs = 48000  # Sampling frequency in Hz
f = 1000  # Signal frequency in Hz
T = 0.01  # Signal duration in seconds
t = np.arange(0, T, 1/fs)

# Microphone array configuration
num_mics_x = 4
num_mics_y = 4
mic_spacing = 0.05  # 5 cm spacing
mic_positions = np.array([[i * mic_spacing, j * mic_spacing] for j in range(num_mics_y) for i in range(num_mics_x)])

# Source configuration
source_position = np.array([1.0, 0.5])  # Position in meters

# Generate signal at source (1 kHz sine wave)
signal = np.sin(2 * np.pi * f * t)

# Function to compute time delays

def compute_delays(mic_positions, source_position):
    distances = np.linalg.norm(mic_positions - source_position, axis=1)
    delays = distances / c
    return delays

# Apply delays to generate signals received at each mic
delays = compute_delays(mic_positions, source_position)
received_signals = np.zeros((len(mic_positions), len(t)))

for m in range(len(mic_positions)):
    delay_samples = int(np.round(delays[m] * fs))
    if delay_samples < len(t):
        received_signals[m, delay_samples:] = signal[:len(t) - delay_samples]

# Convert to frequency domain
freq_bin = int(f * len(t) / fs)
signal_fft = np.fft.fft(received_signals, axis=1)
signal_fft = signal_fft[:, freq_bin]

# Beamforming grid
x_grid = np.linspace(0, 2.0, 100)
y_grid = np.linspace(0, 2.0, 100)
X, Y = np.meshgrid(x_grid, y_grid)
power_map = np.zeros_like(X)

# Frequency-domain beamforming
k = 2 * np.pi * f / c

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        scan_point = np.array([X[i, j], Y[i, j]])
        distances = np.linalg.norm(mic_positions - scan_point, axis=1)
        steering_vector = np.exp(-1j * k * distances)
        power = np.abs(np.dot(steering_vector.conj(), signal_fft))**2
        power_map[i, j] = power

# Normalize and plot
power_map /= np.max(power_map)
plt.figure(figsize=(6, 5))
plt.pcolormesh(X, Y, power_map, shading='auto', cmap='jet')
plt.colorbar(label='Normalized Power')
plt.scatter(source_position[0], source_position[1], color='cyan', marker='x', label='True Source')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('2D Frequency-Domain Beamforming')
plt.legend()
plt.tight_layout()
plt.show()
