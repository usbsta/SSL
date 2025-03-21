import numpy as np
import wave
from scipy.signal import butter, filtfilt
import pyroomacoustics as pra
import matplotlib.pyplot as plt

CHANNELS = 6  # Canales por dispositivo
RATE = 48000  # Frecuencia de muestreo
CHUNK = int(0.2 * RATE)  # Tamaño del buffer en 200 ms
c = 343  # Velocidad del sonido en m/s
RECORD_SECONDS = 120000  # Tiempo de grabación

lowcut = 400.0
highcut = 8000.0

azimuth_range = np.arange(-180, 181, 5)
elevation_range = np.arange(10, 91, 5)

a = [0, -120, -240]
# Configuración 1 equidistante
h = [1.12, 0.92, 0.77, 0.6, 0.42, 0.02]
r = [0.1, 0.17, 0.25, 0.32, 0.42, 0.63]

mic_positions = np.array([
    [r[0] * np.cos(np.radians(a[0])), r[0] * np.sin(np.radians(a[0])), h[0]],  # Mic 1
    [r[0] * np.cos(np.radians(a[1])), r[0] * np.sin(np.radians(a[1])), h[0]],  # Mic 2
    [r[0] * np.cos(np.radians(a[2])), r[0] * np.sin(np.radians(a[2])), h[0]],  # Mic 3
    [r[1] * np.cos(np.radians(a[0])), r[1] * np.sin(np.radians(a[0])), h[1]],  # Mic 4
    [r[1] * np.cos(np.radians(a[1])), r[1] * np.sin(np.radians(a[1])), h[1]],  # Mic 5
    [r[1] * np.cos(np.radians(a[2])), r[1] * np.sin(np.radians(a[2])), h[1]],  # Mic 6
    [r[2] * np.cos(np.radians(a[0])), r[2] * np.sin(np.radians(a[0])), h[2]],  # Mic 7
    [r[2] * np.cos(np.radians(a[1])), r[2] * np.sin(np.radians(a[1])), h[2]],  # Mic 8
    [r[2] * np.cos(np.radians(a[2])), r[2] * np.sin(np.radians(a[2])), h[2]],  # Mic 9
    [r[3] * np.cos(np.radians(a[0])), r[3] * np.sin(np.radians(a[0])), h[3]],  # Mic 10
    [r[3] * np.cos(np.radians(a[1])), r[3] * np.sin(np.radians(a[1])), h[3]],  # Mic 11
    [r[3] * np.cos(np.radians(a[2])), r[3] * np.sin(np.radians(a[2])), h[3]],  # Mic 12
    [r[4] * np.cos(np.radians(a[0])), r[4] * np.sin(np.radians(a[0])), h[4]],  # Mic 13
    [r[4] * np.cos(np.radians(a[1])), r[4] * np.sin(np.radians(a[1])), h[4]],  # Mic 14
    [r[4] * np.cos(np.radians(a[2])), r[4] * np.sin(np.radians(a[2])), h[4]],  # Mic 15
    [r[5] * np.cos(np.radians(a[0])), r[5] * np.sin(np.radians(a[0])), h[5]],  # Mic 16
    [r[5] * np.cos(np.radians(a[1])), r[5] * np.sin(np.radians(a[1])), h[5]],  # Mic 17
    [r[5] * np.cos(np.radians(a[2])), r[5] * np.sin(np.radians(a[2])), h[5]]  # Mic 18
]).T  # Transpose to match pyroomacoustics format

wav_filenames = ['/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/24 sep/equi/device_1_sync.wav',
                 '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/24 sep/equi/device_2_sync.wav',
                 '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/24 sep/equi/device_3_sync.wav']


buffers = [np.zeros((CHUNK, CHANNELS), dtype=np.int32) for _ in range(3)]

# beamforming
def beamform_time(signal_data, mic_positions, azimuth_range, elevation_range, RATE, c):
    num_samples = signal_data.shape[0]
    energy = np.zeros((len(azimuth_range), len(elevation_range)))

    for az_idx, theta in enumerate(azimuth_range):
        azimuth_rad = np.radians(theta)

        for el_idx, phi in enumerate(elevation_range):
            elevation_rad = np.radians(phi)

            # 3D direction vector
            direction_vector = np.array([
                np.cos(elevation_rad) * np.cos(azimuth_rad),
                np.cos(elevation_rad) * np.sin(azimuth_rad),
                np.sin(elevation_rad)
            ])

            delays = (np.dot(mic_positions, direction_vector) / c)

            # aplying delays
            output_signal = np.zeros(num_samples)
            for i, delay in enumerate(delays):
                delay_samples = int(np.round(delay * RATE))
                signal_shifted = np.roll(signal_data[:, i], delay_samples)
                output_signal += signal_shifted

            output_signal /= signal_data.shape[1] # normalize amplitud with num of mics
            energy[az_idx, el_idx] = np.sum(output_signal ** 2)
    return energy

def calculate_time(time_idx, chunk_size, rate):
    # Calcular el tiempo actual en segundos
    time_seconds = (time_idx * chunk_size) / rate
    return time_seconds

def read_wav_block(wav_file, chunk_size):
    data = wav_file.readframes(chunk_size)
    if len(data) == 0:
        return None
    signal_data = np.frombuffer(data, dtype=np.int32)
    return np.reshape(signal_data, (-1, CHANNELS))


def skip_wav_seconds(wav_file, seconds, rate):
    frames_to_skip = int(seconds * rate)
    wav_file.setpos(frames_to_skip)

# band pass design
def butter_bandpass(lowcut, highcut, rate, order=5):
    nyquist = 0.5 * rate
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a


def apply_bandpass_filter(signal_data, lowcut, highcut, rate, order=5):
    b, a = butter_bandpass(lowcut, highcut, rate, order=order)
    filtered_signal = filtfilt(b, a, signal_data, axis=0)
    return filtered_signal


# SRP-PHAT localization
def srp_phat_localization(filtered_signal, mic_positions, rate, c):
    doa = pra.doa.Grid3D(mic_positions, rate, c, n_grid=360)
    doa.locate_sources(filtered_signal.T)
    return doa.grid.azimuth, doa.grid.colatitude, doa.grid.energy


# Read WAV files
def read_wav_block(wav_file, chunk_size):
    data = wav_file.readframes(chunk_size)
    if len(data) == 0:
        return None
    signal_data = np.frombuffer(data, dtype=np.int32)
    return np.reshape(signal_data, (-1, CHANNELS))


# Visualization setup
plt.ion()
fig, ax = plt.subplots(figsize=(12, 3))
cax = ax.imshow(np.zeros((len(elevation_range), len(azimuth_range))),
                extent=[azimuth_range[0], azimuth_range[-1], elevation_range[0], elevation_range[-1]], origin='lower',
                aspect='auto', cmap='viridis')
fig.colorbar(cax, ax=ax, label='Energy')
ax.set_xlabel('Azimuth')
ax.set_ylabel('Elevation')
ax.set_title('SRP-PHAT Beamforming Energy')

max_energy_marker, = ax.plot([], [], 'ro')
max_energy_text = ax.text(0, 0, '', color='white', fontsize=12, ha='center')

wav_files = [wave.open(filename, 'rb') for filename in wav_filenames]

skip_seconds = 115

for wav_file in wav_files:
    skip_wav_seconds(wav_file, skip_seconds, RATE)

try:
    for time_idx in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        buffers = []

        # Leer el siguiente bloque de datos para cada dispositivo
        for i, wav_file in enumerate(wav_files):
            block = read_wav_block(wav_file, CHUNK)
            if block is None:
                break  # Si se alcanzó el final del archivo
            buffers[i] = block

        combined_signal = np.hstack(buffers)

        # Apply bandpass filtering
        filtered_signal = apply_bandpass_filter(combined_signal, lowcut, highcut, RATE)

        # Perform SRP-PHAT localization
        azimuths, elevations, energy = srp_phat_localization(filtered_signal, mic_positions, RATE, c)

        # Find maximum energy location
        max_energy_idx = np.argmax(energy)
        estimated_azimuth = azimuths[max_energy_idx]
        estimated_elevation = elevations[max_energy_idx]

        # Update visualization
        cax.set_data(energy.T)
        cax.set_clim(vmin=np.min(energy), vmax=np.max(energy))
        max_energy_marker.set_data([estimated_azimuth], [estimated_elevation])
        max_energy_text.set_position((estimated_azimuth, estimated_elevation))
        max_energy_text.set_text(f"Az: {estimated_azimuth:.1f}°, El: {estimated_elevation:.1f}°")

        fig.canvas.draw()
        fig.canvas.flush_events()

    print("Localization simulation completed.")
finally:
    for wav_file in wav_files:
        wav_file.close()
    plt.ioff()
    plt.show()
