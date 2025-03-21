import pyaudio
import numpy as np
import wave
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import threading
import signal
import sys
from scipy.signal import butter, filtfilt

CHANNELS = 6  # Canales por dispositivo
RATE = 48000  # Frecuencia de muestreo
CHUNK = int(1 * RATE)  # Tamaño del buffer en 100 ms
c = 343  # Velocidad del sonido en m/s
RECORD_SECONDS = 120000  # Tiempo de grabación

lowcut = 400.0
highcut = 8000.0

azimuth_range = np.arange(-180, 181, 150)
elevation_range = np.arange(10, 91, 30)

a = [0, -120, -240]
# config 1 equidistance
h = [1.12, 0.92, 0.77, 0.6, 0.42, 0.02]
r = [0.1, 0.17, 0.25, 0.32, 0.42, 0.63]

# config 2 augmented
#h = [1.12, 1.02, 0.87, 0.68, 0.47, 0.02]
#r = [0.1, 0.16, 0.23, 0.29, 0.43, 0.63]


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
    [r[5] * np.cos(np.radians(a[2])), r[5] * np.sin(np.radians(a[2])), h[5]] # Mic 18
])

# Nombres de los archivos WAV (para la opción de simulación)

wav_filenames = ['/Users/bjrn/OneDrive - Western Sydney University/18ch 3D drone 19 09/18ch sync/device_1_sync.wav',
                 '/Users/bjrn/OneDrive - Western Sydney University/18ch 3D drone 19 09/18ch sync/device_2_sync.wav']

wav_filenames = ['/Users/30068385/OneDrive - Western Sydney University/18ch 3D drone 19 09/18ch sync/device_1_sync.wav',# drone
                 '/Users/30068385/OneDrive - Western Sydney University/18ch 3D drone 19 09/18ch sync/device_2_sync.wav']

wav_filenames = ['/Users/30068385/OneDrive - Western Sydney University/test recordings/speaker 2 no filter pytho no ref 23 09/device_1_sync.wav', # speaker
                 '/Users/30068385/OneDrive - Western Sydney University/test recordings/speaker 2 no filter pytho no ref 23 09/device_2_sync.wav',
                 '/Users/30068385/OneDrive - Western Sydney University/test recordings/speaker 2 no filter pytho no ref 23 09/device_3_sync.wav']

wav_filenames = ['/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/24 sep/equi/device_1_sync.wav',
                 '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/24 sep/equi/device_2_sync.wav',
                 '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/24 sep/equi/device_3_sync.wav']


buffers = [np.zeros((CHUNK, CHANNELS), dtype=np.int32) for _ in range(3)]



# Initialize WAV files for each (azimuth, elevation) pair
beamformed_wav_files = {}
for theta in azimuth_range:
    for phi in elevation_range:
        filename = f"beamformed_az{theta}_el{phi}.wav"
        wav_file = wave.open(filename, 'wb')
        wav_file.setnchannels(1)  # Mono output
        wav_file.setsampwidth(2)  # 16-bit audio
        wav_file.setframerate(RATE)
        beamformed_wav_files[(theta, phi)] = wav_file


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
    nyquist = 0.5 * rate  # Frecuencia de Nyquist
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a


def apply_bandpass_filter(signal_data, lowcut, highcut, rate, order=5):
    b, a = butter_bandpass(lowcut, highcut, rate, order=order)
    filtered_signal = filtfilt(b, a, signal_data, axis=0)  # Aplicar filtro a lo largo de la señal en cada canal
    return filtered_signal

# Visualization setup
plt.ion()
fig, ax = plt.subplots(figsize=(12, 3))
cax = ax.imshow(np.zeros((len(elevation_range), len(azimuth_range))), extent=[azimuth_range[0], azimuth_range[-1], elevation_range[0], elevation_range[-1]], origin='lower', aspect='auto', cmap='viridis')
fig.colorbar(cax, ax=ax, label='Energy')
ax.set_xlabel('Azimut')
ax.set_ylabel('Elevation')
ax.set_title('beamforming enery')

# max energy point
max_energy_marker, = ax.plot([], [], 'ro')
max_energy_text = ax.text(0, 0, '', color='white', fontsize=12, ha='center')


wav_files = [wave.open(filename, 'rb') for filename in wav_filenames]

skip_seconds = 115
skip_seconds = 530
#skip_seconds = 630

for wav_file in wav_files:
    skip_wav_seconds(wav_file, skip_seconds, RATE)

try:
    for time_idx in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        finished = False

        # Read the next block of data for each device
        for i, wav_file in enumerate(wav_files):
            block = read_wav_block(wav_file, CHUNK)
            if block is None:
                finished = True
                break
            buffers[i] = block

        if finished:
            print("End of audio file.")
            break

        combined_signal = np.hstack(buffers)

        # Apply bandpass filter
        filtered_signal = apply_bandpass_filter(combined_signal, lowcut, highcut, RATE)
        num_samples = filtered_signal.shape[0]

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




                # Apply delays
                output_signal = np.zeros(num_samples)
                for i, delay in enumerate(delays):
                    delay_samples = int(np.round(delay * RATE))
                    signal_shifted = np.roll(filtered_signal[:, i], delay_samples)
                    output_signal += signal_shifted

                output_signal /= filtered_signal.shape[1]  # Normalize amplitude

                # Normalizar la señal de salida después de sumar
                max_amplitude = np.max(np.abs(output_signal))
                if max_amplitude > 0:
                    output_signal /= max_amplitude  # Normalizar al rango -1.0 a 1.0
                else:
                    output_signal = np.zeros_like(output_signal)

                # Compute energy
                energy[az_idx, el_idx] = np.sum(output_signal ** 2)

                # Prepare output signal for WAV file

                output_signal_int16 = np.int16(output_signal * 32767)

                # Write to WAV file
                wav_file = beamformed_wav_files[(theta, phi)]
                wav_file.writeframes(output_signal_int16.tobytes())

        # Find the index of the maximum energy
        max_energy_idx = np.unravel_index(np.argmax(energy), energy.shape)
        estimated_azimuth = azimuth_range[max_energy_idx[0]]
        estimated_elevation = elevation_range[max_energy_idx[1]]

        current_time = calculate_time(time_idx, CHUNK, RATE)

        # Print the estimated angle and time
        print(f"Time: {current_time + skip_seconds:.2f} s - Estimated Angle: Azimuth = {estimated_azimuth:.2f}°, Elevation = {estimated_elevation:.2f}°")

        # Update heatmap visualization
        cax.set_data(energy.T)
        cax.set_clim(vmin=np.min(energy), vmax=np.max(energy))
        max_energy_marker.set_data([estimated_azimuth], [estimated_elevation])
        max_energy_text.set_position((estimated_azimuth, estimated_elevation))
        max_energy_text.set_text(f"Az: {estimated_azimuth:.1f}°, El: {estimated_elevation:.1f}°")
        fig.canvas.draw()
        fig.canvas.flush_events()

    print("Simulation completed.")

finally:
    # Close all WAV files
    for wav_file in beamformed_wav_files.values():
        wav_file.close()
    for wav_file in wav_files:
        wav_file.close()
