import pyaudio
import numpy as np
import wave
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import threading
import signal
import sys
from scipy.signal import butter, filtfilt
from scipy.fft import fft, ifft

CHANNELS = 6  # Canales por dispositivo
RATE = 48000  # Frecuencia de muestreo
CHUNK = int(0.2 * RATE)  # Tamaño del buffer en 200 ms
c = 343  # Velocidad del sonido en m/s
RECORD_SECONDS = 120000  # Tiempo de grabación

lowcut = 200.0
highcut = 3000.0

azimuth_range = np.arange(-180, 181, 5)
elevation_range = np.arange(15, 91, 4)

azimuth_range = np.arange(-180, 181, 10)
elevation_range = np.arange(5, 80, 8)

#r = [0.1, 0.15, 0.25, 0.3, 0.4, 0.6]
#r = [0.12, 0.2, 0.3, 0.35, 0.45, 0.65]
#r = [0.22, 0.3, 0.38, 0.43, 0.6, 0.73]
#h = [-1.1, -0.93, -0.77, -0.6, -0.4, -0.01]

#h = [1.1, 0.93, 0.77, 0.6, 0.4, 0.01]
#h = [1.23, 1.06, 0.87, 0.71, 0.53, 0.01]
#h = [0, -0.17, -0.33, -0.5, -0.7, -1.09]
h = [1.1, 0.93, 0.77, 0.6, 0.4, 0.01]
a = [0, 120, 240]
r = [0.12, 0.2, 0.3, 0.35, 0.45, 0.65]

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

wav_filenames = ['/Users/30068385/OneDrive - Western Sydney University/18ch 3D drone 19 09/18ch sync/device_1_sync.wav',# drone
                 '/Users/30068385/OneDrive - Western Sydney University/18ch 3D drone 19 09/18ch sync/device_2_sync.wav']

wav_filenames = ['/Users/30068385/OneDrive - Western Sydney University/test recordings/speaker 2 no filter pytho no ref 23 09/device_1_sync.wav', # speaker
                 '/Users/30068385/OneDrive - Western Sydney University/test recordings/speaker 2 no filter pytho no ref 23 09/device_2_sync.wav',
                 '/Users/30068385/OneDrive - Western Sydney University/test recordings/speaker 2 no filter pytho no ref 23 09/device_3_sync.wav']

wav_filenames = ['/Users/bjrn/OneDrive - Western Sydney University/18ch 3D drone 19 09/18ch sync/device_1_sync.wav',
                 '/Users/bjrn/OneDrive - Western Sydney University/18ch 3D drone 19 09/18ch sync/device_2_sync.wav']


buffers = [np.zeros((CHUNK, CHANNELS), dtype=np.int32) for _ in range(3)]

# beamforming



def beamform_frequency(signal_data, mic_positions, azimuth_range, elevation_range, RATE, c):
    num_samples = signal_data.shape[0]
    num_mics = signal_data.shape[1]
    energy = np.zeros((len(azimuth_range), len(elevation_range)))

    # FFT de las señales
    signal_fft = fft(signal_data, axis=0)
    freqs = np.fft.fftfreq(num_samples, d=1.0/RATE)

    for az_idx, theta in enumerate(azimuth_range):
        azimuth_rad = np.radians(theta)

        for el_idx, phi in enumerate(elevation_range):
            elevation_rad = np.radians(phi)

            # Vector de dirección 3D
            direction_vector = np.array([
                np.cos(elevation_rad) * np.cos(azimuth_rad),
                np.cos(elevation_rad) * np.sin(azimuth_rad),
                np.sin(elevation_rad)
            ])

            # Calcular los retrasos para cada micrófono
            delays = (np.dot(mic_positions, direction_vector) / c)

            # Aplicar los retrasos en el dominio de la frecuencia ajustando las fases
            output_signal_fft = np.zeros_like(signal_fft, dtype=np.complex128)
            for i in range(num_mics):
                # Calculamos el cambio de fase para cada frecuencia y micrófono
                phase_shift = np.exp(-2j * np.pi * freqs * delays[i])
                output_signal_fft[:, i] = signal_fft[:, i] * phase_shift

            # Sumar las señales compensadas por el retraso (en el dominio de la frecuencia)
            output_signal_combined_fft = np.sum(output_signal_fft, axis=1)

            # Convertir de vuelta al dominio del tiempo
            output_signal_combined = ifft(output_signal_combined_fft)

            # Calcular la energía total para esta dirección
            energy[az_idx, el_idx] = np.sum(np.abs(output_signal_combined) ** 2)

    return energy


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



plt.ion()
fig, ax = plt.subplots(figsize=(6, 3))
cax = ax.imshow(np.zeros((len(elevation_range), len(azimuth_range))), extent=[azimuth_range[0], azimuth_range[-1], elevation_range[0], elevation_range[-1]], origin='lower', aspect='auto', cmap='viridis')
fig.colorbar(cax, ax=ax, label='Energy')
ax.set_xlabel('Azimut')
ax.set_ylabel('Elevation')
ax.set_title('beamforming enery')

# max energy point
max_energy_marker, = ax.plot([], [], 'ro')
max_energy_text = ax.text(0, 0, '', color='white', fontsize=12, ha='center')


wav_files = [wave.open(filename, 'rb') for filename in wav_filenames]

skip_seconds = 51
#skip_seconds = 0
for wav_file in wav_files:
    skip_wav_seconds(wav_file, skip_seconds, RATE)

try:
    for time_idx in range(0, int(RATE / CHUNK * RECORD_SECONDS)):

        # Leer el siguiente bloque de datos para cada dispositivo
        for i, wav_file in enumerate(wav_files):
            block = read_wav_block(wav_file, CHUNK)
            if block is None:
                break  # Si se alcanzó el final del archivo
            buffers[i] = block

        combined_signal = np.hstack(buffers)

        # filtering
        filtered_signal = apply_bandpass_filter(combined_signal, lowcut, highcut, RATE)
        num_samples = filtered_signal.shape[0]

        energy = beamform_time(filtered_signal, mic_positions, azimuth_range, elevation_range, RATE, c)
        #energy = beamform_time(combined_signal, mic_positions, azimuth_range, elevation_range, RATE, c)

        # Realizar beamforming en el dominio de la frecuencia
        #energy = beamform_frequency(filtered_signal, mic_positions, azimuth_range, elevation_range, RATE, c)

        # Encontrar el índice de la máxima energía
        max_energy_idx = np.unravel_index(np.argmax(energy), energy.shape)
        estimated_azimuth = azimuth_range[max_energy_idx[0]]
        estimated_elevation = elevation_range[max_energy_idx[1]]

        # Calcular el tiempo actual de la muestra de audio
        current_time = calculate_time(time_idx, CHUNK, RATE)

        # Imprimir el ángulo estimado y el tiempo
        #print(f"Tiempo: {current_time:.2f} s - Ángulo estimado: Azimut = {estimated_azimuth:.2f}°, Elevación = {estimated_elevation:.2f}°")

        # Actualizar los datos del mapa de calor
        cax.set_data(energy.T)
        cax.set_clim(vmin=np.min(energy), vmax=np.max(energy))  # Actualizar los límites del color

        # Actualizar la posición del marcador de máxima energía
        max_energy_marker.set_data([estimated_azimuth], [estimated_elevation])

        # Actualizar la posición del texto con las coordenadas
        max_energy_text.set_position((estimated_azimuth, estimated_elevation))
        max_energy_text.set_text(f"Az: {estimated_azimuth:.1f}°, El: {estimated_elevation:.1f}°")

        fig.canvas.draw()
        fig.canvas.flush_events()

    print("Simulación completada.")
finally:
    for wav_file in wav_files:
        wav_file.close()