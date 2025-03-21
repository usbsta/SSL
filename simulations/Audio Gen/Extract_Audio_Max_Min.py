import numpy as np
import wave
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import threading
import signal
import sys
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d


# Buffers para acumular las señales de salida
max_energy_signal = []
min_energy_signal = []

CHANNELS = 6  # Canales por dispositivo
RATE = 48000  # Frecuencia de muestreo
CHUNK = int(0.6 * RATE)  # Tamaño del buffer en 100 ms
c = 343  # Velocidad del sonido en m/s
RECORD_SECONDS = 120000  # Tiempo de grabación

lowcut = 100.0
highcut = 8000.0

azimuth_range = np.arange(-180, 181, 3)
elevation_range = np.arange(10, 91, 3)

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
"""
beamformed_wav_files = {}
for theta in azimuth_range:
    for phi in elevation_range:
        filename = f"beamformed_az{theta}_el{phi}.wav"
        wav_file = wave.open(filename, 'wb')
        wav_file.setnchannels(1)  # Mono output
        wav_file.setsampwidth(2)  # 16-bit audio
        wav_file.setframerate(RATE)
        beamformed_wav_files[(theta, phi)] = wav_file
"""


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

def apply_delay(signal, delay, rate):
    """
    Aplica un retraso fraccionario a una señal usando interpolación.
    """
    num_samples = len(signal)
    t = np.arange(num_samples)
    delayed_t = t - delay * rate  # retraso en muestras
    f = interp1d(t, signal, kind='linear', fill_value=0.0, bounds_error=False)
    delayed_signal = f(delayed_t)
    return delayed_signal

skip_seconds = 115
skip_seconds = 530
#skip_seconds = 630

for wav_file in wav_files:
    skip_wav_seconds(wav_file, skip_seconds, RATE)

try:
    for time_idx in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        finished = False

        # Leer el siguiente bloque de datos para cada dispositivo
        for i, wav_file in enumerate(wav_files):
            block = read_wav_block(wav_file, CHUNK)
            if block is None:
                finished = True
                break
            buffers[i] = block

        if finished:
            print("Fin del archivo de audio.")
            break

        combined_signal = np.hstack(buffers)

        # Aplicar filtro de paso de banda
        filtered_signal = apply_bandpass_filter(combined_signal, lowcut, highcut, RATE)
        num_samples = filtered_signal.shape[0]

        energy = np.zeros((len(azimuth_range), len(elevation_range)))

        # Variables para almacenar las señales beamformed de máxima y mínima energía
        output_signal_max = None
        output_signal_min = None

        max_energy = -np.inf
        min_energy = np.inf

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

                delays = (np.dot(mic_positions, direction_vector) / c)

                # Aplicar retrasos fraccionarios
                output_signal = np.zeros(num_samples)
                for mic_idx, delay in enumerate(delays):
                    signal_delayed = apply_delay(filtered_signal[:, mic_idx], delay, RATE)
                    output_signal += signal_delayed

                # Calcular energía
                energy_value = np.sum(output_signal ** 2)
                energy[az_idx, el_idx] = energy_value

                # Verificar si es la energía máxima
                if energy_value > max_energy:
                    max_energy = energy_value
                    output_signal_max = output_signal.copy()
                    max_direction = (theta, phi)

                # Verificar si es la energía mínima
                if energy_value < min_energy:
                    min_energy = energy_value
                    output_signal_min = output_signal.copy()
                    min_direction = (theta, phi)

        # Normalizar y acumular las señales de salida
        if output_signal_max is not None:
            max_amp = np.max(np.abs(output_signal_max))
            if max_amp > 0:
                output_signal_max /= max_amp
            max_energy_signal.append(output_signal_max)

        if output_signal_min is not None:
            min_amp = np.max(np.abs(output_signal_min))
            if min_amp > 0:
                output_signal_min /= min_amp
            min_energy_signal.append(output_signal_min)

        # Calcular el tiempo actual
        current_time = calculate_time(time_idx, CHUNK, RATE)

        # Imprimir las direcciones de máxima y mínima energía
        print(f"Tiempo: {current_time + skip_seconds:.2f} s - Máxima energía en Azimut = {max_direction[0]:.2f}°, Elevación = {max_direction[1]:.2f}°")
        print(f"Tiempo: {current_time + skip_seconds:.2f} s - Mínima energía en Azimut = {min_direction[0]:.2f}°, Elevación = {min_direction[1]:.2f}°")

    print("Procesamiento completado.")

    # Combinar los buffers en una sola señal
    max_energy_signal_array = np.concatenate(max_energy_signal)
    min_energy_signal_array = np.concatenate(min_energy_signal)

    # Escalar las señales al rango adecuado para int16
    max_energy_signal_int16 = np.int16(max_energy_signal_array * 32767)
    min_energy_signal_int16 = np.int16(min_energy_signal_array * 32767)

    # Escribir en archivos WAV
    import wave

    # Archivo para energía máxima
    with wave.open('beamformed_max_energy.wav', 'wb') as max_wav:
        max_wav.setnchannels(1)  # Mono
        max_wav.setsampwidth(2)  # 16 bits
        max_wav.setframerate(RATE)
        max_wav.writeframes(max_energy_signal_int16.tobytes())

    # Archivo para energía mínima
    with wave.open('beamformed_min_energy.wav', 'wb') as min_wav:
        min_wav.setnchannels(1)  # Mono
        min_wav.setsampwidth(2)  # 16 bits
        min_wav.setframerate(RATE)
        min_wav.writeframes(min_energy_signal_int16.tobytes())

finally:
    for wav_file in wav_files:
        wav_file.close()