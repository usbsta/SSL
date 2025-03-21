import numpy as np
import wave
import matplotlib.pyplot as plt

# Parámetros de simulación
CHANNELS = 6
RATE = 48000
CHUNK = int(0.2 * RATE)
RECORD_SECONDS = 1200000  # Duración en segundos
c = 343

r = [0.12, 0.2]

h = [-1.1, -0.93]
#h = [-0.25, -0.5]
a = [0, 120, 240]

mic_positions = np.array([
    [r[0] * np.cos(np.radians(a[0])), r[0] * np.sin(np.radians(a[0])), h[0]],  # Mic 1
    [r[0] * np.cos(np.radians(a[1])), r[0] * np.sin(np.radians(a[1])), h[0]],  # Mic 2
    [r[0] * np.cos(np.radians(a[2])), r[0] * np.sin(np.radians(a[2])), h[0]],  # Mic 3
    [r[1] * np.cos(np.radians(a[0])), r[1] * np.sin(np.radians(a[0])), h[1]],  # Mic 4
    [r[1] * np.cos(np.radians(a[1])), r[1] * np.sin(np.radians(a[1])), h[1]],  # Mic 5
    [r[1] * np.cos(np.radians(a[2])), r[1] * np.sin(np.radians(a[2])), h[1]]  # Mic 6
])

azimuth_range = np.arange(0, 360, 5)
elevation_range = np.arange(5, 91, 5)

wav_filenames = ['/Users/30068385/OneDrive - Western Sydney University/18ch_3D_outside/device_1_sync.wav']


def np_shift(arr, num_shift):
    if num_shift > 0:
        return np.concatenate([np.zeros(num_shift), arr[:-num_shift]])
    elif num_shift < 0:
        return np.concatenate([arr[-num_shift:], np.zeros(-num_shift)])
    else:
        return arr


def beamform_time_ref_mic(signal_data, mic_positions, azimuth_range, elevation_range, RATE, c):
    num_samples = signal_data.shape[0]
    energy = np.zeros((len(azimuth_range), len(elevation_range)))

    # Micrófono de referencia es el micrófono 1, su posición es mic_positions[0]
    ref_mic_position = mic_positions[0]

    for az_idx, theta in enumerate(azimuth_range):
        azimuth_rad = np.radians(theta)

        for el_idx, phi in enumerate(elevation_range):
            elevation_rad = np.radians(phi)

            # Vector de dirección en 3D
            direction_vector = np.array([
                np.cos(elevation_rad) * np.cos(azimuth_rad),
                np.cos(elevation_rad) * np.sin(azimuth_rad),
                np.sin(elevation_rad)
            ])

            # Cálculo de los retrasos en tiempo para cada micrófono con respecto al micrófono 1
            ref_delay = np.dot(ref_mic_position, direction_vector) / c
            delays = np.dot(mic_positions, direction_vector) / c - ref_delay

            # Aplicar los retrasos alineando las señales
            output_signal = np.zeros(num_samples)
            for i, delay in enumerate(delays):
                delay_samples = int(np.round(delay * RATE))  # Convertir el retraso en muestras

                # Usamos np.roll para desplazar las señales, independiente de si el delay es positivo o negativo
                shifted_signal = np.roll(signal_data[:, i], -delay_samples)

                # Sumamos la señal desplazada al resultado final
                output_signal += shifted_signal
            output_signal /= signal_data.shape[1]  # Normalizar por el número de micrófonos
            energy[az_idx, el_idx] = np.sum(output_signal ** 2)
    return energy


def read_wav_block(wav_file, chunk_size):
    data = wav_file.readframes(chunk_size)
    if len(data) == 0:
        return None
    signal_data = np.frombuffer(data, dtype=np.int32)
    return np.reshape(signal_data, (-1, CHANNELS))


def skip_wav_seconds(wav_file, seconds, rate):
    frames_to_skip = int(seconds * rate)
    wav_file.setpos(frames_to_skip)


plt.ion()
fig, ax = plt.subplots(figsize=(8, 6))
cax = ax.imshow(np.zeros((len(elevation_range), len(azimuth_range))), extent=[azimuth_range[0], azimuth_range[-1], elevation_range[0], elevation_range[-1]], origin='lower', aspect='auto', cmap='viridis')
fig.colorbar(cax, ax=ax, label='Energía')
ax.set_xlabel('Azimut (grados)')
ax.set_ylabel('Elevación (grados)')
ax.set_title('Energía del beamforming')

# Identificador inicial en el plot
max_energy_marker, = ax.plot([], [], 'ro')  # Marcador de color rojo en la máxima energía
max_energy_text = ax.text(0, 0, '', color='white', fontsize=12, ha='center')


wav_files = [wave.open(filename, 'rb') for filename in wav_filenames]


skip_seconds = 63
for wav_file in wav_files:
    skip_wav_seconds(wav_file, skip_seconds, RATE)

try:
    for time_idx in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        # Leer el siguiente bloque de datos desde el archivo WAV
        signal_data = read_wav_block(wav_files[0], CHUNK)

        if signal_data is None:
            break  # Salir del bucle si no hay más datos en el archivo WAV

        # Calcular la energía utilizando beamforming en el dominio del tiempo
        energy = beamform_time_ref_mic(signal_data, mic_positions, azimuth_range, elevation_range, RATE, c)

        max_energy_idx = np.unravel_index(np.argmax(energy), energy.shape)
        estimated_azimuth = azimuth_range[max_energy_idx[0]]
        estimated_elevation = elevation_range[max_energy_idx[1]]
        print(f"Ángulo estimado: Azimut = {estimated_azimuth:.2f}°, Elevación = {estimated_elevation:.2f}°")

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
