import numpy as np
import wave
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from pyproj import Proj, Transformer
import pandas as pd

CHANNELS = 6  # Canales por dispositivo
RATE = 48000  # Frecuencia de muestreo
BUFFER = 0.1 # Buffer time 100 ms
CHUNK = int(BUFFER * RATE)
c = 343  # Velocidad del sonido en m/s
RECORD_SECONDS = 120000  # Tiempo de grabación

lowcut = 400.0
highcut = 8000.0

azimuth_range = np.arange(-180, 181, 5)
elevation_range = np.arange(0, 91, 5)

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
    nyquist = 0.5 * rate  # Frecuencia de Nyquist
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(signal_data, lowcut, highcut, rate, order=5):
    b, a = butter_bandpass(lowcut, highcut, rate, order=order)
    filtered_signal = filtfilt(b, a, signal_data, axis=0)  # Aplicar filtro a lo largo de la señal en cada canal
    return filtered_signal

# Read WAV files
def read_wav_block(wav_file, chunk_size):
    data = wav_file.readframes(chunk_size)
    if len(data) == 0:
        return None
    signal_data = np.frombuffer(data, dtype=np.int32)
    return np.reshape(signal_data, (-1, CHANNELS))


# Cargar los archivos CSV
ref_file_path = '/Users/30068385/OneDrive - Western Sydney University/flight records/DJIFlightRecord_2024-09-24_[13-07-49].csv'
file_path_flight = '/Users/30068385/OneDrive - Western Sydney University/flight records/DJIFlightRecord_2024-09-24_[13-23-48].csv'

# Leer el archivo de referencia y de vuelo
ref_data = pd.read_csv(ref_file_path, skiprows=1, delimiter=',', low_memory=False)
flight_data = pd.read_csv(file_path_flight, skiprows=1, delimiter=',', low_memory=False)

# Extraer la posición de referencia (promedio de los valores válidos de latitud y longitud)
reference_latitude = ref_data['OSD.latitude'].dropna().astype(float).mean()
reference_longitude = ref_data['OSD.longitude'].dropna().astype(float).mean()

# Extraer las columnas necesarias: latitud, longitud, altura y tiempo
latitude_col = 'OSD.latitude'
longitude_col = 'OSD.longitude'
altitude_col = 'OSD.altitude [ft]'
time_col = 'OSD.flyTime'

# Obtener el índice inicial correspondiente al segundo deseado (por ejemplo, segundo 4)
start_time_seconds = 3  # Segundo deseado
samples_per_second = 10  # 100 ms por muestra significa 10 muestras por segundo
start_index = start_time_seconds * samples_per_second

# Reajustar el dataframe para empezar desde el índice deseado
flight_data = flight_data.iloc[start_index:].reset_index(drop=True)

# Filtrar las filas con datos válidos
flight_data = flight_data[[latitude_col, longitude_col, altitude_col, time_col]].dropna()

# Convertir la altitud de pies a metros
flight_data[altitude_col] = flight_data[altitude_col] * 0.3048

# Obtener la altitud inicial del vuelo para usarla en la referencia de elevación
initial_altitude = flight_data[altitude_col].iloc[0]

# Configurar la proyección UTM para convertir las coordenadas geográficas a metros
transformer = Transformer.from_crs('epsg:4326', 'epsg:32756', always_xy=True)  # Ajustar la zona UTM 56 south

# Convertir las coordenadas de referencia a metros
ref_x, ref_y = transformer.transform(reference_longitude, reference_latitude)

# Crear nuevas columnas para las coordenadas en metros en el dataframe de vuelo
flight_data['X_meters'], flight_data['Y_meters'] = transformer.transform(
    flight_data[longitude_col].values,
    flight_data[latitude_col].values
)

# Función para calcular la distancia horizontal entre dos puntos en coordenadas cartesianas
def calculate_horizontal_distance_meters(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Función para calcular el azimuth usando coordenadas en metros
def calculate_azimuth_meters(x1, y1, x2, y2):
    azimuth = np.degrees(np.arctan2(y2 - y1, x2 - x1))
    return azimuth

# Función para calcular la elevación utilizando la distancia horizontal en metros
def calculate_elevation_meters(altitude, x1, y1, x2, y2, reference_altitude):
    horizontal_distance = calculate_horizontal_distance_meters(x1, y1, x2, y2)
    relative_altitude = altitude - reference_altitude  # Calcular la altura relativa respecto a la referencia
    return np.degrees(np.arctan2(relative_altitude, horizontal_distance))

# Calcular los valores iniciales de azimuth y elevación
initial_azimuth = calculate_azimuth_meters(ref_x, ref_y,
                                           flight_data.iloc[0]['X_meters'],
                                           flight_data.iloc[0]['Y_meters'])

initial_elevation = calculate_elevation_meters(flight_data.iloc[0][altitude_col], ref_x, ref_y,
                                               flight_data.iloc[0]['X_meters'], flight_data.iloc[0]['Y_meters'],
                                               initial_altitude)
def update(frame):
    x = flight_data.iloc[frame]['X_meters']
    y = flight_data.iloc[frame]['Y_meters']
    altitude = flight_data.iloc[frame][altitude_col]

    # Calcular el azimuth y la elevación relativos al punto de referencia
    #azimuth = -1*(calculate_azimuth_meters(ref_x, ref_y, x, y) - initial_azimuth + 5)
    azimuth = -1*(calculate_azimuth_meters(ref_x, ref_y, x, y) - initial_azimuth)
    elevation = calculate_elevation_meters(altitude, ref_x, ref_y, x, y, initial_altitude) - initial_elevation

    point.set_data([azimuth], [elevation])

    plt.draw()
    #plt.pause(0.001)


# Configuración inicial de la visualización
plt.ion()
fig, ax = plt.subplots(figsize=(12, 3))
cax = ax.imshow(np.zeros((len(elevation_range), len(azimuth_range))),
                extent=[azimuth_range[0], azimuth_range[-1], elevation_range[0], elevation_range[-1]],
                origin='lower', aspect='auto', cmap='viridis')

# Punto que representa la posición del dron
point, = ax.plot([], [], 'bo', markersize=5)  # Crear el punto azul
fig.colorbar(cax, ax=ax, label='Energy')
ax.set_xlabel('Azimut')
ax.set_ylabel('Elevation')
ax.set_title('Beamforming Energy')

# Marcador de la máxima energía
max_energy_marker, = ax.plot([], [], 'ro')
max_energy_text = ax.text(0, 0, '', color='white', fontsize=12, ha='center')

wav_files = [wave.open(filename, 'rb') for filename in wav_filenames]

skip_seconds = 115

for wav_file in wav_files:
    skip_wav_seconds(wav_file, skip_seconds, RATE)


try:
    for time_idx, i in zip(range(0, int(RATE / CHUNK * RECORD_SECONDS)), range(len(flight_data))):

        # Leer el siguiente bloque de datos para cada dispositivo
        for j, wav_file in enumerate(wav_files):
            block = read_wav_block(wav_file, CHUNK)
            if block is None:
                break  # Si se alcanzó el final del archivo
            buffers[j] = block

        combined_signal = np.hstack(buffers)

        # Filtrar el señal
        filtered_signal = apply_bandpass_filter(combined_signal, lowcut, highcut, RATE)

        # Beamforming
        energy = beamform_time(filtered_signal, mic_positions, azimuth_range, elevation_range, RATE, c)

        # Encontrar el índice de la máxima energía
        max_energy_idx = np.unravel_index(np.argmax(energy), energy.shape)
        estimated_azimuth = azimuth_range[max_energy_idx[0]]
        estimated_elevation = elevation_range[max_energy_idx[1]]

        # Calcular el tiempo actual de la muestra de audio
        current_time = calculate_time(time_idx, CHUNK, RATE)

        # Actualizar el punto azul con la nueva posición del dron
        update(i)

        # Imprimir el ángulo estimado y el tiempo
        print(f"Tiempo: {current_time + skip_seconds:.2f} s - Tiempo Log: {time_col} - Ángulo estimado: Azimut = {estimated_azimuth:.2f}°, Elevación = {estimated_elevation:.2f}°")

        # Actualizar la posición del marcador de máxima energía
        max_energy_marker.set_data([estimated_azimuth], [estimated_elevation])

        # Actualizar la posición del texto con las coordenadas
        max_energy_text.set_position((estimated_azimuth, estimated_elevation))
        max_energy_text.set_text(f"Az: {estimated_azimuth:.1f}°, El: {estimated_elevation:.1f}°")

        # Actualizar los datos del mapa de calor
        cax.set_data(energy.T)
        cax.set_clim(vmin=np.min(energy), vmax=np.max(energy))  # Actualizar los límites del color

        fig.canvas.draw()
        fig.canvas.flush_events()

    print("Simulación completada.")
    plt.ioff()
    plt.show()
finally:
    for wav_file in wav_files:
        wav_file.close()