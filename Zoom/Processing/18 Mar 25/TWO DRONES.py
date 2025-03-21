import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pyproj import Transformer

from audio_beamforming import beamform_time
from geo_utils import (wrap_angle, calculate_angle_difference,
                       calculate_azimuth_meters, calculate_elevation_meters,
                       calculate_total_distance_meters)
from io_utils import (read_wav_block, skip_wav_seconds, apply_bandpass_filter,
                      calculate_time, initialize_beamforming_params, open_wav_files)

# ----------------------- GENERAL CONFIGURATIONS -----------------------
CHANNELS = 6
RATE = 48000
BUFFER = 0.1
CHUNK = int(BUFFER * RATE)
c = 343
RECORD_SECONDS = 120000
lowcut = 200.0
highcut = 8000.0
skip_seconds = 225

azimuth_range = np.arange(-180, 181, 5)
elevation_range = np.arange(0, 91, 5)

mic_positions, delay_samples, num_mics = initialize_beamforming_params(azimuth_range, elevation_range, c, RATE)

wav_filenames = [
    '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/22 Nov 24/3/20241122_124721_device_1_nosync_part1.wav',
    '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/22 Nov 24/3/20241122_124723_device_2_nosync_part1.wav',
    '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/22 Nov 24/3/20241122_124725_device_3_nosync_part1.wav',
    '/Users/30068385/OneDrive - Western Sydney University/recordings/Drone/22 Nov 24/3/20241122_124727_device_4_nosync_part1.wav'
]

drones_config = [
    {
        'name': 'DJI Air 3',
        'ref_csv': '/Users/30068385/OneDrive - Western Sydney University/FlightRecord/DJI Air 3/CSV/22 Nov/Ref/Nov-22nd-2024-11-48AM-Flight-Airdata.csv',
        'flight_csv': '/Users/30068385/OneDrive - Western Sydney University/FlightRecord/DJI Air 3/CSV/22 Nov/3/Air_Nov-22nd-2024-12-31PM-Flight-Airdata.csv',
        'latitude_col': 'latitude',
        'altitude_col': 'altitude_above_seaLevel(feet)',
        'longitude_col': 'longitude',
        'time_col': 'time(millisecond)',
        'initial_azimuth': 6.0,
        'initial_elevation': 0.0,
        'start_index': 15,
    },
    {
        'name': 'DJI Inspire 1',
        'ref_csv': '/Users/30068385/OneDrive - Western Sydney University/FlightRecord/DJI Inspire 1/CSV/22 Nov/Ref/Nov-22nd-2024-11-45AM-Flight-Airdata.csv',
        'flight_csv': '/Users/30068385/OneDrive - Western Sydney University/FlightRecord/DJI Inspire 1/CSV/22 Nov/3/Inspire_Nov-22nd-2024-12-31PM-Flight-Airdata.csv',
        'latitude_col': 'latitude',
        'altitude_col': 'altitude_above_seaLevel(feet)',
        'longitude_col': 'longitude',
        'time_col': 'time(millisecond)',
        'initial_azimuth': -26.0,
        'initial_elevation': 0.0,
        'start_index': 1,
    }
]

def prepare_drone_data(drone_config):
    ref_csv = drone_config['ref_csv']
    flight_csv = drone_config['flight_csv']
    latitude_col = drone_config['latitude_col']
    longitude_col = drone_config['longitude_col']
    altitude_col = drone_config['altitude_col']
    time_col = drone_config['time_col']
    initial_azimuth = drone_config['initial_azimuth']
    initial_elevation = drone_config['initial_elevation']
    start_index = drone_config['start_index']

    ref_data = pd.read_csv(ref_csv, skiprows=0, delimiter=',', low_memory=False)
    flight_data = pd.read_csv(flight_csv, skiprows=0, delimiter=',', low_memory=False)

    reference_latitude = ref_data[latitude_col].dropna().astype(float).mean()
    reference_longitude = ref_data[longitude_col].dropna().astype(float).mean()

    flight_data = flight_data.iloc[start_index:].reset_index(drop=True)
    flight_data = flight_data[[latitude_col, longitude_col, altitude_col, time_col]].dropna()
    flight_data[altitude_col] = flight_data[altitude_col] * 0.3048
    initial_altitude = flight_data[altitude_col].iloc[0]

    transformer = Transformer.from_crs('epsg:4326', 'epsg:32756', always_xy=True)
    ref_x, ref_y = transformer.transform(reference_longitude, reference_latitude)
    flight_data['X_meters'], flight_data['Y_meters'] = transformer.transform(
        flight_data[longitude_col].values,
        flight_data[latitude_col].values
    )

    # Calcular offsets iniciales
    drone_initial_azimuth = calculate_azimuth_meters(ref_x, ref_y,
                                                     flight_data.iloc[0]['X_meters'],
                                                     flight_data.iloc[0]['Y_meters'])
    drone_initial_elevation = calculate_elevation_meters(flight_data.iloc[0][altitude_col],
                                                         ref_x, ref_y,
                                                         flight_data.iloc[0]['X_meters'],
                                                         flight_data.iloc[0]['Y_meters'],
                                                         initial_altitude)
    azimuth_offset = initial_azimuth - drone_initial_azimuth
    elevation_offset = initial_elevation - drone_initial_elevation

    return {
        'name': drone_config['name'],
        'flight_data': flight_data,
        'ref_x': ref_x,
        'ref_y': ref_y,
        'initial_altitude': initial_altitude,
        'azimuth_offset': azimuth_offset,
        'elevation_offset': elevation_offset,
        'time_col': time_col,
        'altitude_col': altitude_col,
        'latitude_col': latitude_col,
        'longitude_col': longitude_col
    }

if __name__ == "__main__":
    # Preparar datos de todos los drones antes del bucle principal
    drones_data = [prepare_drone_data(cfg) for cfg in drones_config]

    # Abrir archivos de audio
    wav_files = open_wav_files(wav_filenames)
    for wav_file in wav_files:
        skip_wav_seconds(wav_file, skip_seconds, RATE)

    buffers = [np.zeros((CHUNK, CHANNELS), dtype=np.int32) for _ in range(len(wav_files))]

    # Determinar el número máximo de iteraciones según el dron con la menor cantidad de datos
    min_length = min([len(d['flight_data']) for d in drones_data])
    max_iterations = min(int(RATE / CHUNK * RECORD_SECONDS), min_length)

    # Preparar figura y lines para cada dron
    plt.ion()
    fig, ax = plt.subplots(figsize=(15, 5))
    cax = ax.imshow(np.zeros((len(elevation_range), len(azimuth_range))),
                    extent=[azimuth_range[0], azimuth_range[-1], elevation_range[0], elevation_range[-1]],
                    origin='lower', aspect='auto', cmap='jet', interpolation='nearest')
    fig.colorbar(cax, ax=ax, label='Energy')
    ax.set_xlabel('Azimuth (deg)')
    ax.set_ylabel('Elevation (deg)')
    ax.set_title('Beamforming Energy + CSV Trajectories')

    # Crear lineas para cada dron (trayectorias)
    drone_lines = []
    drone_azimuths = []
    drone_elevations = []
    for d in drones_data:
        line_csv, = ax.plot([], [], '+', label=d['name'])  # marcador '+'
        drone_lines.append(line_csv)
        drone_azimuths.append([])
        drone_elevations.append([])

    # Marcador de energía máxima
    max_energy_marker, = ax.plot([], [], 'ro')
    plt.grid(True)
    plt.legend()
    fig.canvas.draw()
    fig.canvas.flush_events()

    for time_idx, i in zip(range(max_iterations), range(min_length)):
        # Leer bloque de audio
        for j, wav_file in enumerate(wav_files):
            block = read_wav_block(wav_file, CHUNK, CHANNELS)
            if block is None:
                break
            buffers[j] = block

        if block is None:
            break

        combined_signal = np.hstack(buffers)
        filtered_signal = apply_bandpass_filter(combined_signal, lowcut, highcut, RATE)

        # Beamforming
        energy = beamform_time(filtered_signal, delay_samples)
        max_energy_idx = np.unravel_index(np.argmax(energy), energy.shape)
        estimated_azimuth = azimuth_range[max_energy_idx[0]]
        estimated_elevation = elevation_range[max_energy_idx[1]]
        current_time_audio = calculate_time(time_idx, CHUNK, RATE)

        # Actualizar el mapa de energía
        cax.set_data(energy.T)
        cax.set_clim(vmin=np.min(energy), vmax=np.max(energy))

        # Actualizar el marcador de energía máxima
        max_energy_marker.set_data([estimated_azimuth], [estimated_elevation])

        # Para cada dron, calcular posición CSV en este índice
        for d_idx, d in enumerate(drones_data):
            flight_data = d['flight_data']
            altitude_col = d['altitude_col']
            time_col = d['time_col']

            x = flight_data.iloc[i]['X_meters']
            y = flight_data.iloc[i]['Y_meters']
            altitude = flight_data.iloc[i][altitude_col]

            csv_azimuth = calculate_azimuth_meters(d['ref_x'], d['ref_y'], x, y) + d['azimuth_offset']
            csv_azimuth = wrap_angle(csv_azimuth)
            csv_azimuth = -csv_azimuth
            csv_elevation = calculate_elevation_meters(altitude, d['ref_x'], d['ref_y'], x, y, d['initial_altitude']) + d['elevation_offset']
            total_distance = calculate_total_distance_meters(d['ref_x'], d['ref_y'], x, y, d['initial_altitude'], altitude)

            # Agregar el punto CSV actual a la trayectoria de este dron
            drone_azimuths[d_idx].append(csv_azimuth)
            drone_elevations[d_idx].append(csv_elevation)

            # Actualizar la línea del dron
            drone_lines[d_idx].set_data(drone_azimuths[d_idx], drone_elevations[d_idx])

            # Opcional: calcular diferencias, imprimir información
            azimuth_diff, elevation_diff = calculate_angle_difference(
                estimated_azimuth, csv_azimuth, estimated_elevation, csv_elevation
            )
            current_time_csv = flight_data.iloc[i][time_col]
            print(f"{d['name']} Dist: {total_distance:.2f} m "\
                  f"Audio time: {current_time_audio + skip_seconds:.2f} s - CSV time: {current_time_csv} s - "
                  f"SSL: Azim = {estimated_azimuth:.2f}°, Elev = {estimated_elevation:.2f}° "
                  f"CSV: Azim = {csv_azimuth:.2f}°, Elev = {csv_elevation:.2f}° "
                  f"Diff: Azim = {azimuth_diff:.2f}°, Elev = {elevation_diff:.2f}°")

        # Forzar actualización de la figura
        fig.canvas.draw()
        fig.canvas.flush_events()

    # Finalizar
    for wav_file in wav_files:
        wav_file.close()

    plt.ioff()
    plt.show()
