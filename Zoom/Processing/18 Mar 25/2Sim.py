# main.py
# 22 nov 10:57 DJI Inspire 1. Lake. Vid 1
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pyproj import Transformer
import librosa

from audio_beamforming import beamform_time
from geo_utils import (wrap_angle, calculate_angle_difference,
                       calculate_azimuth_meters, calculate_elevation_meters,
                       calculate_total_distance_meters)
from io_utils import (read_wav_block, skip_wav_seconds, apply_bandpass_filter,
                      calculate_time, initialize_beamforming_params, open_wav_files)

# ----------------------- GENERAL CONFIGURATIONS -----------------------
CHANNELS = 6
RATE = 48000
BUFFER = 0.4
CHUNK = int(BUFFER * RATE)
c = 343
RECORD_SECONDS = 120000
lowcut = 200.0
highcut = 800.0
skip_seconds = 0

azimuth_range = np.arange(-180, 181, 4)
elevation_range = np.arange(0, 91, 4)

mic_positions, delay_samples, num_mics = initialize_beamforming_params(azimuth_range, elevation_range, c, RATE)

wav_filenames = [
    'C:/Users/30068385\OneDrive - Western Sydney University\ICNS\PhD\simulations\pyroom\sim_spherical\device_1SEu192.wav',
    'C:/Users/30068385\OneDrive - Western Sydney University\ICNS\PhD\simulations\pyroom\sim_spherical\device_2SEu192.wav',
    'C:/Users/30068385\OneDrive - Western Sydney University\ICNS\PhD\simulations\pyroom\sim_spherical\device_3SEu192.wav',
    'C:/Users/30068385\OneDrive - Western Sydney University\ICNS\PhD\simulations\pyroom\sim_spherical\device_4SEu192.wav'
]

drones_config = [
    {
        'name': 'DJI Air 3',
        'ref_csv': '/Users/30068385/OneDrive - Western Sydney University/FlightRecord/DJI Air 3/CSV/18 Mar 25/Ref/Mar-18th-2025-10-31AM-Flight-Airdata.csv',
        #'ref_csv': '/Users/30068385/OneDrive - Western Sydney University/FlightRecord/DJI Air 3/CSV/22 Nov/Ref/Nov-22nd-2024-11-48AM-Flight-Airdata.csv',
        'flight_csv': '/Users/30068385/OneDrive - Western Sydney University/FlightRecord/DJI Air 3/CSV/18 Mar 25/2/Mar-18th-2025-11-55AM-Flight-Airdata.csv',
        'latitude_col': 'latitude',
        'altitude_col': 'altitude_above_seaLevel(feet)',
        'longitude_col': 'longitude',
        'time_col': 'time(millisecond)',
        'initial_azimuth': 15.0,
        'initial_elevation': 0.0,
        'start_index': 0,
    }
]

def process_drone_data(drone_config):
    ref_csv = drone_config['ref_csv']
    flight_csv = drone_config['flight_csv']
    latitude_col = drone_config['latitude_col']
    longitude_col = drone_config['longitude_col']
    altitude_col = drone_config['altitude_col']
    time_col = drone_config['time_col']
    initial_azimuth = drone_config['initial_azimuth']
    initial_elevation = drone_config['initial_elevation']
    start_index = drone_config['start_index']
    drone_name = drone_config['name']

    ref_data = pd.read_csv(ref_csv, skiprows=0, delimiter=',', low_memory=False)
    flight_data = pd.read_csv(flight_csv, skiprows=0, delimiter=',', low_memory=False)

    reference_latitude = ref_data[latitude_col].dropna().astype(float).mean()
    reference_longitude = ref_data[longitude_col].dropna().astype(float).mean()

    flight_data = flight_data.iloc[start_index:].reset_index(drop=True)
    flight_data = flight_data[[latitude_col, longitude_col, altitude_col, time_col]].dropna()
    ref_data = ref_data[[latitude_col, longitude_col, altitude_col, time_col]].dropna()
    flight_data[altitude_col] = flight_data[altitude_col] * 0.3048
    ref_data[altitude_col] = ref_data[altitude_col] * 0.3048
    #initial_altitude = flight_data[altitude_col].iloc[0]
    initial_altitude = ref_data[altitude_col].iloc[0]

    transformer = Transformer.from_crs('epsg:4326', 'epsg:32756', always_xy=True)
    ref_x, ref_y = transformer.transform(reference_longitude, reference_latitude)
    flight_data['X_meters'], flight_data['Y_meters'] = transformer.transform(
        flight_data[longitude_col].values,
        flight_data[latitude_col].values
    )

    drone_initial_azimuth = calculate_azimuth_meters(ref_x, ref_y,
                                                     flight_data.iloc[0]['X_meters'],
                                                     flight_data.iloc[0]['Y_meters'])
    drone_initial_elevation = calculate_elevation_meters(flight_data.iloc[0][altitude_col],
                                                         ref_x, ref_y,
                                                         flight_data.iloc[0]['X_meters'],
                                                         flight_data.iloc[0]['Y_meters'],
                                                         initial_altitude)
    azimuth_offset = initial_azimuth - drone_initial_azimuth
    #elevation_offset = initial_elevation - drone_initial_elevation
    elevation_offset = initial_elevation

    wav_files = open_wav_files(wav_filenames)


    for wav_file in wav_files:
        skip_wav_seconds(wav_file, skip_seconds, RATE)

    buffers = [np.zeros((CHUNK, CHANNELS), dtype=np.int32) for _ in range(len(wav_files))]

    results_columns = ['Tiempo_Audio', 'Tiempo_CSV', 'Azimut_Estimado', 'Elevacion_Estimada',
                       'Azimut_CSV', 'Elevacion_CSV', 'Dif_Azimut', 'Dif_Elevacion', 'Distancia_Metros']
    results_df = pd.DataFrame(columns=results_columns)

    # Preparar el modo interactivo de matplotlib
    plt.ion()
    fig, ax = plt.subplots(figsize=(15, 5))
    # Crear el mapa inicial vacio
    cax = ax.imshow(np.zeros((len(elevation_range), len(azimuth_range))),
                    extent=[azimuth_range[0], azimuth_range[-1], elevation_range[0], elevation_range[-1]],
                    origin='lower', aspect='auto', cmap='jet', interpolation='nearest')

    line_csv, = ax.plot([], [], 'k+', markersize = 30, label='CSV Trajectory')
    fig.colorbar(cax, ax=ax, label='Energy')
    ax.set_xlabel('Azimuth (deg)')
    ax.set_ylabel('Elevation (deg)')
    ax.set_title(f'{drone_name} - Beamforming Energy + CSV Trajectory')

    # Marcador de energía máxima
    max_energy_marker, = ax.plot([], [], 'ro')
    # Trayectoria CSV del dron
    # Iremos agregando puntos a medida que pasa el tiempo
    csv_azimuths = []
    csv_elevations = []

    plt.legend()
    plt.grid(True)
    fig.canvas.draw()
    fig.canvas.flush_events()

    max_iterations = min(int(RATE / CHUNK * RECORD_SECONDS), len(flight_data))
    for time_idx, i in zip(range(0, max_iterations), range(len(flight_data))):
        for j, wav_file in enumerate(wav_files):
            block = read_wav_block(wav_file, CHUNK, CHANNELS)

            if block is None:
                break
            buffers[j] = block

        if block is None:
            break

        combined_signal = np.hstack(buffers)
        filtered_signal = apply_bandpass_filter(combined_signal, lowcut, highcut, RATE)

        # Beamform
        #energy = beamform_time(combined_signal, delay_samples)
        energy = beamform_time(filtered_signal, delay_samples)
        max_energy_idx = np.unravel_index(np.argmax(energy), energy.shape)
        estimated_azimuth = azimuth_range[max_energy_idx[0]]
        estimated_elevation = elevation_range[max_energy_idx[1]]
        current_time_audio = calculate_time(time_idx, CHUNK, RATE)

        # Calcular CSV azimuth/elevación/distancia
        x = flight_data.iloc[i]['X_meters']
        y = flight_data.iloc[i]['Y_meters']
        altitude = flight_data.iloc[i][altitude_col]

        csv_azimuth = calculate_azimuth_meters(ref_x, ref_y, x, y) + azimuth_offset
        csv_azimuth = wrap_angle(csv_azimuth)
        csv_azimuth = -csv_azimuth
        csv_elevation = calculate_elevation_meters(altitude, ref_x, ref_y, x, y, initial_altitude) + elevation_offset
        total_distance = calculate_total_distance_meters(ref_x, ref_y, x, y, initial_altitude, altitude)

        azimuth_diff, elevation_diff = calculate_angle_difference(
            estimated_azimuth, csv_azimuth, estimated_elevation, csv_elevation
        )

        current_time_csv = flight_data.iloc[i][time_col]

        new_data = pd.DataFrame([{
            'Tiempo_Audio': current_time_audio + skip_seconds,
            'Tiempo_CSV': current_time_csv,
            'Azimut_Estimado': estimated_azimuth,
            'Elevacion_Estimada': estimated_elevation,
            'Azimut_CSV': csv_azimuth,
            'Elevacion_CSV': csv_elevation,
            'Dif_Azimut': azimuth_diff,
            'Dif_Elevacion': elevation_diff,
            'Distancia_Metros': total_distance
        }])

        results_df = pd.concat([results_df, new_data], ignore_index=True)

        print(f"Dist: {total_distance:.2f} mts "\
              f"Audio time: {current_time_audio + skip_seconds:.2f} s - CSV time: {current_time_csv} s - " \
              f"SSL: Azim = {estimated_azimuth:.2f}°, Elev = {estimated_elevation:.2f}° " \
              f"CSV: Azim = {csv_azimuth:.2f}°, Elev = {csv_elevation:.2f}° " \
              f"Diff: Azim = {azimuth_diff:.2f}°, Elev = {elevation_diff:.2f}° - ")

        # Update energy map
        cax.set_data(energy.T)
        cax.set_clim(vmin=np.min(energy), vmax=np.max(energy))

        # update max energy marker
        max_energy_marker.set_data([estimated_azimuth], [estimated_elevation])

        # Agregar el punto CSV actual a la trayectoria
        #csv_azimuths.append(csv_azimuth)
        #csv_elevations.append(csv_elevation)
        #line_csv.set_data(csv_azimuths, csv_elevations)
        line_csv.set_data([csv_azimuth], [csv_elevation])

        fig.canvas.draw()
        fig.canvas.flush_events()

    print(f"{drone_name} processing completed.")
    plt.ioff()
    plt.show()

    for wav_file in wav_files:
        wav_file.close()

    return results_df

if __name__ == "__main__":
    all_results = {}

    for drone_cfg in drones_config:
        df_results = process_drone_data(drone_cfg)
        all_results[drone_cfg['name']] = df_results

        # Save results to CSV for each drone
        output_filename = f"{drone_cfg['name'].replace(' ', '_')}_results.csv"
        df_results.to_csv(output_filename, index=False)
        print(f"Results saved to {output_filename}")