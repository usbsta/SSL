import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import wave
import time
from pyproj import Transformer

from Utilities.functions import (
    initialize_microphone_positions,
    calculate_delays_for_direction,
    apply_beamforming,
    apply_bandpass_filter
)

from Utilities import pantilt

from geo_utils import (
    wrap_angle, calculate_angle_difference, calculate_azimuth_meters,
    calculate_elevation_meters, calculate_total_distance_meters
)

pantilt = pantilt.Pantilt("COM4", window_size=30, slow_factor=0.2, threshold=1.0, initial_pan=10.0, initial_tilt=5.0)
# ----------------------- GENERAL CONFIGURATIONS -----------------------
RECORD_SECONDS = 600
RATE = 48000
CHUNK = int(0.1 * RATE)  # Chunk duration (100 ms)
LOWCUT = 400.0
HIGHCUT = 3000.0
FILTER_ORDER = 5
c = 343
skip_seconds = 5.4  # Seconds to skip at the beginning of the audio

azimuth_range = np.arange(-180, 181, 4)
elevation_range = np.arange(0, 91, 4)

mic_positions = initialize_microphone_positions()
CHANNELS = mic_positions.shape[0]

precomputed_delays = np.empty((len(azimuth_range), len(elevation_range), CHANNELS), dtype=np.int32)
for i, az in enumerate(azimuth_range):
    for j, el in enumerate(elevation_range):
        precomputed_delays[i, j, :] = calculate_delays_for_direction(mic_positions, az, el, RATE, c)

# WAV filename (single file with eight channels)
wav_filename = '/Users/30068385/OneDrive - Western Sydney University/FlightRecord/DJI Inspire 1/CSV/03 Mar 25/2/20250303_140608_File0_Master_device.wav'

# Drone configuration and CSV file paths
drones_config = [
    {
        'name': 'DJI Inspire 1_1',
        'ref_csv': '/Users/30068385/OneDrive - Western Sydney University/FlightRecord/DJI Inspire 1/CSV/03 Mar 25/Ref/Mar-3rd-2025-01-35PM-Flight-Airdata.csv',
        'flight_csv': '/Users/30068385/OneDrive - Western Sydney University/FlightRecord/DJI Inspire 1/CSV/03 Mar 25/2/Mar-3rd-2025-02-06PM-Flight-Airdata2.csv',
        'latitude_col': 'latitude',
        'longitude_col': 'longitude',
        'altitude_col': 'altitude_above_seaLevel(feet)',
        'time_col': 'time(millisecond)',
        'initial_azimuth': 6.0,
        'initial_elevation': 2.0,
        'start_index': 30,
    }
]

def process_drone_data(drone_config):
    """
    Process drone flight data and audio beamforming using new beamforming functions.
    This function reads CSV flight data and processes a single WAV file (8 channels)
    to display a real-time beamforming energy map along with the CSV trajectory.
    """
    # Load CSV data
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

    # Read reference and flight CSV files
    ref_data = pd.read_csv(ref_csv, delimiter=',', low_memory=False)
    flight_data = pd.read_csv(flight_csv, delimiter=',', low_memory=False)

    # Compute reference coordinates (mean value)
    reference_latitude = ref_data[latitude_col].dropna().astype(float).mean()
    reference_longitude = ref_data[longitude_col].dropna().astype(float).mean()

    # Process flight data starting from a specified index
    flight_data = flight_data.iloc[start_index:].reset_index(drop=True)
    flight_data = flight_data[[latitude_col, longitude_col, altitude_col, time_col]].dropna()
    ref_data = ref_data[[latitude_col, longitude_col, altitude_col, time_col]].dropna()
    # Convert altitude from feet to meters
    flight_data[altitude_col] = flight_data[altitude_col].astype(float) * 0.3048
    ref_data[altitude_col] = ref_data[altitude_col].astype(float) * 0.3048
    initial_altitude = ref_data[altitude_col].iloc[0]

    # Coordinate transformation from geographic (epsg:4326) to metric (epsg:32756)
    transformer = Transformer.from_crs('epsg:4326', 'epsg:32756', always_xy=True)
    ref_x, ref_y = transformer.transform(reference_longitude, reference_latitude)
    flight_data['X_meters'], flight_data['Y_meters'] = transformer.transform(
        flight_data[longitude_col].values,
        flight_data[latitude_col].values
    )

    # Calculate initial drone azimuth and elevation from the first flight data point
    drone_initial_azimuth = calculate_azimuth_meters(ref_x, ref_y,
                                                     flight_data.iloc[0]['X_meters'],
                                                     flight_data.iloc[0]['Y_meters'])
    drone_initial_elevation = calculate_elevation_meters(flight_data.iloc[0][altitude_col],
                                                         ref_x, ref_y,
                                                         flight_data.iloc[0]['X_meters'],
                                                         flight_data.iloc[0]['Y_meters'],
                                                         initial_altitude)
    azimuth_offset = initial_azimuth - drone_initial_azimuth
    elevation_offset = initial_elevation  # No correction applied to elevation

    # Open WAV file using the wave module
    wf = wave.open(wav_filename, 'rb')
    # Verify WAV file parameters
    if wf.getnchannels() != CHANNELS:
        print(f"Error: Expected {CHANNELS} channels, but found {wf.getnchannels()} in {wav_filename}")
        wf.close()
        return
    if wf.getsampwidth() != 2:
        print("Error: WAV file sample width is not 16-bit.")
        wf.close()
        return
    if wf.getframerate() != RATE:
        print("Error: WAV file sampling rate does not match the expected RATE.")
        wf.close()
        return

    # Skip initial seconds if needed
    wf.setpos(int(skip_seconds * RATE))

    # Set up an interactive plot for real-time visualization
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 3))
    heatmap = ax.imshow(np.zeros((len(azimuth_range), len(elevation_range))).T,
                        extent=[azimuth_range[0], azimuth_range[-1], elevation_range[0], elevation_range[-1]],
                        origin='lower', aspect='auto', cmap='jet')
    fig.colorbar(heatmap, ax=ax, label='Energy')
    ax.set_xlabel('Azimuth (degrees)')
    ax.set_ylabel('Elevation (degrees)')
    ax.set_title(f'{drone_name} - Beamforming Energy + CSV Trajectory')

    # Marker for maximum energy and CSV trajectory line
    max_energy_marker, = ax.plot([], [], 'ro', label='Max Energy')
    line_csv, = ax.plot([], [], 'k+', label='CSV Trajectory')
    plt.legend()
    plt.grid(True)
    fig.canvas.draw()
    fig.canvas.flush_events()

    # Initialize CSV trajectory lists and results DataFrame
    csv_azimuths = []
    csv_elevations = []
    results_columns = ['Audio_Time', 'CSV_Time', 'Estimated_Azimuth', 'Estimated_Elevation',
                       'CSV_Azimuth', 'CSV_Elevation', 'Azimuth_Diff', 'Elevation_Diff', 'Distance_Meters']
    results_df = pd.DataFrame(columns=results_columns)

    # Determine maximum iterations based on the available frames and CSV data length
    #max_iterations = min(int((wf.getnframes() / CHANNELS) / CHUNK), len(flight_data))
    max_iterations = min(int(RATE / CHUNK * RECORD_SECONDS), len(flight_data))

    # Process audio in chunks and synchronize with CSV flight data
    for time_idx in range(max_iterations):
        # Read a chunk of audio data
        data = wf.readframes(CHUNK)
        if len(data) < CHUNK * CHANNELS * 2:
            break
        audio_chunk = np.frombuffer(data, dtype=np.int16).reshape((-1, CHANNELS))

        # Apply bandpass filter to the audio chunk
        filtered_chunk = apply_bandpass_filter(audio_chunk, LOWCUT, HIGHCUT, RATE, order=FILTER_ORDER)

        # Compute beamforming energy map using new beamforming functions
        energy_map = np.zeros((len(azimuth_range), len(elevation_range)))
        for i_az, az in enumerate(azimuth_range):
            for j_el, el in enumerate(elevation_range):
                beamformed_signal = apply_beamforming(filtered_chunk, precomputed_delays[i_az, j_el, :])
                energy_map[i_az, j_el] = np.sum(beamformed_signal ** 2)

        # Find the direction with maximum energy
        max_idx = np.unravel_index(np.argmax(energy_map), energy_map.shape)
        estimated_azimuth = azimuth_range[max_idx[0]]
        estimated_elevation = elevation_range[max_idx[1]]

        #pantilt.set(pan_degrees=estimated_azimuth, tilt_degrees=estimated_elevation)
        pantilt.set_smoothed(estimated_azimuth, estimated_elevation)

        #time.sleep(1)

        current_time_audio = time_idx * (CHUNK / RATE) + skip_seconds

        # Retrieve corresponding CSV flight data
        csv_row = flight_data.iloc[time_idx]
        x = csv_row['X_meters']
        y = csv_row['Y_meters']
        altitude = csv_row[altitude_col]
        csv_azimuth = calculate_azimuth_meters(ref_x, ref_y, x, y) + azimuth_offset
        csv_azimuth = wrap_angle(csv_azimuth)
        csv_azimuth = -csv_azimuth  # Invert azimuth if necessary
        csv_elevation = calculate_elevation_meters(altitude, ref_x, ref_y, x, y, initial_altitude) + elevation_offset
        total_distance = calculate_total_distance_meters(ref_x, ref_y, x, y, initial_altitude, altitude)

        azimuth_diff, elevation_diff = calculate_angle_difference(
            estimated_azimuth, csv_azimuth, estimated_elevation, csv_elevation
        )
        current_time_csv = csv_row[time_col]

        # Append new data to the results DataFrame
        new_data = pd.DataFrame([{
            'Audio_Time': current_time_audio,
            'CSV_Time': current_time_csv,
            'Estimated_Azimuth': estimated_azimuth,
            'Estimated_Elevation': estimated_elevation,
            'CSV_Azimuth': csv_azimuth,
            'CSV_Elevation': csv_elevation,
            'Azimuth_Diff': azimuth_diff,
            'Elevation_Diff': elevation_diff,
            'Distance_Meters': total_distance
        }])
        results_df = pd.concat([results_df, new_data], ignore_index=True)

        print(f"Dist: {total_distance:.2f} mts "\
              f"Audio time: {current_time_audio + skip_seconds:.2f} s - CSV time: {current_time_csv} s - " \
              f"SSL: Azim = {estimated_azimuth:.2f}°, Elev = {estimated_elevation:.2f}° " \
              f"CSV: Azim = {csv_azimuth:.2f}°, Elev = {csv_elevation:.2f}° " \
              f"Diff: Azim = {azimuth_diff:.2f}°, Elev = {elevation_diff:.2f}° - ")

        # Update the heatmap display and markers
        heatmap.set_data(energy_map.T)
        heatmap.set_clim(vmin=np.min(energy_map), vmax=np.max(energy_map))
        max_energy_marker.set_data([estimated_azimuth], [estimated_elevation])

        # Append CSV trajectory point and update the trajectory line
        csv_azimuths.append(csv_azimuth)
        csv_elevations.append(csv_elevation)
        line_csv.set_data(csv_azimuths, csv_elevations)





        fig.canvas.draw()
        fig.canvas.flush_events()

    print(f"{drone_name} processing completed.")
    plt.ioff()
    plt.show()
    wf.close()

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
