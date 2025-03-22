# geo_utils.py
import numpy as np
import pandas as pd
import numpy as np
from pyproj import Transformer

def wrap_angle(angle):
    # Wrap angle to [-180, 180)
    return ((angle + 180) % 360) - 180

def angular_difference(angle1, angle2):
    diff = (angle1 - angle2 + 180) % 360 - 180
    return abs(diff)

def calculate_horizontal_distance_meters(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def calculate_azimuth_meters(x1, y1, x2, y2):
    # Calculate azimuth in degrees
    azimuth = np.degrees(np.arctan2(y2 - y1, x2 - x1))
    return azimuth

def calculate_elevation_meters(altitude, x1, y1, x2, y2, reference_altitude):
    # Calculate elevation in degrees
    horizontal_distance = calculate_horizontal_distance_meters(x1, y1, x2, y2)
    relative_altitude = altitude - reference_altitude
    return np.degrees(np.arctan2(relative_altitude, horizontal_distance))

def calculate_total_distance_meters(x1, y1, x2, y2, alt1, alt2):
    horizontal_distance = calculate_horizontal_distance_meters(x1, y1, x2, y2)
    altitude_difference = alt2 - alt1
    total_distance = np.sqrt(horizontal_distance**2 + altitude_difference**2)
    return total_distance

def calculate_angle_difference(beamform_az, csv_az, beamform_el, csv_el):
    az_diff = angular_difference(beamform_az, csv_az)
    el_diff = abs(beamform_el - csv_el)
    return az_diff, el_diff


def compute_relative_flight_positions(flight_csv_path, reference_csv_path, utm_zone=18, save_csv=False,
                                      output_path=None):
    """
    Convert drone GPS flight data to relative ENU positions (meters) with respect to a reference GPS point.
    Returns positions as [X = East, Y = North, Z = Altitude].

    Parameters:
        flight_csv_path (str): Path to the drone's flight CSV log.
        reference_csv_path (str): Path to the reference CSV for position 0,0,0 (mic array origin).
        utm_zone (int): UTM zone for projection (default = 18).
        save_csv (bool): If True, saves the result to CSV.
        output_path (str): Path to save the output CSV if save_csv is True.

    Returns:
        np.ndarray: Array of shape [N x 3] with [X, Y, Z] positions in meters.
    """
    ref_df = pd.read_csv(reference_csv_path, low_memory=False)
    flight_df = pd.read_csv(flight_csv_path, low_memory=False)

    ref_df.columns = [col.strip().lower() for col in ref_df.columns]
    flight_df.columns = [col.strip().lower() for col in flight_df.columns]

    lat_col_ref = [col for col in ref_df.columns if 'lat' in col][0]
    lon_col_ref = [col for col in ref_df.columns if 'lon' in col][0]
    alt_col_ref = [col for col in ref_df.columns if 'alt' in col][0]

    lat_col_flight = [col for col in flight_df.columns if 'lat' in col][0]
    lon_col_flight = [col for col in flight_df.columns if 'lon' in col][0]
    alt_col_flight = [col for col in flight_df.columns if 'alt' in col][0]

    lat0 = ref_df[lat_col_ref].dropna().values[0]
    lon0 = ref_df[lon_col_ref].dropna().values[0]
    alt0 = ref_df[alt_col_ref].dropna().values[0]

    crs_utm = f"EPSG:326{utm_zone:02d}"
    transformer = Transformer.from_crs("EPSG:4326", crs_utm, always_xy=True)
    x0, y0 = transformer.transform(lon0, lat0)

    relative_positions = []
    for _, row in flight_df.iterrows():
        try:
            lat = float(row[lat_col_flight])
            lon = float(row[lon_col_flight])
            alt = float(row[alt_col_flight])
            x, y = transformer.transform(lon, lat)
            dx = x - x0  # East (X)
            dy = y - y0  # North (Y)
            dz = alt - alt0  # Altitude (Z)
            relative_positions.append([dx, dy, dz])
        except:
            continue

    relative_positions_np = np.array(relative_positions)

    #if save_csv and output_path:
    #    np.savetxt(output_path, relative_positions_np, delimiter=',', header='X,Y,Z', comments='')
    #    print(f"Saved relative positions to {output_path}")

    return relative_positions_np


def calculate_delays_for_direction(mic_positions, azimuth, elevation, sample_rate, speed_of_sound):
    """
    Computes delay samples for beamforming given a direction (azimuth and elevation).

    Parameters:
        mic_positions (np.ndarray): Microphone positions.
        azimuth (float): Target azimuth (degrees).
        elevation (float): Target elevation (degrees).
        sample_rate (int): Audio sample rate.
        speed_of_sound (float): Speed of sound in m/s.

    Returns:
        delay_samples (np.ndarray): Array of delay values (in samples) for each microphone.
    """
    # Convert angles from degrees to radians
    azimuth_rad = np.radians(azimuth)
    elevation_rad = np.radians(elevation)

    # Define the unit direction vector for the beamforming target.
    direction_vector = np.array([
        np.cos(elevation_rad) * np.cos(azimuth_rad),
        np.cos(elevation_rad) * np.sin(azimuth_rad),
        np.sin(elevation_rad)
    ])

    # Compute the delay (in seconds) for each microphone
    delays = np.dot(mic_positions, direction_vector) / speed_of_sound
    # Convert delays to integer sample delays
    delay_samples = np.round(delays * sample_rate).astype(np.int32)
    return delay_samples