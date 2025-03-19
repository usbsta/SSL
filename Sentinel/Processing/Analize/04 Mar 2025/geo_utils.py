# geo_utils.py
import numpy as np

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