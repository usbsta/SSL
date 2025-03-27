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


def compute_relative_flight_positions(flight_csv_path, reference_csv_path, save_csv=False, output_path=None):
    """
    Convert drone GPS flight data to relative ENU positions (meters) with respect to a reference point.
    Uses the EPSG:32756 projection (same as the beamforming code) and converts altitude from feet to meters.
    Returns an array of shape [N x 3] with [X (East), Y (North), Z (Altitude difference in meters)].
    """
    import pandas as pd
    # Load CSV files
    ref_df = pd.read_csv(reference_csv_path, low_memory=False)
    flight_df = pd.read_csv(flight_csv_path, low_memory=False)

    # Standardize column names
    ref_df.columns = [col.strip().lower() for col in ref_df.columns]
    flight_df.columns = [col.strip().lower() for col in flight_df.columns]

    # Identify latitude, longitude, and altitude columns (assuming they contain 'lat', 'lon' and 'alt')
    lat_col = [col for col in ref_df.columns if 'lat' in col][0]
    lon_col = [col for col in ref_df.columns if 'lon' in col][0]
    alt_col = [col for col in ref_df.columns if 'alt' in col][0]

    # Get reference coordinates from the reference CSV
    lat0 = ref_df[lat_col].dropna().values[0]
    lon0 = ref_df[lon_col].dropna().values[0]
    # Convert altitude from feet to meters (assuming input altitude is in feet)
    alt0 = ref_df[alt_col].dropna().values[0] * 0.3048

    # Create a transformer using the same projection as used in beamforming
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32756", always_xy=True)
    ref_x, ref_y = transformer.transform(lon0, lat0)

    relative_positions = []
    for _, row in flight_df.iterrows():
        try:
            lat = float(row[lat_col])
            lon = float(row[lon_col])
            alt = float(row[alt_col]) * 0.3048  # Convert altitude to meters
            x, y = transformer.transform(lon, lat)
            dx = x - ref_x
            dy = y - ref_y
            dz = alt - alt0
            relative_positions.append([dx, dy, dz])
        except Exception as e:
            continue

    return np.array(relative_positions)



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


def compute_drone_xyz(ref_csv, flight_csv, lat_col="latitude", lon_col="longitude", alt_col="altitude",
                      alt_factor=0.3048):
    """
    Compute the drone's x, y, z coordinates using a unified method.
    This function reads the reference and flight CSV files, computes a reference point (mean of latitudes and longitudes),
    and uses pyproj to convert geographic coordinates to a Cartesian coordinate system.

    Parameters:
        ref_csv (str): Path to the reference CSV file.
        flight_csv (str): Path to the flight CSV file.
        lat_col (str): Column name for latitude.
        lon_col (str): Column name for longitude.
        alt_col (str): Column name for altitude (in feet).
        alt_factor (float): Factor to convert altitude to meters (default 0.3048 for feet-to-meters).

    Returns:
        drone_xyz (np.ndarray): Array with shape (n_positions, 3) containing x, y, z coordinates.
        ref_point (tuple): (ref_x, ref_y) in the target projection.
    """
    import numpy as np
    import pandas as pd
    from pyproj import Transformer

    # Load CSV data
    ref_data = pd.read_csv(ref_csv)
    flight_data = pd.read_csv(flight_csv)

    # Compute reference latitude and longitude (mean)
    ref_lat = ref_data[lat_col].dropna().astype(float).mean()
    ref_lon = ref_data[lon_col].dropna().astype(float).mean()

    # Initialize transformer (e.g., from WGS84 to UTM zone; adjust EPSG as needed)
    transformer = Transformer.from_crs('epsg:4326', 'epsg:32756', always_xy=True)

    # Convert reference coordinates
    ref_x, ref_y = transformer.transform(ref_lon, ref_lat)

    # Convert flight data coordinates
    flight_x, flight_y = transformer.transform(
        flight_data[lon_col].values.astype(float),
        flight_data[lat_col].values.astype(float)
    )

    # Convert altitude from feet to meters
    flight_z = flight_data[alt_col].values.astype(float) * alt_factor

    # Stack coordinates: x, y, z
    drone_xyz = np.column_stack((flight_x, flight_y, flight_z))

    return drone_xyz, (ref_x, ref_y)
