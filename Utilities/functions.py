# functions.py
import wave
import numpy as np
import pandas as pd
from pyproj import Transformer
from numba import njit
from scipy.signal import butter, filtfilt
#from experiments_config import get_experiment_config

def microphone_positions_8_helicop():
    """
    Initialize microphone positions in 3D space.
    Returns:
        mic_positions (np.ndarray): Array of microphone positions.
    """
    # Angles for microphone placement
    a = [0, -90, -180, -270]

    # progressive distance configuration
    h = [0.29, 0.06]
    r = [0.16, 0.35]

    # Define microphone positions
    mic_positions = np.array([
        [r[0] * np.cos(np.radians(a[0])), r[0] * np.sin(np.radians(a[0])), h[0]], # mic 1
        [r[1] * np.cos(np.radians(a[1])), r[1] * np.sin(np.radians(a[1])), h[1]], # mic 2
        [r[0] * np.cos(np.radians(a[2])), r[0] * np.sin(np.radians(a[2])), h[0]], # mic 3
        [r[1] * np.cos(np.radians(a[3])), r[1] * np.sin(np.radians(a[3])), h[1]], # mic 4
        [r[0] * np.cos(np.radians(a[0])), r[0] * np.sin(np.radians(a[0])), h[0]], # mic 5
        [r[1] * np.cos(np.radians(a[1])), r[1] * np.sin(np.radians(a[1])), h[1]], # mic 6
        [r[0] * np.cos(np.radians(a[2])), r[0] * np.sin(np.radians(a[2])), h[0]], # mic 7
        [r[1] * np.cos(np.radians(a[3])), r[1] * np.sin(np.radians(a[3])), h[1]]  # mic 8
    ])
    return mic_positions

def microphone_positions_8_medium():
    """
    Initialize microphone positions in 3D space.
    Returns:
        mic_positions (np.ndarray): Array of microphone positions.
    """
    # Angles for microphone placement
    a = [0, -60, -180, -300]

    # progressive distance configuration
    h = [0.45, 0.25, 0.11]
    r = [0.07, 0.02, 0.18]

    # Define microphone positions
    mic_positions = np.array([
        [r[0] * np.cos(np.radians(a[0])), r[0] * np.sin(np.radians(a[0])), h[0]], # mic 1
        [r[1] * np.cos(np.radians(a[2])), r[1] * np.sin(np.radians(a[2])), h[1]], # mic 2
        [r[0] * np.cos(np.radians(a[1])), r[0] * np.sin(np.radians(a[1])), h[0]], # mic 3
        [r[2] * np.cos(np.radians(a[1])), r[2] * np.sin(np.radians(a[1])), h[2]], # mic 4
        [r[0] * np.cos(np.radians(a[2])), r[0] * np.sin(np.radians(a[2])), h[0]], # mic 5
        [r[2] * np.cos(np.radians(a[2])), r[2] * np.sin(np.radians(a[2])), h[2]], # mic 6
        [r[0] * np.cos(np.radians(a[3])), r[0] * np.sin(np.radians(a[3])), h[0]], # mic 7
        [r[2] * np.cos(np.radians(a[3])), r[2] * np.sin(np.radians(a[3])), h[2]]  # mic 8
    ])
    return mic_positions

def microphone_positions_8_medium2():
    """
    Initialize microphone positions in 3D space.
    Returns:
        mic_positions (np.ndarray): Array of microphone positions.
    """
    # Angles for microphone placement
    a = [0, -60, -180, -300]

    # progressive distance configuration
    #h = [0.45, 0.25, 0.11]
    h = [-0.36, -0.56, -0.7]
    r = [0.07, 0.02, 0.18]

    # Define microphone positions
    mic_positions = np.array([
        [r[0] * np.cos(np.radians(a[0])), r[0] * np.sin(np.radians(a[0])), h[0]], # mic 1
        [r[1] * np.cos(np.radians(a[2])), r[1] * np.sin(np.radians(a[2])), h[1]], # mic 2
        [r[0] * np.cos(np.radians(a[1])), r[0] * np.sin(np.radians(a[1])), h[0]], # mic 3
        [r[2] * np.cos(np.radians(a[1])), r[2] * np.sin(np.radians(a[1])), h[2]], # mic 4
        [r[0] * np.cos(np.radians(a[2])), r[0] * np.sin(np.radians(a[2])), h[0]], # mic 5
        [r[2] * np.cos(np.radians(a[2])), r[2] * np.sin(np.radians(a[2])), h[2]], # mic 6
        [r[0] * np.cos(np.radians(a[3])), r[0] * np.sin(np.radians(a[3])), h[0]], # mic 7
        [r[2] * np.cos(np.radians(a[3])), r[2] * np.sin(np.radians(a[3])), h[2]]  # mic 8
    ])
    return mic_positions



def microphone_positions_8_small():
    """
    Initialize microphone positions in 3D space.
    Returns:
        mic_positions (np.ndarray): Array of microphone positions.
    """
    # Angles for microphone placement
    a = [0, -60, -180, -300]

    # progressive distance configuration
    h = [0.35, 0.21, 0.08]
    r = [0.05, 0.015, 0.15]

    # Define microphone positions
    mic_positions = np.array([
        [r[1] * np.cos(np.radians(a[0])), r[1] * np.sin(np.radians(a[0])), h[0]], # mic 1
        [r[1] * np.cos(np.radians(a[2])), r[1] * np.sin(np.radians(a[2])), h[1]], # mic 2
        [r[0] * np.cos(np.radians(a[1])), r[0] * np.sin(np.radians(a[1])), h[0]], # mic 3
        [r[2] * np.cos(np.radians(a[1])), r[2] * np.sin(np.radians(a[1])), h[2]], # mic 4
        [r[0] * np.cos(np.radians(a[2])), r[0] * np.sin(np.radians(a[2])), h[0]], # mic 5
        [r[2] * np.cos(np.radians(a[2])), r[2] * np.sin(np.radians(a[2])), h[2]], # mic 6
        [r[0] * np.cos(np.radians(a[3])), r[0] * np.sin(np.radians(a[3])), h[0]], # mic 7
        [r[2] * np.cos(np.radians(a[3])), r[2] * np.sin(np.radians(a[3])), h[2]]  # mic 8
    ])
    return mic_positions

def initialize_microphone_positions_24():
    """
    Initialize microphone positions in 3D space.
    Returns:
        mic_positions (np.ndarray): Array of microphone positions.
    """
    # Angles for microphone placement
    a = [0, -120, -240]
    a2 = [-40, -80, -160, -200, -280, -320]

    # progressive distance configuration
    h = [1.12, 1.02, 0.87, 0.68, 0.47, 0.02]
    r = [0.1, 0.16, 0.23, 0.29, 0.43, 0.63]

    # Define microphone positions
    mic_positions = np.array([
        [r[0] * np.cos(np.radians(a[0])), r[0] * np.sin(np.radians(a[0])), h[0]], #mic 1
        [r[0] * np.cos(np.radians(a[1])), r[0] * np.sin(np.radians(a[1])), h[0]], #mic 2
        [r[0] * np.cos(np.radians(a[2])), r[0] * np.sin(np.radians(a[2])), h[0]], #mic 3
        [r[1] * np.cos(np.radians(a[0])), r[1] * np.sin(np.radians(a[0])), h[1]], #mic 4
        [r[1] * np.cos(np.radians(a[1])), r[1] * np.sin(np.radians(a[1])), h[1]], #mic 5
        [r[1] * np.cos(np.radians(a[2])), r[1] * np.sin(np.radians(a[2])), h[1]], #mic 6
        [r[2] * np.cos(np.radians(a[0])), r[2] * np.sin(np.radians(a[0])), h[2]], #mic 7
        [r[2] * np.cos(np.radians(a[1])), r[2] * np.sin(np.radians(a[1])), h[2]], #mic 8
        [r[2] * np.cos(np.radians(a[2])), r[2] * np.sin(np.radians(a[2])), h[2]], #mic 9
        [r[3] * np.cos(np.radians(a[0])), r[3] * np.sin(np.radians(a[0])), h[3]], #mic 10
        [r[3] * np.cos(np.radians(a[1])), r[3] * np.sin(np.radians(a[1])), h[3]], #mic 11
        [r[3] * np.cos(np.radians(a[2])), r[3] * np.sin(np.radians(a[2])), h[3]], #mic 12
        [r[4] * np.cos(np.radians(a[0])), r[4] * np.sin(np.radians(a[0])), h[4]], #mic 13
        [r[4] * np.cos(np.radians(a[1])), r[4] * np.sin(np.radians(a[1])), h[4]], #mic 14
        [r[4] * np.cos(np.radians(a[2])), r[4] * np.sin(np.radians(a[2])), h[4]], #mic 15
        [r[5] * np.cos(np.radians(a[0])), r[5] * np.sin(np.radians(a[0])), h[5]], #mic 16
        [r[5] * np.cos(np.radians(a[1])), r[5] * np.sin(np.radians(a[1])), h[5]], #mic 17
        [r[5] * np.cos(np.radians(a[2])), r[5] * np.sin(np.radians(a[2])), h[5]], #mic 18
        [r[2] * np.cos(np.radians(a2[0])), r[2] * np.sin(np.radians(a2[0])), h[2]], #mic 19
        [r[2] * np.cos(np.radians(a2[1])), r[2] * np.sin(np.radians(a2[1])), h[2]], #mic 20
        [r[2] * np.cos(np.radians(a2[2])), r[2] * np.sin(np.radians(a2[2])), h[2]], #mic 21
        [r[2] * np.cos(np.radians(a2[3])), r[2] * np.sin(np.radians(a2[3])), h[2]], #mic 22
        [r[2] * np.cos(np.radians(a2[4])), r[2] * np.sin(np.radians(a2[4])), h[2]], #mic 23
        [r[2] * np.cos(np.radians(a2[5])), r[2] * np.sin(np.radians(a2[5])), h[2]], #mic 24
    ])
    return mic_positions

def microphone_positions_4_ultra():
    """Return microphone positions for a compact 4-channel array."""

    # Angles for microphone placement
    a = [0, -60, -180, -300]

    # Distance configuration (radius and height)
    h = [0.35, 0.21, 0.08]
    r = [0.05, 0.015, 0.15]

    # Only four microphones are required.  The original implementation
    # mistakenly returned eight positions which caused shape mismatches in
    # consumers expecting a 4-element array.
    mic_positions = np.array([
        [r[1] * np.cos(np.radians(a[0])), r[1] * np.sin(np.radians(a[0])), h[0]],  # mic 1
        [r[1] * np.cos(np.radians(a[2])), r[1] * np.sin(np.radians(a[2])), h[1]],  # mic 2
        [r[0] * np.cos(np.radians(a[1])), r[0] * np.sin(np.radians(a[1])), h[0]],  # mic 3
        [r[2] * np.cos(np.radians(a[1])), r[2] * np.sin(np.radians(a[1])), h[2]],  # mic 4
    ])

    return mic_positions

# ----------------------------
# Filter Design Functions
# ----------------------------
def butter_bandpass(lowcut, highcut, fs, order=5):
    """Design a Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band')

def apply_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Apply a zero-phase Butterworth bandpass filter along the time axis."""
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return filtfilt(b, a, data, axis=0)

##### Audio Functions

def skip_wav_seconds(wav_file, seconds, rate):
    """
    Skip initial seconds in WAV file.
    """
    frames_to_skip = int(seconds * rate)
    wav_file.setpos(frames_to_skip)

def read_and_synchronize_audio(wav_filenames, channels, corrections, skip_seconds, sample_rate):
    """
    Reads unsynchronized WAV files, applies synchronization shifts based on provided corrections,
    and returns a combined signal (horizontal concatenation).

    Parameters:
        wav_filenames (list of str): List of paths to unsynchronized WAV files.
        channels (int): Number of channels per WAV file.
        corrections (list of int): Synchronization corrections (in samples) for each device.
        skip_seconds (float): Duration (in seconds) to skip at the beginning of each file of audio.
        sample_rate (int): Audio sample rate in Hz.

    Returns:
        combined_signal (np.ndarray): Combined audio signal from all devices.
    """
    signals = []
    min_length = None

    for idx, wav_file in enumerate(wav_filenames):
        with wave.open(wav_file, 'rb') as wf:
            # Skip the initial seconds (if needed)
            skip_wav_seconds(wf, skip_seconds, sample_rate)
            n_frames = wf.getnframes()
            frames = wf.readframes(n_frames)
            signal_data = np.frombuffer(frames, dtype=np.int32)
            signal_array = signal_data.reshape(-1, channels)
            # Apply synchronization shift
            correction = corrections[idx]
            shifted_signal = shift_signal(signal_array, correction)
            signals.append(shifted_signal)

            # Keep track of the shortest signal length to enable truncation later.
            if min_length is None or shifted_signal.shape[0] < min_length:
                min_length = shifted_signal.shape[0]

    # Truncate all signals to the length of the shortest one
    signals = [signal[:min_length] for signal in signals]
    # Combine signals horizontally (i.e., concatenate the channels from all devices)
    combined_signal = np.hstack(signals)
    return combined_signal

def shift_signal(signal, shift_amount):
    """
    Shifts a multi-channel signal by a specified number of samples.
    Positive shift_amount delays the signal (pads at the beginning),
    while a negative shift_amount advances it (pads at the end).

    Parameters:
        signal (np.ndarray): Input signal array (num_samples x num_channels).
        shift_amount (int): Number of samples to shift.

    Returns:
        shifted_signal (np.ndarray): Time-shifted signal.
    """
    if shift_amount > 0:
        shifted_signal = np.zeros_like(signal)
        shifted_signal[shift_amount:] = signal[:-shift_amount]
    elif shift_amount < 0:
        shift_amount = abs(shift_amount)
        shifted_signal = np.zeros_like(signal)
        shifted_signal[:-shift_amount] = signal[shift_amount:]
    else:
        shifted_signal = signal.copy()
    return shifted_signal

def beamform_in_direction(signal_data, mic_positions, azimuth, elevation, sample_rate, speed_of_sound):
    """
    Performs delay-and-sum beamforming in a specified direction.

    Parameters:
        signal_data (np.ndarray): Audio data (num_samples x num_mics).
        mic_positions (np.ndarray): Array with microphone positions.
        azimuth (float): Target azimuth angle in degrees.
        elevation (float): Target elevation angle in degrees.
        sample_rate (int): Audio sample rate in Hz.
        speed_of_sound (float): Speed of sound in m/s.

    Returns:
        beamformed_signal (np.ndarray): Beamformed (summed and normalized) audio signal.
    """
    # Compute the delay (in samples) for each microphone for the desired beamforming direction.
    delay_samples = calculate_delays_for_direction(mic_positions, azimuth, elevation, sample_rate, speed_of_sound)
    # Apply delays and sum the signals.
    beamformed_signal = apply_beamforming(signal_data, delay_samples)
    return beamformed_signal

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


def apply_beamforming(signal_data, delay_samples):
    """
    Applies the delay-and-sum beamforming algorithm to the input signal.

    Parameters:
        signal_data (np.ndarray): Multichannel audio data (num_samples x num_mics).
        delay_samples (np.ndarray): Delay (in samples) for each microphone.

    Returns:
        output_signal (np.ndarray): Beamformed (summed and normalized) output signal.
    """
    num_samples, num_mics = signal_data.shape
    output_signal = np.zeros(num_samples, dtype=np.float64)

    for mic_idx in range(num_mics):
        delay = delay_samples[mic_idx]
        shifted_signal = shift_signal_beamforming(signal_data[:, mic_idx], delay)
        output_signal += shifted_signal

    # Normalize by the number of microphones
    output_signal /= num_mics
    return output_signal


@njit
def shift_signal_beamforming(signal, delay_samples):
    """
    JIT-compiled function to shift a 1D signal by a given number of samples.
    Pads with zeros to maintain the signal length.

    Parameters:
        signal (np.ndarray): 1D input signal.
        delay_samples (int): Number of samples to shift.

    Returns:
        shifted_signal (np.ndarray): Shifted signal.
    """
    num_samples = signal.shape[0]
    shifted_signal = np.zeros_like(signal)
    if delay_samples > 0:
        if delay_samples < num_samples:
            shifted_signal[delay_samples:] = signal[:-delay_samples]
    elif delay_samples < 0:
        ds = -delay_samples
        if ds < num_samples:
            shifted_signal[:-ds] = signal[ds:]
    else:
        for i in range(num_samples):
            shifted_signal[i] = signal[i]
    return shifted_signal



##### CSV Functions

def load_flight_data(start_index, ref_file_path, file_path_flight):
    """
    Load flight data from CSV files and process coordinates.
    """
    # File paths
    #ref_file_path = '/Users/30068385/OneDrive - Western Sydney University/FlightRecord/DJI Air 3/CSV/25 Nov/Ref2/Air_Nov-25th-2024-03-19PM-Flight-Airdata.csv'
    #file_path_flight = '/Users/30068385/OneDrive - Western Sydney University/FlightRecord/DJI Air 3/CSV/25 Nov/11/Air_Nov-25th-2024-04-32PM-Flight-Airdata.csv'

    # Load CSV data
    ref_data = pd.read_csv(ref_file_path, skiprows=0, delimiter=',', low_memory=False)
    flight_data = pd.read_csv(file_path_flight, skiprows=0, delimiter=',', low_memory=False)

    # Calculate reference position
    reference_latitude = ref_data['latitude'].dropna().astype(float).mean()
    reference_longitude = ref_data['longitude'].dropna().astype(float).mean()

    # Column names
    latitude_col = 'latitude'
    longitude_col = 'longitude'
    altitude_col = 'altitude_above_seaLevel(feet)'
    time_col = 'time(millisecond)'

    # Process flight data
    flight_data = flight_data.iloc[start_index:].reset_index(drop=True)
    flight_data = flight_data[[latitude_col, longitude_col, altitude_col, time_col]].dropna()
    ref_data = ref_data[[latitude_col, longitude_col, altitude_col, time_col]].dropna()
    flight_data[altitude_col] = flight_data[altitude_col] * 0.3048  # Convert feet to meters
    ref_data[altitude_col] = ref_data[altitude_col] * 0.3048
    #initial_altitude = flight_data[altitude_col].iloc[0]
    initial_altitude = ref_data[altitude_col].iloc[0]


    # Coordinate transformation
    transformer = Transformer.from_crs('epsg:4326', 'epsg:32756', always_xy=True)  # Adjust UTM zone 56 south

    ref_x, ref_y = transformer.transform(reference_longitude, reference_latitude)

    flight_data['X_meters'], flight_data['Y_meters'] = transformer.transform(
        flight_data[longitude_col].values,
        flight_data[latitude_col].values
    )

    return flight_data, ref_x, ref_y, initial_altitude

def calculate_initial_offsets(flight_data, ref_x, ref_y, initial_altitude, initial_azimuth, initial_elevation):
    """
    Calculate initial azimuth and elevation offsets.
    """
    # Calculate initial drone azimuth and elevation
    drone_initial_azimuth = calculate_azimuth_meters(ref_x, ref_y,
                                                     flight_data.iloc[0]['X_meters'],
                                                     flight_data.iloc[0]['Y_meters'])


    azimuth_offset = initial_azimuth - drone_initial_azimuth
    elevation_offset = initial_elevation
    return azimuth_offset, elevation_offset

def calculate_azimuth_meters(x1, y1, x2, y2):
    """
    Calculate azimuth angle between two points in meters.
    """
    azimuth = np.degrees(np.arctan2(y2 - y1, x2 - x1))
    return azimuth

def calculate_elevation_meters(altitude, x1, y1, x2, y2, reference_altitude):
    """
    Calculate elevation angle between two points in meters.
    """
    horizontal_distance = calculate_horizontal_distance_meters(x1, y1, x2, y2)
    relative_altitude = altitude - reference_altitude
    return np.degrees(np.arctan2(relative_altitude, horizontal_distance))

def calculate_total_distance_meters(x1, y1, x2, y2, alt1, alt2):
    """
    Calculate total distance between two points in meters, including altitude difference.
    """
    horizontal_distance = calculate_horizontal_distance_meters(x1, y1, x2, y2)
    altitude_difference = alt2 - alt1
    total_distance = np.sqrt(horizontal_distance**2 + altitude_difference**2)
    return total_distance

def calculate_horizontal_distance_meters(x1, y1, x2, y2):
    """
    Calculate horizontal distance between two points in meters.
    """
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def wrap_angle(angle):
    """
    Wrap angle to the range [-180, 180] degrees.
    """
    return ((angle + 180) % 360) - 180


def update_flight_data(flight_data, index, ref_x, ref_y, initial_altitude, azimuth_offset, elevation_offset):
    """
    Updates flight data for the current frame and computes the corresponding azimuth and elevation.

    Parameters:
        flight_data (pd.DataFrame): Flight data containing position (and altitude) information.
        index (int): Index of the current flight data sample.
        ref_x (float): Reference x-coordinate.
        ref_y (float): Reference y-coordinate.
        initial_altitude (float): Initial altitude.
        azimuth_offset (float): Azimuth offset (degrees).
        elevation_offset (float): Elevation offset (degrees).

    Returns:
        csv_azimuth (float): Calculated azimuth (degrees) for the current flight data sample.
        csv_elevation (float): Calculated elevation (degrees) for the current sample.
        total_distance (float): Total distance traveled (meters).
    """
    x = flight_data.iloc[index]['X_meters']
    y = flight_data.iloc[index]['Y_meters']
    # Depending on your CSV structure, adjust the altitude column name accordingly.
    #altitude = flight_data.iloc[index]['OSD.altitude [ft]']
    altitude = flight_data.iloc[index]['altitude_above_seaLevel(feet)']

    # Compute the azimuth from reference to current position and apply offset
    csv_azimuth = calculate_azimuth_meters(ref_x, ref_y, x, y) + azimuth_offset
    csv_azimuth = wrap_angle(csv_azimuth)
    # Invert azimuth if needed by your beamforming convention
    csv_azimuth = -csv_azimuth

    # Compute elevation based on altitude difference and horizontal displacement, plus offset
    csv_elevation = calculate_elevation_meters(altitude, ref_x, ref_y, x, y, initial_altitude) + elevation_offset

    total_distance = calculate_total_distance_meters(ref_x, ref_y, x, y, initial_altitude, altitude)
    return csv_azimuth, csv_elevation, total_distance







#### Prueba

#
# config = get_experiment_config(1)  # Retrieve parameters for experiment 11
# ref_file_path = config["ref_file_path"]
# file_path_flight = config["file_path_flight"]
# print(ref_file_path)
# print(type(ref_file_path))
#
#
# flight_data, ref_x, ref_y, initial_altitude = load_flight_data(0, ref_file_path, file_path_flight)
#
# print(flight_data, ref_x, ref_y, initial_altitude)