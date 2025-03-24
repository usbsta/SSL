#!/usr/bin/env python3
"""
Simulate drone flight audio using a single-frequency (no Doppler) approximation.
This script generates simulated multichannel WAV files (4 devices, 6 channels each)
that are compatible with the beamforming code. The CSV processing for flight positions
has been adapted to match the beamforming transformation.
"""

import numpy as np
import os
import soundfile as sf
from scipy.signal import resample
from tqdm import tqdm
from pyproj import Transformer

from Utilities.Acoustic import iso9613_attenuation_factor
from Utilities.functions import initialize_microphone_positions_24

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

def simulate_drone_flight_singleband_no_pitch(
    reference_csv,
    flight_csv,
    drone_audio_file,
    freq_approx=1000.0,         # Frequency used for ISO 9613 attenuation approximation (Hz)
    output_folder='sim_singleband_nopitch',
    sample_rate=48000,
    dt_flight=0.1,              # Time step between flight positions (seconds)
    sound_speed=343.0,          # Speed of sound in m/s
    temperature=20.0,           # Ambient temperature in Celsius
    humidity=70.0,              # Relative humidity in %
    pressure_kpa=101.325,       # Atmospheric pressure in kPa
    epsilon=1e-12               # Small constant to avoid division by zero
):
    """
    Simulate the drone flight using a single-frequency approach with ISO 9613 attenuation,
    without applying Doppler shift. The drone audio is segmented based on flight positions,
    delayed and attenuated for each microphone channel, and then the 24 microphone signals are grouped
    into 4 multi-channel WAV files (each device contains 6 channels).

    Args:
        reference_csv (str): CSV file with the reference point (0,0,0).
        flight_csv (str): CSV file with the flight data.
        drone_audio_file (str): Path to the drone WAV file.
        freq_approx (float): The single frequency (Hz) for attenuation approximation.
        output_folder (str): Directory to save the output WAV files.
        sample_rate (int): Audio sample rate.
        dt_flight (float): Time step between flight positions.
        sound_speed (float): Speed of sound (m/s).
        temperature (float): Ambient temperature in Celsius.
        humidity (float): Relative humidity in %.
        pressure_kpa (float): Atmospheric pressure in kPa.
        epsilon (float): A small constant to avoid division by zero.
    """
    # 1) Load microphone positions and flight positions
    mic_positions = initialize_microphone_positions_24()  # Expected shape: (24, 3)
    # Use the modified CSV function to compute flight positions consistently with beamforming
    flight_positions = compute_relative_flight_positions(flight_csv, reference_csv)
    n_positions = flight_positions.shape[0]
    n_mics = mic_positions.shape[0]  # Should be 24

    total_duration = n_positions * dt_flight
    n_samples_total = int(total_duration * sample_rate)

    # 2) Load drone audio and resample if necessary
    raw_drone, sr_drone = sf.read(drone_audio_file)
    if sr_drone != sample_rate:
        factor = sample_rate / sr_drone
        raw_drone = resample(raw_drone, int(len(raw_drone) * factor))

    # 3) Initialize a buffer for each microphone signal
    mic_signals = np.zeros((n_mics, n_samples_total))

    # 4) Process each flight position
    for idx in tqdm(range(n_positions), desc="Simulating flight (no Doppler shift)"):
        t_emit = idx * dt_flight
        sample_in_audio = int(t_emit * sample_rate)
        desired_segment_length = int(dt_flight * sample_rate)

        # Extract an audio segment; pad with zeros if needed
        if sample_in_audio + desired_segment_length > len(raw_drone):
            segment = raw_drone[sample_in_audio:]
            segment = np.concatenate((segment, np.zeros(desired_segment_length - len(segment))))
        else:
            segment = raw_drone[sample_in_audio: sample_in_audio + desired_segment_length]

        # Normalize the segment to avoid saturation (no Doppler shift is applied)
        segment_norm = segment / (np.max(np.abs(segment)) + epsilon)

        # For each microphone, apply time-of-arrival delay and ISO 9613 attenuation
        for mic_idx in range(n_mics):
            # Compute the Euclidean distance between the current flight position and microphone position
            distance = np.linalg.norm(mic_positions[mic_idx] - flight_positions[idx]) + epsilon

            # Calculate the time-of-arrival at the microphone
            t_arrival = t_emit + (distance / sound_speed)
            sample_arrival = int(t_arrival * sample_rate)

            # Compute the attenuation factor based on ISO 9613 at the specified frequency
            att_factor = iso9613_attenuation_factor(
                distance=distance,
                frequency=freq_approx,
                temperature=temperature,
                humidity=humidity,
                pressure=pressure_kpa
            )

            # Determine the segment length that fits in the overall simulation buffer
            end_idx = min(sample_arrival + len(segment_norm), n_samples_total)
            seg_len = end_idx - sample_arrival
            if seg_len > 0:
                mic_signals[mic_idx, sample_arrival:end_idx] += segment_norm[:seg_len] * att_factor

    # 5) Group the 24 microphone signals into 4 devices (6 microphones each)
    n_devices = 4
    mics_per_device = 6
    devices = []
    for device_idx in range(n_devices):
        # Extract signals for the current device's 6 microphones
        device_signal = mic_signals[device_idx * mics_per_device : (device_idx + 1) * mics_per_device, :]
        # Transpose to shape (n_samples_total, 6) for multichannel WAV file writing
        device_signal = device_signal.T
        devices.append(device_signal)

    # 6) Save each device as a multichannel WAV file
    os.makedirs(output_folder, exist_ok=True)
    for device_idx, device_signal in enumerate(devices):
        # Normalize the signal to avoid clipping
        device_signal = device_signal / (np.max(np.abs(device_signal)) + epsilon)
        outpath = os.path.join(output_folder, f"device_{device_idx+1}.wav")
        sf.write(outpath, device_signal, sample_rate)
        print(f"Saved {outpath}")

# Example usage
if __name__ == '__main__':
    simulate_drone_flight_singleband_no_pitch(
        reference_csv="/Users/a30068385/OneDrive - Western Sydney University/FlightRecord/DJI Air 3/CSV/18 Mar 25/Ref/Mar-18th-2025-10-31AM-Flight-Airdata.csv",
        flight_csv="/Users/a30068385/OneDrive - Western Sydney University/FlightRecord/DJI Air 3/CSV/18 Mar 25/2/Mar-18th-2025-11-55AM-Flight-Airdata.csv",
        drone_audio_file="/Users/a30068385/OneDrive - Western Sydney University/recordings/Noise Ref/DJI_Air_Sound_20min.wav",
        freq_approx=1000.0,
        output_folder='sim_singleband_nopitch'
    )
