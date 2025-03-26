"""
Simulate drone flight audio using a spherical wave propagation model.
Instead of assuming a planar wavefront, this simulation computes delays for each microphone
based on the full Euclidean distance between the drone (source) and each microphone.
The relative delays are then applied (by subtracting the minimum delay) so that the earliest
arrival is set to zero delay. The simulated 24-channel signal is then grouped into 4 devices
(6 channels each) and saved as multichannel WAV files.
"""

import numpy as np
import os
import soundfile as sf
from scipy.signal import resample
from tqdm import tqdm
from pyproj import Transformer

from Utilities.Acoustic import iso9613_attenuation_factor
from Utilities.functions import initialize_microphone_positions_24
from Utilities.geo_utils import compute_relative_flight_positions


def rotate_positions_z(positions, angle_degs):
    """
    Rotate a set of positions around the Z-axis by a given angle in degrees.
    positions: ndarray of shape (N, 3) with columns [x, y, z].
    angle_degs: float, rotation angle in degrees (positive means counterclockwise).

    Returns an ndarray (N, 3) with the rotated positions.
    """
    angle_rad = np.deg2rad(angle_degs)
    Rz = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), 0.0],
        [np.sin(angle_rad), np.cos(angle_rad), 0.0],
        [0.0, 0.0, 1.0]
    ])
    # Multiply (N, 3) x (3, 3) => (N, 3)
    return positions @ Rz.T


def simulate_drone_flight_spherical(
        reference_csv,
        flight_csv,
        drone_audio_file,
        freq_approx=1000.0,  # Frequency for ISO 9613 attenuation (Hz)
        output_folder='sim_spherical',
        sample_rate=48000,
        dt_flight=0.1,  # Time step between flight positions (seconds)
        sound_speed=343.0,  # Speed of sound in m/s
        temperature=20.0,  # Ambient temperature (°C)
        humidity=70.0,  # Relative humidity (%)
        pressure_kpa=101.325,  # Atmospheric pressure (kPa)
        epsilon=1e-12,
        azimuth_offset_degs=-15.0
):
    """
    Simulate drone flight audio using full spherical wave propagation and allow
    specifying an azimuth offset (in degrees).

    The positions from the flight_csv are rotated in the XY plane by azimuth_offset_degs,
    simulating that the drone enters at -15° (or any chosen value) offset in azimuth
    with respect to the reference microphone (mic 1).
    """
    # 1) Load microphone positions and drone flight positions
    mic_positions = initialize_microphone_positions_24()  # shape: (24, 3)

    # Read flight positions (without any offset initially)
    flight_positions = compute_relative_flight_positions(flight_csv, reference_csv)

    # Apply azimuth offset by rotating flight positions around Z
    if azimuth_offset_degs != 0.0:
        flight_positions = rotate_positions_z(flight_positions, azimuth_offset_degs)

    n_positions = flight_positions.shape[0]
    n_mics = mic_positions.shape[0]  # Should be 24
    total_duration = n_positions * dt_flight
    n_samples_total = int(total_duration * sample_rate)

    # 2) Load drone audio file and resample if necessary
    raw_drone, sr_drone = sf.read(drone_audio_file)
    if sr_drone != sample_rate:
        factor = sample_rate / sr_drone
        raw_drone = resample(raw_drone, int(len(raw_drone) * factor))

    # 3) Initialize an output buffer for each microphone
    mic_signals = np.zeros((n_mics, n_samples_total))

    # 4) Process each flight position
    for idx in tqdm(range(n_positions), desc="Simulating flight (spherical propagation)"):
        t_emit = idx * dt_flight
        sample_in_audio = int(t_emit * sample_rate)
        segment_length = int(dt_flight * sample_rate)

        # Extract a segment from the drone audio, pad with zeros if it goes beyond its length
        if sample_in_audio + segment_length > len(raw_drone):
            segment = raw_drone[sample_in_audio:]
            segment = np.concatenate((segment, np.zeros(segment_length - len(segment))))
        else:
            segment = raw_drone[sample_in_audio: sample_in_audio + segment_length]

        # Normalize segment to avoid saturations
        segment_norm = segment / (np.max(np.abs(segment)) + epsilon)

        # Compute full Euclidean distance to each mic (spherical propagation)
        delays = np.zeros(n_mics)
        for mic_idx in range(n_mics):
            distance = np.linalg.norm(mic_positions[mic_idx] - flight_positions[idx]) + epsilon
            delays[mic_idx] = distance / sound_speed

        # Compute relative delays by subtracting the earliest arrival
        min_delay = delays.min()
        relative_delays = delays - min_delay

        # For each mic, insert the delayed and attenuated segment into the buffer
        for mic_idx in range(n_mics):
            delay_samples = int(round(relative_delays[mic_idx] * sample_rate))
            sample_arrival = int(t_emit * sample_rate) + delay_samples

            # If delay makes the segment start before index 0, trim it
            if sample_arrival < 0:
                trim = abs(sample_arrival)
                sample_arrival = 0
                effective_segment = segment_norm[trim:]
            else:
                effective_segment = segment_norm

            available_length = n_samples_total - sample_arrival
            seg_len = min(len(effective_segment), available_length)
            if seg_len > 0:
                # Use the full distance (no offset) for the attenuation factor
                distance = np.linalg.norm(mic_positions[mic_idx] - flight_positions[idx]) + epsilon
                att_factor = iso9613_attenuation_factor(
                    distance=distance,
                    frequency=freq_approx,
                    temperature=temperature,
                    humidity=humidity,
                    pressure=pressure_kpa
                )
                mic_signals[mic_idx, sample_arrival:sample_arrival + seg_len] += (
                        effective_segment[:seg_len] * att_factor
                )

    # 5) Group the 24 channels into 4 devices (6 channels each) and save them
    n_devices = 4
    mics_per_device = 6
    os.makedirs(output_folder, exist_ok=True)

    for device_idx in range(n_devices):
        device_signal = mic_signals[device_idx * mics_per_device:(device_idx + 1) * mics_per_device, :]
        device_signal = device_signal.T  # shape: (n_samples_total, 6)

        # Normalize again to avoid clipping in the final file
        device_signal = device_signal / (np.max(np.abs(device_signal)) + epsilon)
        outpath = os.path.join(output_folder, f"device_{device_idx + 1}SEu192.wav")
        sf.write(outpath, device_signal, sample_rate)
        print(f"Saved {outpath}")


if __name__ == '__main__':
    # Example usage
    simulate_drone_flight_spherical(
        reference_csv="/Users/30068385/OneDrive - Western Sydney University/FlightRecord/DJI Air 3/CSV/18 Mar 25/Ref/Mar-18th-2025-10-31AM-Flight-Airdata.csv",
        flight_csv="/Users/30068385/OneDrive - Western Sydney University/FlightRecord/DJI Air 3/CSV/18 Mar 25/2/Mar-18th-2025-11-55AM-Flight-Airdata.csv",
        drone_audio_file="/Users/30068385/OneDrive - Western Sydney University/recordings/Noise Ref/sine_20min.wav",
        freq_approx=1000.0,
        output_folder='sim_spherical',
        sample_rate=192000,
        dt_flight=0.1,
        sound_speed=343.0,
        temperature=20.0,
        humidity=70.0,
        pressure_kpa=101.325,
        azimuth_offset_degs=90.0  # The desired azimuth offset in degrees
    )