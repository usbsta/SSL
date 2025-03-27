#!/usr/bin/env python3
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


def simulate_drone_flight_spherical(
        reference_csv,
        flight_csv,
        drone_audio_file,
        freq_approx=1000.0,  # Frequency for ISO 9613 attenuation (Hz)
        output_folder='sim_spherical',
        sample_rate=48000,
        dt_flight=0.1,  # Time step between flight positions (seconds)
        sound_speed=343.0,  # Speed of sound (m/s)
        temperature=20.0,  # Ambient temperature (°C)
        humidity=70.0,  # Relative humidity (%)
        pressure_kpa=101.325,  # Atmospheric pressure (kPa)
        epsilon=1e-12  # Small constant to avoid division by zero
):
    """
    Simulate the drone flight audio using spherical wave propagation.
    For each flight position (from CSV), the absolute Euclidean distances from the drone to each
    microphone are computed. The delays are calculated as:

         delay[mic] = (||mic_position - drone_position||) / sound_speed

    To simulate relative delays (so that one mic receives the signal at t_emit),
    we subtract the minimum delay among all mics.

    Each segment of the drone audio is then inserted into the microphone buffers at:
         sample_index = int(t_emit * sample_rate) + round((delay[mic] - min_delay)*sample_rate)

    The segment is also attenuated using ISO 9613 based on the actual distance.
    """
    # 1) Load microphone positions and flight positions
    mic_positions = initialize_microphone_positions_24()  # Shape: (24, 3)
    flight_positions = compute_relative_flight_positions(flight_csv, reference_csv)
    n_positions = flight_positions.shape[0]
    n_mics = mic_positions.shape[0]  # Should be 24

    total_duration = n_positions * dt_flight
    n_samples_total = int(total_duration * sample_rate)

    # 2) Load drone audio and resample if needed
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

        # Extract an audio segment; pad with zeros if necessary
        if sample_in_audio + segment_length > len(raw_drone):
            segment = raw_drone[sample_in_audio:]
            segment = np.concatenate((segment, np.zeros(segment_length - len(segment))))
        else:
            segment = raw_drone[sample_in_audio: sample_in_audio + segment_length]

        # Normalize the segment to avoid saturation
        segment_norm = segment / (np.max(np.abs(segment)) + epsilon)

        # For spherical propagation, compute delays based on full Euclidean distances.
        delays = np.zeros(n_mics)
        for mic_idx in range(n_mics):
            # Compute the Euclidean distance from the current flight position (drone) to the mic
            distance = np.linalg.norm(mic_positions[mic_idx] - flight_positions[idx]) + epsilon
            delays[mic_idx] = distance / sound_speed

        # To simulate relative delays, subtract the minimum delay (the earliest arrival)
        min_delay = delays.min()
        relative_delays = delays - min_delay  # These will be >= 0

        # For each microphone, insert the delayed and attenuated segment
        for mic_idx in range(n_mics):
            delay_samples = int(round(relative_delays[mic_idx] * sample_rate))
            sample_arrival = int(t_emit * sample_rate) + delay_samples

            # If delay causes negative indexing, trim the segment accordingly
            if sample_arrival < 0:
                trim = abs(sample_arrival)
                sample_arrival = 0
                effective_segment = segment_norm[trim:]
            else:
                effective_segment = segment_norm

            available_length = n_samples_total - sample_arrival
            seg_len = min(len(effective_segment), available_length)
            if seg_len > 0:
                # Use the full Euclidean distance (without subtracting min_delay) for attenuation
                distance = np.linalg.norm(mic_positions[mic_idx] - flight_positions[idx]) + epsilon
                att_factor = iso9613_attenuation_factor(
                    distance=distance,
                    frequency=freq_approx,
                    temperature=temperature,
                    humidity=humidity,
                    pressure=pressure_kpa
                )
                mic_signals[mic_idx, sample_arrival:sample_arrival + seg_len] += effective_segment[
                                                                                 :seg_len] * att_factor

    # 5) Group the 24-channel signal into 4 devices (6 channels each)
    n_devices = 4
    mics_per_device = 6
    devices = []
    for device_idx in range(n_devices):
        device_signal = mic_signals[device_idx * mics_per_device: (device_idx + 1) * mics_per_device, :]
        device_signal = device_signal.T  # Shape: (n_samples_total, 6)
        devices.append(device_signal)

    # 6) Save each device as a multichannel WAV file
    os.makedirs(output_folder, exist_ok=True)
    for device_idx, device_signal in enumerate(devices):
        device_signal = device_signal / (np.max(np.abs(device_signal)) + epsilon)
        outpath = os.path.join(output_folder, f"device_{device_idx + 1}SEu96.wav")
        sf.write(outpath, device_signal, sample_rate)
        print(f"Saved {outpath}")


if __name__ == '__main__':
    simulate_drone_flight_spherical(
        reference_csv="/Users/a30068385/OneDrive - Western Sydney University/FlightRecord/DJI Air 3/CSV/18 Mar 25/Ref/Mar-18th-2025-10-31AM-Flight-Airdata.csv",
        flight_csv="/Users/a30068385/OneDrive - Western Sydney University/FlightRecord/DJI Air 3/CSV/18 Mar 25/2/Mar-18th-2025-11-55AM-Flight-Airdata.csv",
        drone_audio_file="/Users/a30068385/OneDrive - Western Sydney University/recordings/Noise Ref/sine_20min.wav",
        freq_approx=1000.0,
        output_folder='sim_spherical'
    )
