#!/usr/bin/env python3
"""
Simulate drone flight audio using the same delay calculations as used in beamforming.
The simulation generates simulated multichannel WAV files (4 devices, 6 channels each)
that are fully compatible with the beamforming code.
"""

import numpy as np
import os
import soundfile as sf
from scipy.signal import resample
from tqdm import tqdm
from pyproj import Transformer

from Utilities.Acoustic import iso9613_attenuation_factor
from Utilities.functions import initialize_microphone_positions_24
from Utilities.geo_utils import compute_relative_flight_positions, calculate_delays_for_direction


def simulate_drone_flight_beamform_based(
        reference_csv,
        flight_csv,
        drone_audio_file,
        freq_approx=1000.0,  # Frequency for ISO 9613 attenuation (Hz)
        output_folder='sim_beamform_based',
        sample_rate=96000,
        dt_flight=0.1,  # Time step between flight positions (seconds)
        sound_speed=343.0,  # Speed of sound (m/s)
        temperature=20.0,  # Ambient temperature (Celsius)
        humidity=70.0,  # Relative humidity (%)
        pressure_kpa=101.325,  # Atmospheric pressure (kPa)
        epsilon=1e-12  # Small constant to avoid division by zero
):
    """
    Simulate drone flight audio using the same delay calculation as in beamforming.
    For each flight position, the arrival direction is computed as the normalized negative
    of the flight position vector (assuming the array is at the origin). Then, using
    calculate_delays_for_direction, we obtain the delays for each mic.

    Each extracted audio segment (from the drone_audio_file) is shifted according to the
    computed delays and attenuated using ISO 9613 (at a given frequency). Finally, the 24-channel
    simulated audio is grouped into 4 devices (6 channels each) and saved as WAV files.
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

    # 4) Process each flight position (simulate each time frame)
    for idx in tqdm(range(n_positions), desc="Simulating flight (beamforming-based delays)"):
        t_emit = idx * dt_flight
        sample_in_audio = int(t_emit * sample_rate)
        desired_segment_length = int(dt_flight * sample_rate)

        # Extract audio segment; pad with zeros if needed
        if sample_in_audio + desired_segment_length > len(raw_drone):
            segment = raw_drone[sample_in_audio:]
            segment = np.concatenate((segment, np.zeros(desired_segment_length - len(segment))))
        else:
            segment = raw_drone[sample_in_audio: sample_in_audio + desired_segment_length]

        # Normalize the segment to avoid saturation
        segment_norm = segment / (np.max(np.abs(segment)) + epsilon)

        # Compute the unit arrival direction vector
        # (Assuming the array is at the origin, the source (drone) is at flight_positions[idx])
        norm_flight = np.linalg.norm(flight_positions[idx]) + epsilon
        unit_direction = - flight_positions[idx] / norm_flight  # Negative: direction from source to array

        # Extract azimuth and elevation from the unit direction
        sim_az = np.degrees(np.arctan2(unit_direction[1], unit_direction[0]))
        sim_el = np.degrees(np.arcsin(unit_direction[2]))

        # Calculate delay samples using the same method as in beamforming
        delay_samples = calculate_delays_for_direction(mic_positions, sim_az, sim_el, sample_rate, sound_speed)

        # For each microphone, insert the delayed and attenuated segment into the global buffer
        for mic_idx in range(n_mics):
            # Compute the effective starting sample index for this mic
            sample_arrival = int(t_emit * sample_rate) + delay_samples[mic_idx]
            # Handle negative delays by trimming the beginning of the segment
            if sample_arrival < 0:
                trim = abs(sample_arrival)
                sample_arrival = 0
                effective_segment = segment_norm[trim:]
            else:
                effective_segment = segment_norm

            available_length = n_samples_total - sample_arrival
            seg_len = min(len(effective_segment), available_length)
            if seg_len > 0:
                # For attenuation, use the Euclidean distance between the mic and the flight position
                distance = np.linalg.norm(mic_positions[mic_idx] - flight_positions[idx]) + epsilon
                att_factor = iso9613_attenuation_factor(
                    distance=distance,
                    frequency=freq_approx,
                    temperature=temperature,
                    humidity=humidity,
                    pressure=pressure_kpa
                )
                # Add the delayed and attenuated segment to the mic's signal
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
        outpath = os.path.join(output_folder, f"device_{device_idx + 1}D96Dot.wav")
        sf.write(outpath, device_signal, sample_rate)
        print(f"Saved {outpath}")


if __name__ == '__main__':
    simulate_drone_flight_beamform_based(
        reference_csv="/Users/a30068385/OneDrive - Western Sydney University/FlightRecord/DJI Air 3/CSV/18 Mar 25/Ref/Mar-18th-2025-10-31AM-Flight-Airdata.csv",
        flight_csv="/Users/a30068385/OneDrive - Western Sydney University/FlightRecord/DJI Air 3/CSV/18 Mar 25/2/Mar-18th-2025-11-55AM-Flight-Airdata.csv",
        drone_audio_file="/Users/a30068385/OneDrive - Western Sydney University/recordings/Noise Ref/DJI_Air_Sound_20min.wav",
        freq_approx=1000.0,
        output_folder='sim_beamform_based'
    )
