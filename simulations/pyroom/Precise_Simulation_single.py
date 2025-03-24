# simulate_flight_singleband.py
"""
Main simulation script using a SINGLE-BAND approximation for ISO 9613.

Workflow:
1) Load mic positions, flight positions.
2) Load (and optionally calibrate) drone audio.
3) For each flight position:
   - loop portion of the audio
   - pitch shift for Doppler
   - compute ISO 9613 attenuation at a single freq (e.g., 1000 Hz or user-chosen)
   - apply time-of-arrival delay
4) Sum the resulting signal in mic buffers.
5) Export final waveforms.

If desired, add noise afterwards.
"""

import numpy as np
import soundfile as sf
import os
from scipy.signal import resample, resample_poly
from tqdm import tqdm

from Utilities.Acoustic import (filter_octave_band,iso9613_attenuation_factor)
from Utilities.functions import initialize_microphone_positions_24
from Utilities.geo_utils import compute_relative_flight_positions

def simulate_drone_flight_singleband(
    reference_csv,
    flight_csv,
    drone_audio_file,
    freq_approx=1000.0,           # single freq approximation
    output_folder='simulated_singleband',
    sample_rate=48000,
    dt_flight=0.1,
    sound_speed=343.0,
    max_pitch_shift=1.1,
    speed_threshold=10.0,
    temperature=20.0,
    humidity=70.0,
    pressure_kpa=101.325,
    epsilon=1e-12
):
    """
    Simulate the drone flight using a SINGLE-FREQUENCY approach with ISO 9613 attenuation.
    The entire audio is treated as if it were at `freq_approx`.

    Args:
        reference_csv (str): CSV with reference point (0,0,0).
        flight_csv (str): CSV with flight data.
        drone_audio_file (str): Path to the drone WAV.
        freq_approx (float): The single frequency (Hz) to approximate.
        output_folder (str): Where to save the multi-mic WAVs.
        sample_rate (int): Audio sample rate.
        dt_flight (float): Time step between flight positions.
        sound_speed (float): Speed of sound in m/s.
        max_pitch_shift (float): Max pitch factor.
        speed_threshold (float): Speed (m/s) that saturates pitch shift.
        temperature (float): Celsius.
        humidity (float): %.
        pressure_kpa (float): kPa.
        epsilon (float): small.
    """
    # 1) Load mic positions + flight
    mic_positions = initialize_microphone_positions_24()  # shape (24,3)
    flight_positions = compute_relative_flight_positions(flight_csv, reference_csv)
    n_positions = flight_positions.shape[0]
    n_mics = mic_positions.shape[0]

    total_duration = n_positions * dt_flight
    n_samples_total = int(total_duration * sample_rate)

    # 2) Load drone audio
    raw_drone, sr_drone = sf.read(drone_audio_file)
    if sr_drone != sample_rate:
        factor = sample_rate / sr_drone
        raw_drone = resample(raw_drone, int(len(raw_drone)*factor))

    # 3) Prepare final buffers for each mic
    mic_signals = np.zeros((n_mics, n_samples_total))

    # 4) Calculate velocity -> pitch factor
    velocities = np.linalg.norm(np.diff(flight_positions, axis=0), axis=1) / dt_flight
    velocities = np.concatenate(([velocities[0]], velocities))
    norm_speed = np.clip(velocities / speed_threshold, 0, 1)
    pitch_factors = 1.0 + (max_pitch_shift - 1.0)*(norm_speed**2)

    # Info for looping audio
    drone_len = len(raw_drone)
    drone_duration = drone_len / sample_rate

    # 5) Main loop: for each flight position
    for idx in tqdm(range(n_positions), desc="Simulating flight single freq"):
        t_emit = idx * dt_flight
        drone_pos = flight_positions[idx]  # [X, Y, Z]

        distances = np.linalg.norm(mic_positions - drone_pos, axis=1) + epsilon
        delays = distances / sound_speed

        pitch = pitch_factors[idx]

        # Loop portion of the audio
        t_in_audio = (t_emit % drone_duration)
        sample_in_audio = int(t_in_audio * sample_rate)
        segment = raw_drone[sample_in_audio:]

        # If near end, wrap
        if len(segment) < 2000:
            wrap_needed = 2000 - len(segment)
            segment = np.concatenate((segment, raw_drone[:wrap_needed]))

        # Apply pitch shift
        mod_drone = resample_poly(segment, up=int(pitch*1000), down=1000)
        mod_drone = mod_drone / (np.max(np.abs(mod_drone)) + epsilon)

        # For each mic
        for mic_idx in range(n_mics):
            dist = distances[mic_idx]
            t_arrival = t_emit + delays[mic_idx]
            sample_arrival = int(t_arrival * sample_rate)

            # Single freq attenuation
            att_factor = iso9613_attenuation_factor(
                distance=dist,
                frequency=freq_approx,
                temperature=temperature,
                humidity=humidity,
                pressure=pressure_kpa
            )

            end_idx = min(sample_arrival+len(mod_drone), n_samples_total)
            seg_len = end_idx - sample_arrival
            if seg_len > 0:
                mic_signals[mic_idx, sample_arrival:end_idx] += mod_drone[:seg_len]*att_factor

    # 6) Save final result
    os.makedirs(output_folder, exist_ok=True)
    for m in range(n_mics):
        sig_m = mic_signals[m]
        sig_m = sig_m / (np.max(np.abs(sig_m))+epsilon)
        outpath = os.path.join(output_folder, f"mic_{m+1:02d}_Precise.wav")
        sf.write(outpath, sig_m, sample_rate)
        print(f"Saved {outpath}")

# Example usage:
if __name__ == '__main__':
     simulate_drone_flight_singleband(
        reference_csv="/Users/30068385/OneDrive - Western Sydney University/FlightRecord/DJI Air 3/CSV/18 Mar 25/Ref/Mar-18th-2025-10-31AM-Flight-Airdata.csv",
        flight_csv="/Users/30068385/OneDrive - Western Sydney University/FlightRecord/DJI Air 3/CSV/18 Mar 25/2/Mar-18th-2025-11-55AM-Flight-Airdata2.csv",
        drone_audio_file="/Users/30068385/OneDrive - Western Sydney University/recordings/Noise Ref/DJI_Air_Sound_20min.wav",
         freq_approx=1000.0,
         output_folder='sim_singleband'
     )
