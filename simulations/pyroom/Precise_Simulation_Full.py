# simulate_flight_banded.py
"""
Main simulation script that uses acoustic_utils_iso9613 in a band-by-band approach.

Workflow:
1) Load mic positions, flight positions.
2) Load and calibrate drone audio if needed.
3) Filter the drone signal into octave bands.
4) For each flight position, for each band:
   - loop audio portion
   - pitch shift for Doppler
   - compute ISO 9613 attenuation at band center frequency
   - apply delay and accumulate in mic buffers
5) Export final waveforms.

If you want to add noise with real or changed SNR, do so after.
"""

import numpy as np
import soundfile as sf
import os
from scipy.signal import resample, resample_poly
from tqdm import tqdm

from Utilities.Acoustic import (filter_octave_band,iso9613_attenuation_factor)
from Utilities.functions import initialize_microphone_positions_24
from Utilities.geo_utils import compute_relative_flight_positions


def simulate_drone_flight_banded(
    reference_csv,
    flight_csv,
    drone_audio_file,
    center_frequencies=[125, 250, 500, 1000, 2000, 4000],
    output_folder='simulated_banded',
    sample_rate=48000,
    dt_flight=0.1,
    sound_speed=343.0,
    max_pitch_shift=1.1,
    speed_threshold=10.0,
    temperature=28.0,
    humidity=70.0,
    pressure_kpa=101.925,
    epsilon=1e-12
):
    """
    Simulate the drone flight using a band-by-band approach with ISO 9613 attenuation.
    Each band is filtered, pitched, delayed, and attenuated separately, then summed.

    Args:
        reference_csv (str): CSV with reference point (0,0,0).
        flight_csv (str): CSV with flight data.
        drone_audio_file (str): Path to the drone WAV.
        center_frequencies (list): List of octave band center frequencies.
        output_folder (str): Where to save the multi-mic WAVs.
        sample_rate (int): Audio sample rate.
        dt_flight (float): Time step between flight positions.
        sound_speed (float): Speed of sound in m/s.
        max_pitch_shift (float): Max pitch factor.
        speed_threshold (float): Speed (m/s) that sets pitch shift range.
        temperature (float): Celsius.
        humidity (float): Relative humidity in %.
        pressure_kpa (float): Pressure in kPa.
        epsilon (float): small constant.
    """

    # 1) Load mic positions + flight
    mic_positions = initialize_microphone_positions_24()   # shape (24,3)
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

    # 3) Filter the drone in each band
    band_signals = {}
    for f_center in center_frequencies:
        band_signals[f_center] = filter_octave_band(
            signal=raw_drone,
            center_freq=f_center,
            fs=sample_rate,
            order=4
        )

    # 4) Prepare final buffers for each mic
    mic_signals = np.zeros((n_mics, n_samples_total))

    # 5) Calculate velocity -> pitch factor
    velocities = np.linalg.norm(np.diff(flight_positions, axis=0), axis=1) / dt_flight
    velocities = np.concatenate(([velocities[0]], velocities))
    norm_speed = np.clip(velocities / speed_threshold, 0, 1)
    pitch_factors = 1.0 + (max_pitch_shift - 1.0)*(norm_speed**2)

    # Info for looping audio
    drone_len = len(raw_drone)
    drone_duration = drone_len / sample_rate

    # 6) Main loop: for each flight position
    for idx in tqdm(range(n_positions), desc="Simulating flight in octave bands"):
        t_emit = idx * dt_flight
        drone_pos = flight_positions[idx]  # [X, Y, Z]

        distances = np.linalg.norm(mic_positions - drone_pos, axis=1) + epsilon
        delays = distances / sound_speed

        pitch = pitch_factors[idx]

        # For each band
        for f_center in center_frequencies:
            # Step (a): loop portion of band signal
            t_in_audio = (t_emit % drone_duration)
            sample_in_audio = int(t_in_audio * sample_rate)
            source_band = band_signals[f_center]
            segment = source_band[sample_in_audio:]

            # If near end, wrap
            if len(segment) < 2000:
                wrap_needed = 2000 - len(segment)
                segment = np.concatenate((segment, source_band[:wrap_needed]))

            # Step (b): pitch shift
            mod_band = resample_poly(segment, up=int(pitch*1000), down=1000)
            mod_band = mod_band / (np.max(np.abs(mod_band)) + epsilon)

            # Step (c): apply to each mic
            # iso9613 at freq = f_center
            for mic_idx in range(n_mics):
                dist = distances[mic_idx]
                t_arrival = t_emit + delays[mic_idx]
                sample_arrival = int(t_arrival*sample_rate)

                att_factor = iso9613_attenuation_factor(
                    distance=dist,
                    frequency=f_center,
                    temperature=temperature,
                    humidity=humidity,
                    pressure=pressure_kpa
                )

                end_idx = min(sample_arrival+len(mod_band), n_samples_total)
                seg_len = end_idx - sample_arrival
                if seg_len > 0:
                    mic_signals[mic_idx, sample_arrival:end_idx] += mod_band[:seg_len]*att_factor

    # 7) Save result
    os.makedirs(output_folder, exist_ok=True)
    for m in range(n_mics):
        sig_m = mic_signals[m]
        sig_m = sig_m / (np.max(np.abs(sig_m))+1e-12)
        sf.write(os.path.join(output_folder, f"mic_{m+1:02d}_precise.wav"), sig_m, sample_rate)
        print(f"Saved mic_{m+1:02d}.wav")


# Example usage:
if __name__ == "__main__":
    simulate_drone_flight_banded(
        reference_csv="/Users/30068385/OneDrive - Western Sydney University/FlightRecord/DJI Air 3/CSV/18 Mar 25/Ref/Mar-18th-2025-10-31AM-Flight-Airdata.csv",
        flight_csv="/Users/30068385/OneDrive - Western Sydney University/FlightRecord/DJI Air 3/CSV/18 Mar 25/1/Mar-18th-2025-11-19AM-Flight-Airdata.csv",
        drone_audio_file="/Users/30068385/OneDrive - Western Sydney University/recordings/Noise Ref/DJI_Air_Sound_30min.wav",
        center_frequencies=[125,250,500,1000,2000,4000,8000],
        output_folder="sim_banded_octave"
    )
