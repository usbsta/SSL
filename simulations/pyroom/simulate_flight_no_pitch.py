import numpy as np
import soundfile as sf
import os
from scipy.signal import resample
from tqdm import tqdm

from Utilities.Acoustic import iso9613_attenuation_factor
from Utilities.functions import initialize_microphone_positions_24
from Utilities.geo_utils import compute_relative_flight_positions


def simulate_drone_flight_singleband_no_pitch(
        reference_csv,
        flight_csv,
        drone_audio_file,
        freq_approx=1000.0,  # single freq approximation
        output_folder='simulated_singleband_nopitch',
        sample_rate=48000,
        dt_flight=0.1,
        sound_speed=343.0,
        temperature=20.0,
        humidity=70.0,
        pressure_kpa=101.325,
        epsilon=1e-12
):
    """
    Simulate the drone flight using a SINGLE-FREQUENCY approach with ISO 9613 attenuation,
    but WITHOUT pitch shift (no Doppler effect).

    Workflow:
    1) Load mic positions & flight CSV
    2) Load drone audio (optionally resample)
    3) For each flight position:
        - 'loop' a portion of the audio
        - compute single freq (ISO 9613) attenuation
        - apply time-of-arrival delay
    4) Sum into mic buffers
    5) Save final multi-mic WAVs
    """

    # 1) Load positions
    mic_positions = initialize_microphone_positions_24()  # (24, 3)
    flight_positions = compute_relative_flight_positions(flight_csv, reference_csv)
    n_positions = flight_positions.shape[0]
    n_mics = mic_positions.shape[0]

    total_duration = n_positions * dt_flight
    n_samples_total = int(total_duration * sample_rate)

    # 2) Load drone audio
    raw_drone, sr_drone = sf.read(drone_audio_file)
    if sr_drone != sample_rate:
        factor = sample_rate / sr_drone
        raw_drone = resample(raw_drone, int(len(raw_drone) * factor))

    drone_len = len(raw_drone)
    drone_duration = drone_len / sample_rate

    # 3) Prepare final buffers
    mic_signals = np.zeros((n_mics, n_samples_total))

    # 4) Main loop: no pitch factor, just distances & attenuation
    from numpy.linalg import norm

    for idx in tqdm(range(n_positions), desc="Simulating flight (no pitch)"):
        t_emit = idx * dt_flight
        drone_pos = flight_positions[idx]  # [X, Y, Z]

        # Distances to each mic
        distances = norm(mic_positions - drone_pos, axis=1) + epsilon
        delays = distances / sound_speed

        # Take a small chunk from raw_drone
        t_in_audio = (t_emit % drone_duration)
        sample_in_audio = int(t_in_audio * sample_rate)
        segment = raw_drone[sample_in_audio:]

        # If near end, wrap a bit
        needed = 2000  # ~2000 samples chunk
        if len(segment) < needed:
            wrap_needed = needed - len(segment)
            segment = np.concatenate((segment, raw_drone[:wrap_needed]))

        # Normalizing chunk to avoid saturation (optional)
        max_val = np.max(np.abs(segment))
        if max_val > 1e-12:
            segment = segment / max_val

        # For each mic
        for mic_idx in range(n_mics):
            dist = distances[mic_idx]
            t_arrival = t_emit + delays[mic_idx]
            sample_arrival = int(t_arrival * sample_rate)

            # Single freq attenuation (no Doppler)
            att_factor = iso9613_attenuation_factor(
                distance=dist,
                frequency=freq_approx,
                temperature=temperature,
                humidity=humidity,
                pressure=pressure_kpa
            )

            end_idx = min(sample_arrival + needed, n_samples_total)
            seg_len = end_idx - sample_arrival
            if seg_len > 0:
                mic_signals[mic_idx, sample_arrival:end_idx] += segment[:seg_len] * att_factor

    # 5) Save final result
    os.makedirs(output_folder, exist_ok=True)
    for m in range(n_mics):
        sig_m = mic_signals[m]
        sig_m = sig_m / (np.max(np.abs(sig_m)) + epsilon)
        outpath = os.path.join(output_folder, f"mic_{m + 1:02d}_NoPitch.wav")
        sf.write(outpath, sig_m, sample_rate)
        print(f"Saved {outpath}")


# Example usage
if __name__ == '__main__':
    simulate_drone_flight_singleband_no_pitch(
        reference_csv="/Users/30068385/OneDrive - Western Sydney University/FlightRecord/DJI Air 3/CSV/18 Mar 25/Ref/Mar-18th-2025-10-31AM-Flight-Airdata.csv",
        flight_csv="/Users/30068385/OneDrive - Western Sydney University/FlightRecord/DJI Air 3/CSV/18 Mar 25/2/Mar-18th-2025-11-55AM-Flight-Airdata2.csv",
        drone_audio_file="/Users/30068385/OneDrive - Western Sydney University/recordings/Noise Ref/DJI_Air_Sound_20min.wav",
        freq_approx=1000.0,
        output_folder='sim_singleband_nopitch'
    )
