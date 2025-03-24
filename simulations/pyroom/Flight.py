import numpy as np
import soundfile as sf
import os
from scipy.signal import resample, resample_poly
from tqdm import tqdm
from Utilities.functions import  initialize_microphone_positions_24
from Utilities.geo_utils import compute_relative_flight_positions

# --- CONFIGURATION ---
reference_csv = '/Users/30068385/OneDrive - Western Sydney University/FlightRecord/DJI Air 3/CSV/18 Mar 25/Ref/Mar-18th-2025-10-31AM-Flight-Airdata.csv'
flight_csv = '/Users/30068385/OneDrive - Western Sydney University/FlightRecord/DJI Air 3/CSV/18 Mar 25/1/Mar-18th-2025-11-19AM-Flight-Airdata.csv'

drone_audio_file = '/Users/30068385/OneDrive - Western Sydney University/recordings/Noise Ref/DJI_Air_Sound_20seg.wav'
output_folder = 'simulated_mics_wav'

sound_speed = 343  # m/s
sample_rate = 48000  # Hz
dt_flight = 0.1  # seconds between samples
max_pitch_shift = 1.1  # Max pitch scaling
speed_threshold = 10.0  # m/s (reference speed for pitch modulation)
epsilon = 1e-12

# --- LOAD MIC POSITIONS ---
mic_positions = initialize_microphone_positions_24()  # shape (24, 3)
# --- LOAD DRONE POSITIONS ---
flight_positions = compute_relative_flight_positions(flight_csv, reference_csv)


# --- LOAD DRONE AUDIO ---
drone_signal, sr = sf.read(drone_audio_file)
if sr != sample_rate:
    drone_signal = resample(drone_signal, int(len(drone_signal) * sample_rate / sr))
drone_signal = drone_signal / np.max(np.abs(drone_signal))

#Compute total duration in seconds and samples
drone_len_samples = len(drone_signal)
drone_duration_sec = drone_len_samples / sample_rate


# --- INIT BUFFERS ---
n_positions = flight_positions.shape[0]
n_mics = mic_positions.shape[0]
total_duration = n_positions * dt_flight
n_samples_total = int(total_duration * sample_rate)
mic_signals = np.zeros((n_mics, n_samples_total))

# --- CALCULATE DRONE VELOCITIES AND PITCH FACTORS ---
velocities = np.linalg.norm(np.diff(flight_positions, axis=0), axis=1) / dt_flight
velocities = np.concatenate(([velocities[0]], velocities))  # Match length
normalized_speed = np.clip(velocities / speed_threshold, 0, 1)
curve_factor = normalized_speed ** 2
pitch_factors = 1.0 + (max_pitch_shift - 1.0) * curve_factor  # Per frame pitch

# --- SIMULATE FLIGHT ---
for idx in tqdm(range(n_positions), desc="Simulating flight with looping audio"):
    drone_pos = flight_positions[idx]  # [X,Y,Z]
    t_emit = idx * dt_flight
    sample_emit = int(t_emit * sample_rate)

    distances = np.linalg.norm(mic_positions - drone_pos, axis=1) + epsilon
    delays = distances / sound_speed
    attenuation = 1.0 / distances

    pitch = pitch_factors[idx]

    # --- Looping audio: calculate sample start in drone_signal ---
    # Total offset time in drone audio (modulo duration)
    t_in_audio = (t_emit % drone_duration_sec)
    sample_in_audio = int(t_in_audio * sample_rate)
    segment = drone_signal[sample_in_audio:]  # From current point to end

    # Ensure we have enough length for resampling
    if len(segment) < 1000:
        # Pad if at end of file
        segment = np.concatenate((segment, drone_signal))

    # Apply pitch (Doppler)
    mod_drone = resample_poly(segment, up=int(pitch * 1000), down=1000)
    mod_drone = mod_drone / np.max(np.abs(mod_drone) + epsilon)

    for mic_idx in range(n_mics):
        t_arrival = t_emit + delays[mic_idx]
        sample_arrival = int(t_arrival * sample_rate)
        end_idx = min(sample_arrival + len(mod_drone), n_samples_total)
        segment_len = end_idx - sample_arrival
        if segment_len > 0:
            mic_signals[mic_idx, sample_arrival:end_idx] += mod_drone[:segment_len] * attenuation[mic_idx]


# --- NORMALIZE AND SAVE ---
os.makedirs(output_folder, exist_ok=True)
for mic_idx in range(n_mics):
    mic_wave = mic_signals[mic_idx]
    mic_wave = mic_wave / np.max(np.abs(mic_wave) + epsilon)
    sf.write(os.path.join(output_folder, f"mic_{mic_idx+1:02d}.wav"), mic_wave, sample_rate)
    print(f"Saved mic_{mic_idx+1:02d}.wav")
