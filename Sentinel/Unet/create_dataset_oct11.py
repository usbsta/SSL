"""
create_dataset_1.py

This script performs delay-and-sum beamforming on multi-microphone audio recordings
while reading drone flight data from CSV logs. It synchronizes them chunk-by-chunk,
computes the “true” azimuth/elevation from flight data, and saves a labeled dataset
where each .npz file contains:
  - Beamformed audio chunks for every angle in the az/el grid
  - A binary label (0 or 1) depending on how close the angle is to the “true” angle
  - Drone metadata (altitude, distance, speed, true az/el, etc.)

The intention is that you can later load these .npz files to extract features (e.g.,
STFT/FFT) for machine learning tasks without having to re-run the beamforming.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pyproj import Transformer
from matplotlib.animation import FFMpegWriter
from numba import njit

# Local imports
from audio_beamforming import beamform_time  # Provided externally
from geo_utils import (
    wrap_angle,
    calculate_angle_difference,
    calculate_azimuth_meters,
    calculate_elevation_meters,
    calculate_total_distance_meters
)
from io_utils import (
    read_wav_block,
    apply_bandpass_filter,
    calculate_time,
    initialize_beamforming_params,
    open_wav_files
)

def skip_wav_seconds(wav_file, seconds, rate, extra_samples=0):
    """
    Skip frames in a WAV file to align audio with the flight data timeline.
    """
    frames_to_skip = int(seconds * rate) + extra_samples
    wav_file.setpos(frames_to_skip)

def format_time_s(total_seconds: float) -> str:
    """
    Convert a time in seconds to a string 'MM:SS.ss'.
    Example: 125.7 -> '02:05.70'
    """
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:05.2f}"

def calculate_angular_distance(az1, el1, az2, el2):
    """
    Computes the angular distance (in degrees) between two directions
    defined by (azimuth, elevation).
    """
    az1_rad, el1_rad = np.radians(az1), np.radians(el1)
    az2_rad, el2_rad = np.radians(az2), np.radians(el2)

    # Cartesian conversion
    x1 = np.cos(el1_rad) * np.cos(az1_rad)
    y1 = np.cos(el1_rad) * np.sin(az1_rad)
    z1 = np.sin(el1_rad)

    x2 = np.cos(el2_rad) * np.cos(az2_rad)
    y2 = np.cos(el2_rad) * np.sin(az2_rad)
    z2 = np.sin(el2_rad)

    dot_product = x1 * x2 + y1 * y2 + z1 * z2
    dot_product = np.clip(dot_product, -1.0, 1.0)
    angular_distance_rad = np.arccos(dot_product)
    angular_distance_deg = np.degrees(angular_distance_rad)
    return angular_distance_deg

@njit
def shift_signal_beamforming(signal, delay_samples):
    """
    JIT-compiled function to shift a 1D signal by a given number of samples
    for beamforming alignment.
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

def beamform_in_direction(signal_data, mic_positions, azimuth, elevation,
                          sample_rate, speed_of_sound):
    """
    Performs delay-and-sum beamforming for one (azimuth, elevation) direction.
    This version calculates delays on the fly rather than using a precomputed table.
    """
    delay_samples = calculate_delays_for_direction(
        mic_positions, azimuth, elevation, sample_rate, speed_of_sound
    )
    beamformed_signal = apply_beamforming(signal_data, delay_samples)
    return beamformed_signal

def calculate_delays_for_direction(mic_positions, azimuth, elevation,
                                   sample_rate, speed_of_sound):
    """
    Computes delay (in samples) for each microphone, given a direction vector.
    """
    azimuth_rad = np.radians(azimuth)
    elevation_rad = np.radians(elevation)

    direction_vector = np.array([
        np.cos(elevation_rad) * np.cos(azimuth_rad),
        np.cos(elevation_rad) * np.sin(azimuth_rad),
        np.sin(elevation_rad)
    ])

    delays = np.dot(mic_positions, direction_vector) / speed_of_sound
    delay_samples = np.round(delays * sample_rate).astype(np.int32)
    return delay_samples

def apply_beamforming(signal_data, delay_samples):
    """
    Applies simple delay-and-sum beamforming to multi-channel data.
    """
    num_samples, num_mics = signal_data.shape
    output_signal = np.zeros(num_samples, dtype=np.float64)

    for mic_idx in range(num_mics):
        delay = delay_samples[mic_idx]
        shifted_signal = shift_signal_beamforming(signal_data[:, mic_idx], delay)
        output_signal += shifted_signal

    output_signal /= num_mics
    return output_signal

def process_drone_data(drone_config, experiment_params):
    """
    Processes audio + drone data for one 'drone_config' object. This includes:
      - Reading/transforming flight CSV
      - Opening & aligning WAV files
      - Iterating through short audio chunks
      - Performing beamforming across (azimuth_range x elevation_range)
      - Labeling each angle based on its difference from the CSV angle
      - Saving an .npz per chunk with the beamformed waveforms and metadata
      - Generating an animation stored in 'output.mp4'
      - Returning a DataFrame of summarized results
    """
    # Unpack experiment parameters:
    sr = experiment_params["sample_rate"]
    chunk_s = experiment_params["chunk_duration_s"]
    chunk_n = int(sr * chunk_s)
    c = experiment_params["speed_of_sound"]
    record_s = experiment_params["RECORD_SECONDS"]
    lowcut = experiment_params["lowcut"]
    highcut = experiment_params["highcut"]
    azimuth_range = experiment_params["azimuth_range"]
    elevation_range = experiment_params["elevation_range"]
    angular_threshold = experiment_params["angular_threshold"]
    make_dataset = experiment_params["make_dataset"]

    # Create a 'dataset' folder if not exists
    os.makedirs("dataset", exist_ok=True)

    skip_s = drone_config["skip_seconds"]

    # Read CSVs
    ref_data = pd.read_csv(drone_config['ref_csv'], skiprows=0, delimiter=',', low_memory=False)
    flight_data = pd.read_csv(drone_config['flight_csv'], skiprows=0, delimiter=',', low_memory=False)

    # Generate reference lat/long from the "ref_csv"
    reference_latitude = ref_data[drone_config['latitude_col']].dropna().astype(float).mean()
    reference_longitude = ref_data[drone_config['longitude_col']].dropna().astype(float).mean()

    # Keep only desired columns from flight_data, starting at 'start_index'
    cols_to_keep = [
        drone_config['latitude_col'],
        drone_config['longitude_col'],
        drone_config['altitude_col'],
        drone_config['time_col']
    ] + drone_config['speed_cols_mph']

    flight_data = flight_data.iloc[drone_config['start_index']:].reset_index(drop=True)
    flight_data = flight_data[cols_to_keep].dropna()

    # Convert altitude from feet to meters
    flight_data[drone_config['altitude_col']] *= 0.3048
    ref_data[drone_config['altitude_col']] *= 0.3048

    # Store the initial altitude for reference
    initial_altitude = ref_data[drone_config['altitude_col']].iloc[0]

    # Convert lat/long to XY using a chosen EPSG (edit as needed)
    transformer = Transformer.from_crs('epsg:4326', 'epsg:32756', always_xy=True)
    ref_x, ref_y = transformer.transform(reference_longitude, reference_latitude)

    flight_data['X_meters'], flight_data['Y_meters'] = transformer.transform(
        flight_data[drone_config['longitude_col']].values,
        flight_data[drone_config['latitude_col']].values
    )

    # Compute offsets so the CSV-based angles line up with how we want them displayed
    drone_init_az = calculate_azimuth_meters(
        ref_x, ref_y,
        flight_data.iloc[0]['X_meters'],
        flight_data.iloc[0]['Y_meters']
    )
    drone_init_el = calculate_elevation_meters(
        flight_data.iloc[0][drone_config['altitude_col']],
        ref_x, ref_y,
        flight_data.iloc[0]['X_meters'],
        flight_data.iloc[0]['Y_meters'],
        initial_altitude
    )
    az_offset = drone_config['initial_azimuth'] - drone_init_az
    el_offset = drone_config['initial_elevation']  # user-defined offset

    # Precompute mic positions & table of sample delays for the entire angle grid
    mic_positions, table_delays, num_mics = initialize_beamforming_params(
        azimuth_range,
        elevation_range,
        c,
        sr,
        a_=[0, -120, -240],
        a2_=[-40, -80, -160, -200, -280, -320],
        h_=[1.12, 0.92, 0.77, 0.6, 0.42, 0.02],
        r_=[0.1, 0.17, 0.25, 0.32, 0.42, 0.63]
    )

    # Open WAV files
    wav_files = open_wav_files(drone_config['wav_filenames'])
    for i, wf in enumerate(wav_files):
        corr = drone_config["corrections"][i] if i < len(drone_config["corrections"]) else 0
        skip_wav_seconds(wf, skip_s, sr, extra_samples=corr)

    # Prepare an array for reading blocks from each wav_file
    buffers = [np.zeros((chunk_n, experiment_params["CHANNELS"]), dtype=np.int32)
               for _ in range(len(wav_files))]

    # Result columns to keep track of summary
    results_columns = [
        'Tiempo_Audio_s', 'Tiempo_CSV_ms',
        'Azimut_Estimado_deg', 'Elevacion_Estimada_deg',
        'Azimut_CSV_deg', 'Elevacion_CSV_deg',
        'Dif_Azimut_deg', 'Dif_Elevacion_deg',
        'Distancia_Metros',
        'xSpeed_mps', 'ySpeed_mps', 'zSpeed_mps'
    ]
    results_df = pd.DataFrame(columns=results_columns)

    # Plot setup (interactive)
    plt.ion()
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 8))
    fig.suptitle(f'{drone_config["name"]} - XY, Altitude, Beamforming + Labels')

    ax_xy = axes[0, 0]
    ax_alt = axes[0, 1]
    ax_beam = axes[0, 2]
    ax_labels = axes[1, 2]

    # XY subplot
    ax_xy.set_title('XY Position (m)')
    ax_xy.set_xlabel('X (m)')
    ax_xy.set_ylabel('Y (m)')
    ax_xy.grid(True)
    csv_x_positions = []
    csv_y_positions = []
    line_xy, = ax_xy.plot([], [], 'b-', label='Drone XY Path')
    marker_xy, = ax_xy.plot([], [], 'ro', label='Current Pos')
    ax_xy.legend()

    # Altitude subplot
    ax_alt.set_title('Altitude vs Time')
    ax_alt.set_xlabel('CSV Time (mm:ss)')
    ax_alt.set_ylabel('Altitude above initial (m)')
    ax_alt.grid(True)
    alt_times = []
    alt_values = []
    line_alt, = ax_alt.plot([], [], 'g-', label='Altitude Trace')
    marker_alt, = ax_alt.plot([], [], 'mo', label='Current Alt')
    ax_alt.legend()

    # Beamforming subplot
    cax = ax_beam.imshow(
        np.zeros((len(elevation_range), len(azimuth_range))),
        extent=[azimuth_range[0], azimuth_range[-1],
                elevation_range[0], elevation_range[-1]],
        origin='lower',
        aspect='auto',
        cmap='jet',
        interpolation='nearest'
    )
    fig.colorbar(cax, ax=ax_beam, label='Energy')
    line_csv, = ax_beam.plot([], [], 'k+', markersize=30, label='CSV Trajectory')
    marker_max, = ax_beam.plot([], [], 'ro', label='Max Energy')
    ax_beam.set_xlabel('Azimuth (deg)')
    ax_beam.set_ylabel('Elevation (deg)')
    ax_beam.set_title('Beamforming')
    ax_beam.legend()
    ax_beam.grid(True)

    # Labels subplot
    label_im = ax_labels.imshow(
        np.zeros((len(elevation_range), len(azimuth_range))),
        extent=[azimuth_range[0], azimuth_range[-1],
                elevation_range[0], elevation_range[-1]],
        origin='lower',
        aspect='auto',
        cmap='jet',
        interpolation='nearest'
    )
    ax_labels.set_title('Labels Grid')
    ax_labels.set_xlabel('Azimuth (deg)')
    ax_labels.set_ylabel('Elevation (deg)')

    plt.subplots_adjust(right=0.82, bottom=0.15)
    info_text = fig.text(
        0.84, 0.7, '',
        va='top', ha='left',
        fontsize=9,
        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.5)
    )
    footer_text = fig.text(0.5, 0.02, '', va='bottom', ha='center', fontsize=10)
    fig.canvas.draw()
    fig.canvas.flush_events()

    max_iterations = min(int(sr / chunk_n * record_s), len(flight_data))
    writer = FFMpegWriter(fps=10)

    # Record animation to MP4
    with writer.saving(fig, "output_oct11.mp4", dpi=100):
        first_block_shown = False

        for time_idx in range(max_iterations):
            # (A) Read audio block
            for j, wf in enumerate(wav_files):
                block = read_wav_block(wf, chunk_n, experiment_params["CHANNELS"])
                if block is None:
                    break
                buffers[j] = block
            if block is None:
                break

            # Combine signals from all wave files
            combined_signal = np.hstack(buffers)

            # ------------- OPTIONAL PLOT OF FIRST BLOCK -------------
            if not first_block_shown:
                first_block_shown = True
                channels_to_plot = list(range(5, combined_signal.shape[1], 6))

                fig_block, axes_block = plt.subplots(
                    nrows=len(channels_to_plot), ncols=1,
                    figsize=(10, 2 * len(channels_to_plot)),
                    sharex=True
                )
                if len(channels_to_plot) == 1:
                    axes_block = [axes_block]  # make it iterable

                fig_block.suptitle("First Combined Signal (Selected Channels)")

                for idx, ch in enumerate(channels_to_plot):
                    axes_block[idx].plot(combined_signal[:, ch])
                    axes_block[idx].set_title(f"Channel {ch}")
                    axes_block[idx].grid(True)

                plt.tight_layout()
                plt.show(block=False)
                input("Press Enter to continue (or Ctrl+C to abort)... ")
                plt.close(fig_block)
            # ---------------------------------------------------------

            # Apply bandpass
            filtered_signal = apply_bandpass_filter(combined_signal, lowcut, highcut, sr)

            # (B) Beamform over entire grid (2D)
            energy = beamform_time(filtered_signal, table_delays)
            max_energy_idx = np.unravel_index(np.argmax(energy), energy.shape)
            estimated_azimuth = azimuth_range[max_energy_idx[0]]
            estimated_elevation = elevation_range[max_energy_idx[1]]

            # (C) Audio time
            audio_time_s = calculate_time(time_idx, chunk_n, sr) + skip_s

            # (D) Drone CSV row
            row = flight_data.iloc[time_idx]
            x = row['X_meters']
            y = row['Y_meters']
            alt_absolute = row[drone_config['altitude_col']]

            # Speeds mph->m/s
            vx_mps = row[drone_config['speed_cols_mph'][0]] * 0.44704
            vy_mps = row[drone_config['speed_cols_mph'][1]] * 0.44704
            vz_mps = row[drone_config['speed_cols_mph'][2]] * 0.44704

            alt_relative = alt_absolute - initial_altitude
            current_time_csv_ms = row[drone_config['time_col']]
            csv_time_s = current_time_csv_ms / 1000.0

            # (E) Compute CSV-based az/el
            csv_azimuth = calculate_azimuth_meters(ref_x, ref_y, x, y) + az_offset
            csv_azimuth = wrap_angle(csv_azimuth)
            # If your coordinate system requires flipping sign, do so:
            csv_azimuth = -csv_azimuth
            csv_elevation = (
                calculate_elevation_meters(alt_absolute, ref_x, ref_y, x, y, initial_altitude)
                + el_offset
            )

            total_distance = calculate_total_distance_meters(
                ref_x, ref_y, x, y, initial_altitude, alt_absolute
            )
            az_diff, el_diff = calculate_angle_difference(
                estimated_azimuth, csv_azimuth,
                estimated_elevation, csv_elevation
            )

            # (F) Append summary info
            new_data = pd.DataFrame([{
                'Tiempo_Audio_s': audio_time_s,
                'Tiempo_CSV_ms': current_time_csv_ms,
                'Azimut_Estimado_deg': estimated_azimuth,
                'Elevacion_Estimada_deg': estimated_elevation,
                'Azimut_CSV_deg': csv_azimuth,
                'Elevacion_CSV_deg': csv_elevation,
                'Dif_Azimut_deg': az_diff,
                'Dif_Elevacion_deg': el_diff,
                'Distancia_Metros': total_distance,
                'xSpeed_mps': vx_mps,
                'ySpeed_mps': vy_mps,
                'zSpeed_mps': vz_mps
            }])
            results_df = pd.concat([results_df, new_data], ignore_index=True)

            # Console log
            audio_time_str = format_time_s(audio_time_s)
            csv_time_str = format_time_s(csv_time_s)
            print(
                f"Dist={total_distance:5.2f} m | "
                f"Audio={audio_time_str} | CSV={csv_time_str} | "
                f"SSL=(Az={estimated_azimuth:.1f}, El={estimated_elevation:.1f}) | "
                f"CSV=(Az={csv_azimuth:.1f}, El={csv_elevation:.1f}) | "
                f"Diff=(Az={az_diff:.1f}, El={el_diff:.1f}) | "
                f"Vx={vx_mps:.2f}, Vy={vy_mps:.2f}, Vz={vz_mps:.2f} m/s"
            )

            # (G) Update XY plot
            csv_x_positions.append(x-ref_x)
            csv_y_positions.append(y-ref_y)
            line_xy.set_data(csv_x_positions, csv_y_positions)
            marker_xy.set_data([x-ref_x], [y-ref_y])
            ax_xy.relim()
            ax_xy.autoscale_view()

            # (H) Update Alt plot
            alt_times.append(csv_time_s)
            alt_values.append(alt_relative)
            line_alt.set_data(alt_times, alt_values)
            marker_alt.set_data([csv_time_s], [alt_relative])
            ax_alt.relim()
            ax_alt.autoscale_view()

            # (I) Update beamforming image
            cax.set_data(energy.T)
            cax.set_clim(vmin=np.min(energy), vmax=np.max(energy))
            marker_max.set_data([estimated_azimuth], [estimated_elevation])
            line_csv.set_data([csv_azimuth], [csv_elevation])

            # (J) Info box
            info_text.set_text(
                f"Alt: {alt_relative:.2f} m\n"
                f"Dist: {total_distance:.2f} m\n"
                f"Vx: {vx_mps:.2f} m/s\n"
                f"Vy: {vy_mps:.2f} m/s\n"
                f"Vz: {vz_mps:.2f} m/s"
            )

            # (K) Footer
            footer_str = (
                f"Audio={audio_time_str} | CSV={csv_time_str} | "
                f"SSL=(Az={estimated_azimuth:.1f}, El={estimated_elevation:.1f}) | "
                f"CSV=(Az={csv_azimuth:.1f}, El={csv_elevation:.1f}) | "
                f"Diff=(Az={az_diff:.1f}, El={el_diff:.1f})"
            )
            footer_text.set_text(footer_str)

            # Force draw, capture MP4 frame
            fig.canvas.draw()
            fig.canvas.flush_events()
            writer.grab_frame()

            # -------------------------------------------------------
            # (L) If making a dataset, create an array of beamformed
            #     signals for each angle, then label them.
            #     Instead of storing FFT features, we store the
            #     time-domain waveform in beamformed_data[az, el, :].
            # -------------------------------------------------------
            if make_dataset:
                num_az = len(azimuth_range)
                num_el = len(elevation_range)
                beamformed_data = np.empty(
                    (num_az, num_el, chunk_n),
                    dtype=np.float32
                )
                labels_grid = np.empty((num_az, num_el), dtype=np.int32)

                for az_idx, grid_azimuth in enumerate(azimuth_range):
                    for el_idx, grid_elevation in enumerate(elevation_range):
                        # Beamform at this specific angle
                        bf_chunk = beamform_in_direction(
                            filtered_signal,
                            mic_positions,
                            grid_azimuth,
                            grid_elevation,
                            sr,
                            c
                        ).astype(np.float32)

                        beamformed_data[az_idx, el_idx, :] = bf_chunk

                        # Label = 1 if angular distance <= threshold
                        ang_dist = calculate_angular_distance(
                            csv_azimuth, csv_elevation,
                            grid_azimuth, grid_elevation
                        )
                        labels_grid[az_idx, el_idx] = 1 if ang_dist <= angular_threshold else 0

                # Update label image with the current chunk's labels (for visualization)
                label_im.set_data(labels_grid.T)
                label_im.set_clim(vmin=0, vmax=1)
                fig.canvas.draw()
                fig.canvas.flush_events()
                writer.grab_frame()

                # Include drone metadata in the saved file
                dataset_filename = os.path.join(drone_config["output_folder"], f"chunk_{time_idx:04d}.npz")
                np.savez(
                    dataset_filename,
                    beamformed_data=beamformed_data,
                    labels=labels_grid,
                    # Drone metadata:
                    csv_azimuth=csv_azimuth,
                    csv_elevation=csv_elevation,
                    csv_time_ms=current_time_csv_ms,
                    audio_time_s=audio_time_s,
                    total_distance=total_distance,
                    altitude_m=alt_absolute,
                    vx_mps=vx_mps,
                    vy_mps=vy_mps,
                    vz_mps=vz_mps,
                    pos_x=x-ref_x,
                    pos_y=y-ref_y,
                    # Possibly store raw angles for reference:
                    estimated_azimuth=estimated_azimuth,
                    estimated_elevation=estimated_elevation,
                )

        print(f"{drone_config['name']} processing completed.")

    plt.ioff()
    plt.show()

    # close wav files
    for wf in wav_files:
        wf.close()

    return results_df

def main():
    """
    Main driver function. Defines both the experiment parameters and the
    drone configuration(s), calls 'process_drone_data()', and saves final CSVs.
    """
    # -- 1) Gather all experiment-specific parameters --
    experiment_params = {
        "CHANNELS": 6,
        "sample_rate": 48000,
        "chunk_duration_s": 0.1,
        "speed_of_sound": 343,
        "RECORD_SECONDS": 1200000,
        "lowcut": 200.0,
        "highcut": 8000.0,
        "azimuth_range": np.arange(-180, 181, 4),
        "elevation_range": np.arange(0, 91, 4),
        # Beamforming label threshold
        "angular_threshold": 10,
        # Whether or not to generate dataset files (.npz)
        "make_dataset": True,
    }

    # -- 2) Drone configuration for this experiment --
    drones_config = [
        {
            "name": "DJI Air 3",
            "ref_csv": "dataset/original/Oct_11_2024/Ref/DJIFlightRecord_2024-10-11_[14-32-34].csv",
            "flight_csv": "dataset/original/Oct_11_2024/Oct-11th-2024-03-49PM-Flight-Airdata.csv",
            "latitude_col": "latitude",
            "longitude_col": "longitude",
            "altitude_col": "altitude_above_seaLevel(feet)",
            "time_col": "time(millisecond)",
            "initial_azimuth": -5.0,
            "initial_elevation": 0.0,
            "start_index": 39,
            "speed_cols_mph": [" xSpeed(mph)", " ySpeed(mph)", " zSpeed(mph)"],
            "corrections": [0, -866, -626, -729],
            # "skip_seconds": 3.6,  # sync pulse
            "skip_seconds": 82.0,   # seconds to skip from start of WAV
            "wav_filenames": [
                "dataset/original/Oct_11_2024/5/device_1_nosync.wav",
                "dataset/original/Oct_11_2024/5/device_2_nosync.wav",
                "dataset/original/Oct_11_2024/5/device_3_nosync.wav",
                "dataset/original/Oct_11_2024/5/device_4_nosync.wav"
            ],
            "output_folder": "/run/media/sergio/Extreme SSD/Datasets/processed_dataset_drones/Oct_11_2024/"
        }
    ]

    # -- 3) Process each drone configuration --
    all_results = {}
    for cfg in drones_config:
        df_results = process_drone_data(cfg, experiment_params)
        all_results[cfg["name"]] = df_results

        out_file = f"{cfg['name'].replace(' ', '_')}_results.csv"
        df_results.to_csv(out_file, index=False)
        print(f"Results saved to {out_file}")

# Standard entry point
if __name__ == "__main__":
    main()
