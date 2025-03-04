import numpy as np
import argparse

from extract_fft_features import process_audio_data

# Import parameters of the experiments and files
from Record.experiments_config import get_experiment_config

# Import helper functions from your beamforming module
from Record.functions import (
    initialize_microphone_positions,  # Returns an array with microphone positions in 3D space.
    load_flight_data,  # Loads flight data from a CSV file (returns a DataFrame and reference parameters).
    calculate_initial_offsets,  # Computes initial azimuth and elevation offsets based on flight data.
    # Computes azimuth angle from 2D coordinates.
    # Computes elevation angle from altitude differences.
    # Computes the total Euclidean distance between flight positions.
    # Wraps an angle to a standard range (e.g., -180° to 180°).
    read_and_synchronize_audio,
    update_flight_data,
    beamform_in_direction,
)

def calculate_angular_distance(az1, el1, az2, el2):
    """
    Computes the angular distance (in degrees) between two directions defined by azimuth and elevation.

    Parameters:
        az1, el1 (float): Azimuth and elevation (degrees) of the first direction.
        az2, el2 (float): Azimuth and elevation (degrees) of the second direction.

    Returns:
        angular_distance_deg (float): Angular distance in degrees.
    """
    # Convert angles from degrees to radians
    az1_rad, el1_rad = np.radians(az1), np.radians(el1)
    az2_rad, el2_rad = np.radians(az2), np.radians(el2)

    # Convert spherical coordinates to Cartesian coordinates
    x1 = np.cos(el1_rad) * np.cos(az1_rad)
    y1 = np.cos(el1_rad) * np.sin(az1_rad)
    z1 = np.sin(el1_rad)

    x2 = np.cos(el2_rad) * np.cos(az2_rad)
    y2 = np.cos(el2_rad) * np.sin(az2_rad)
    z2 = np.sin(el2_rad)

    # Compute the dot product and use arccos to get the angle between the vectors
    dot_product = x1 * x2 + y1 * y2 + z1 * z2
    dot_product = np.clip(dot_product, -1.0, 1.0)
    angular_distance_rad = np.arccos(dot_product)
    angular_distance_deg = np.degrees(angular_distance_rad)
    return angular_distance_deg


def save_dataset(features, labels, output_file):
    """
    Saves the dataset (features and labels) to an NPZ file.

    Parameters:
        features (np.ndarray): Array containing the beamformed audio samples.
        labels (np.ndarray): Array containing binary labels.
        output_file (str): Output filename for the dataset.
    """
    np.savez(output_file, X=features, y=labels)
    print(f"Dataset saved to '{output_file}'.")



def main():
    config = get_experiment_config(0)  # Retrieve parameters for experiment 11
    corrections = config["corrections"]
    wav_filenames = config["wav_filenames"]
    skip_seconds = config["skip_seconds"]
    initial_azimuth = config["initial_azimuth"]
    initial_elevation = config["initial_elevation"]
    start_index = config["start_index"]
    ref_file_path = config["ref_file_path"]
    file_path_flight = config["file_path_flight"]

    parser = argparse.ArgumentParser(
        description="Generate a grid-based dataset for beamformed audio with binary labels based on proximity to CSV direction."
    )

    # Audio configuration
    parser.add_argument('--num_bins', type=int, default=128, help='Number of frequency bins (default: 128)')
    parser.add_argument('--freq_range', type=float, nargs=2, default=[200, 2300],
                        help='Frequency range in Hz (default: 200 2300)')

    # Grid parameters
    parser.add_argument('--grid_azimuth_min', type=float, default=-180.0,
                        help="Minimum azimuth value for grid (degrees).")
    parser.add_argument('--grid_azimuth_max', type=float, default=170.0,
                        help="Maximum azimuth value for grid (degrees).")
    parser.add_argument('--num_azimuth_points', type=int, default=36, help="Number of points in the azimuth grid.")
    parser.add_argument('--grid_elevation_min', type=float, default=0.0,
                        help="Minimum elevation value for grid (degrees).")
    parser.add_argument('--grid_elevation_max', type=float, default=90.0,
                        help="Maximum elevation value for grid (degrees).")
    parser.add_argument('--num_elevation_points', type=int, default=10, help="Number of points in the elevation grid.")

    # Labeling threshold (angular distance)
    parser.add_argument('--angular_threshold', type=float, default=10.0,
                        help="Angular threshold (degrees) for labeling as near (1).")

    # Output dataset file
    parser.add_argument('--output_file', type=str, default='11_10deg_clean-17.npz', help="Output NPZ file for the dataset.")

    args = parser.parse_args()

    # Calculate number of samples per chunk based on the duration and sample rate
    CHUNK_DURATION = 0.1
    SAMPLE_RATE = 48000
    CHUNK = int(CHUNK_DURATION * SAMPLE_RATE)
    SPEED_OF_SOUND = 343
    CHANNELS = 6

    # Initialize microphone positions using the beamforming configuration
    mic_positions = initialize_microphone_positions()

    # Read and synchronize WAV files
    combined_signal = read_and_synchronize_audio(
        wav_filenames, CHANNELS, corrections, skip_seconds, SAMPLE_RATE)

    # Load flight data from CSV using the provided start index.
    # This function is assumed to return the flight data DataFrame, reference x, reference y, and initial altitude.
    flight_data, ref_x, ref_y, initial_altitude = load_flight_data(start_index, ref_file_path, file_path_flight)

    # Calculate the initial offsets based on flight data and provided initial offsets
    azimuth_offset, elevation_offset = calculate_initial_offsets(
        flight_data, ref_x, ref_y, initial_altitude, initial_azimuth, initial_elevation)

    # Define the grid of (azimuth, elevation) points
    azimuth_grid = np.linspace(args.grid_azimuth_min, args.grid_azimuth_max, args.num_azimuth_points)
    elevation_grid = np.linspace(args.grid_elevation_min, args.grid_elevation_max, args.num_elevation_points)

    # Initialize lists to store the grid-structured features and labels for each chunk.
    dataset_features = []  # Each element will have shape (num_azimuth_points, num_elevation_points, num_bins)
    dataset_labels = []    # Each element will have shape (num_azimuth_points, num_elevation_points)

    # Determine the number of chunks in the combined signal.
    num_chunks = int(len(combined_signal) / CHUNK)
    print(f"Total number of chunks to process: {num_chunks}")

    # Main processing loop: iterate over each chunk.
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * CHUNK
        end_idx = start_idx + CHUNK
        signal_chunk = combined_signal[start_idx:end_idx]

        # Use the flight data index corresponding to the current chunk.
        if chunk_idx >= len(flight_data):
            break  # End processing if flight data is exhausted.
        csv_azimuth, csv_elevation, _ = update_flight_data(
            flight_data, chunk_idx, ref_x, ref_y, initial_altitude, azimuth_offset, initial_elevation
        )

        # Initialize arrays for the current chunk:
        features_grid = np.empty((args.num_azimuth_points, args.num_elevation_points, args.num_bins))
        labels_grid = np.empty((args.num_azimuth_points, args.num_elevation_points), dtype=np.int32)

        for az_idx, grid_azimuth in enumerate(azimuth_grid):
            for el_idx, grid_elevation in enumerate(elevation_grid):
                grid_elevation = np.clip(grid_elevation, 0, 90)

                # Compute the beamformed signal for the current grid direction.
                beamformed_chunk = beamform_in_direction(
                    signal_chunk, mic_positions, grid_azimuth, grid_elevation, SAMPLE_RATE, SPEED_OF_SOUND
                )

                # Process the beamformed signal to obtain FFT features.
                # 'process_audio_data' returns a list; as the chunk is equal to window we only need the first element
                fft_features = process_audio_data(
                    beamformed_chunk,
                    SAMPLE_RATE,
                    num_bins=args.num_bins,
                    freq_range=tuple(args.freq_range),
                    window_duration=CHUNK_DURATION,
                    debug=False
                )
                features_grid[az_idx, el_idx, :] = fft_features[0]

                # Compute angular distance between the CSV direction and the current grid point.
                angular_distance = calculate_angular_distance(csv_azimuth, csv_elevation, grid_azimuth, grid_elevation)
                labels_grid[az_idx, el_idx] = 1 if angular_distance <= args.angular_threshold else 0

        # --- local normalization for each chunk ---
        chunk_min = features_grid.min()
        chunk_max = features_grid.max()
        features_grid_norm = (features_grid - chunk_min) / (chunk_max - chunk_min + 1e-6)
        # ----------------------------------------------

        dataset_features.append(features_grid_norm)
        dataset_labels.append(labels_grid)

        # Optionally, print status every 100 chunks.
        if chunk_idx % 100 == 0:
            print(
                f"Processed chunk {chunk_idx}: CSV direction (azimuth={csv_azimuth:.2f}°, elevation={csv_elevation:.2f}°)")

    # Convert lists to numpy arrays.
    # X shape (num_chunks, num_azimuth_points, num_elevation_points, num_bins)
    # y shape (num_chunks, num_azimuth_points, num_elevation_points)
    X_norm = np.array(dataset_features)
    y = np.array(dataset_labels)

    # Save the dataset
    save_dataset(X_norm, y, args.output_file)

    # Print dataset information
    print("Dataset Information:")
    print(f"  Total samples (chunks): {X_norm.shape[0]}")
    print(f"  Grid shape per sample (azimuth x elevation x num_bins): {X_norm.shape[1:]}")

if __name__ == '__main__':
    main()