#!/usr/bin/env python3
"""
inference_from_dataset.py

This script performs inference on a pre-generated dataset for segmentation of beamformed audio.
It loads a dataset (NPZ file) created using generate_grid_dataset.py, runs a trained UNet model to
obtain segmentation predictions, compares predictions to ground truth labels using the F1 score,
and animates the results.
The animation shows:
    - Predictions of 1 in red,
    - Predictions of 0 in green,
    - Ground truth positive cells (label==1) overlaid with an orange border.
At the end, the overall F1 score is printed and a CSV file with per-chunk results is saved.

Author: [Your Name]
Date: [Current Date]
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
import pandas as pd
import argparse
from pyproj import Transformer

from functions import calculate_total_distance_meters  # Computes the total Euclidean distance between flight positions.
from experiments_config import get_experiment_config

# ---------------------------
# Model Architecture: UNetSmall
# ---------------------------
class UNetSmall(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        A modified U-Net architecture for small input grids.

        Parameters:
            in_channels (int): Number of FFT bins (input channels).
            out_channels (int): Number of output channels (typically 1 for binary segmentation).
        """
        super(UNetSmall, self).__init__()
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.down1 = self.conv_block(in_channels, 64)
        self.down2 = self.conv_block(64, 128)
        self.down3 = self.conv_block(128, 256)
        self.bottleneck = self.conv_block(256, 512)
        self.up3 = torch.nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv3 = self.conv_block(512, 256)
        self.up2 = torch.nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = self.conv_block(256, 128)
        self.up1 = torch.nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv1 = self.conv_block(128, 64)
        self.final_conv = torch.nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        """Two consecutive Conv2D layers with BatchNorm and ReLU activation."""
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        c1 = self.down1(x)
        p1 = self.pool(c1)
        c2 = self.down2(p1)
        p2 = self.pool(c2)
        c3 = self.down3(p2)
        p3 = self.pool(c3)
        bn = self.bottleneck(p3)
        u3 = self.up3(bn)
        if u3.shape[2:] != c3.shape[2:]:
            u3 = torch.nn.functional.interpolate(u3, size=c3.shape[2:])
        u3 = torch.cat([u3, c3], dim=1)
        c4 = self.conv3(u3)
        u2 = self.up2(c4)
        if u2.shape[2:] != c2.shape[2:]:
            u2 = torch.nn.functional.interpolate(u2, size=c2.shape[2:])
        u2 = torch.cat([u2, c2], dim=1)
        c5 = self.conv2(u2)
        u1 = self.up1(c5)
        if u1.shape[2:] != c1.shape[2:]:
            u1 = torch.nn.functional.interpolate(u1, size=c1.shape[2:])
        u1 = torch.cat([u1, c1], dim=1)
        c6 = self.conv1(u1)
        output = self.final_conv(c6)
        return output

# ---------------------------
# Main Inference and Visualization Function
# ---------------------------
def main():
    flight_experiment = 0
    config = get_experiment_config(flight_experiment)  # Retrieve parameters for experiment 11
    initial_azimuth = config["initial_azimuth"]
    initial_elevation = config["initial_elevation"]
    start_index = config["start_index"]
    ref_file_path = config["ref_file_path"]
    file_path_flight = config["file_path_flight"]

    ### CSV calculations

    ref_data = pd.read_csv(ref_file_path, delimiter=',', low_memory=False)
    flight_data = pd.read_csv(file_path_flight, delimiter=',', low_memory=False)

    reference_latitude = ref_data['latitude'].dropna().astype(float).mean()
    reference_longitude = ref_data['longitude'].dropna().astype(float).mean()

    flight_data = flight_data.iloc[start_index:].reset_index(drop=True)
    flight_data = flight_data[['latitude', 'longitude', 'altitude_above_seaLevel(feet)', 'time(millisecond)']].dropna()
    ref_data = ref_data[['latitude', 'longitude', 'altitude_above_seaLevel(feet)', 'time(millisecond)']].dropna()
    flight_data['altitude_above_seaLevel(feet)'] = flight_data['altitude_above_seaLevel(feet)'] * 0.3048
    ref_data['altitude_above_seaLevel(feet)'] = ref_data['altitude_above_seaLevel(feet)'] * 0.3048
    initial_altitude = ref_data['altitude_above_seaLevel(feet)'].iloc[0]

    transformer = Transformer.from_crs('epsg:4326', 'epsg:32756', always_xy=True)
    ref_x, ref_y = transformer.transform(reference_longitude, reference_latitude)
    flight_data['X_meters'], flight_data['Y_meters'] = transformer.transform(
        flight_data['longitude'].values,
        flight_data['latitude'].values
    )

    #####

    parser = argparse.ArgumentParser(
        description="Perform inference on a pre-generated dataset and animate segmentation results."
    )
    parser.add_argument('--dataset', type=str, default=f'Datasets/5_Oct_10deg.npz',
                        help="Path to the NPZ dataset file containing features (X) and labels (y).")
    #parser.add_argument('--model_path', type=str, default='model_fft_Air_11_loss.pth',
    parser.add_argument('--model_path', type=str, default='model_fft_Air_11_own_loss.pth',
                        help="Path to the trained UNet model (.pth file).")
    parser.add_argument('--output_csv', type=str, default='inference_results.csv',
                        help="Path to save the CSV file with per-chunk inference results.")
    parser.add_argument('--threshold', type=float, default=0.5,
                        help="Threshold for converting model output probabilities to binary predictions.")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help="Computation device: 'cuda' or 'cpu'.")
    args = parser.parse_args()

    # Load the dataset from the NPZ file.
    data = np.load(args.dataset)
    X = data['X']  # Shape: (num_chunks, num_azimuth_points, num_elevation_points, num_bins)
    y_true = data['y']  # Shape: (num_chunks, num_azimuth_points, num_elevation_points)
    num_chunks = X.shape[0]
    num_azimuth_points = X.shape[1]
    num_elevation_points = X.shape[2]
    num_bins = X.shape[3]

    print(
        f"Loaded dataset with {num_chunks} chunks, grid shape: ({num_azimuth_points}, {num_elevation_points}), and {num_bins} FFT bins."
    )

    # Load the trained UNet model and set it to evaluation mode.
    device = torch.device(args.device)
    model = UNetSmall(in_channels=num_bins, out_channels=1).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print("Trained model loaded for inference.")

    # Lists to store per-chunk results for animation and CSV output.
    prediction_grids = []      # List of binary prediction grids.
    ground_truth_grids = []    # List of ground truth grids.
    chunk_f1_scores = []       # List of per-chunk F1 scores.
    total_distances = []       # List of total distances for each chunk.

    # Variables to accumulate global true positives, false positives, and false negatives.
    global_tp = 0
    global_fp = 0
    global_fn = 0

    # Inference loop: process each chunk independently.
    for chunk_idx in range(num_chunks):
        # Prepare the feature grid.
        features = X[chunk_idx].transpose(2, 0, 1)  # (num_bins, num_azimuth_points, num_elevation_points)
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)

        # Calculate the total distance for the current chunk.
        x = flight_data.iloc[chunk_idx]['X_meters']
        y = flight_data.iloc[chunk_idx]['Y_meters']
        altitude = flight_data.iloc[chunk_idx]['altitude_above_seaLevel(feet)']
        total_distance = calculate_total_distance_meters(ref_x, ref_y, x, y, initial_altitude, altitude)
        total_distances.append(total_distance)  # Store the computed distance

        # Run inference through the model.
        with torch.no_grad():
            output = model(features_tensor)
            probabilities = torch.sigmoid(output)
            predicted = (probabilities > args.threshold).float()

        predicted_grid = predicted.squeeze().cpu().numpy()
        predicted_grid = (predicted_grid > 0.5).astype(np.int32)
        true_grid = y_true[chunk_idx].astype(np.int32)

        # Compute confusion matrix values.
        tp = np.sum((predicted_grid == 1) & (true_grid == 1))
        fp = np.sum((predicted_grid == 1) & (true_grid == 0))
        fn = np.sum((predicted_grid == 0) & (true_grid == 1))

        # Compute F1 score for the chunk.
        if (2 * tp + fp + fn) > 0:
            chunk_f1 = 2 * tp / (2 * tp + fp + fn)
        else:
            chunk_f1 = 0.0

        chunk_f1_scores.append(chunk_f1)

        # Update global counts for overall F1 score computation.
        global_tp += tp
        global_fp += fp
        global_fn += fn

        prediction_grids.append(predicted_grid)
        ground_truth_grids.append(true_grid)

        print(f"Processed chunk {chunk_idx + 1}/{num_chunks}: F1 Score = {chunk_f1:.2%}")

    # Compute overall F1 score from aggregated counts.
    if (2 * global_tp + global_fp + global_fn) > 0:
        overall_f1 = 2 * global_tp / (2 * global_tp + global_fp + global_fn)
    else:
        overall_f1 = 0.0
    print(f"\nOverall F1 Score: {overall_f1:.2%}")

    # Save per-chunk results to a CSV file.
    results_df = pd.DataFrame({
        'chunk_index': np.arange(num_chunks),
        'num_cells': [num_azimuth_points * num_elevation_points] * num_chunks,
        'chunk_f1': chunk_f1_scores
    })
    results_df.to_csv(args.output_csv, index=False)
    print(f"Per-chunk inference results saved to '{args.output_csv}'.")

    # ---------------------------
    # Animation of Inference Results
    # ---------------------------
    # Create a custom colormap: 0 -> green, 1 -> red.
    cmap = ListedColormap(['green', 'red'])
    fig, ax = plt.subplots()

    # Initialize the image and the title text object.
    initial_prediction = prediction_grids[0]
    im = ax.imshow(initial_prediction.T, cmap=cmap, vmin=0, vmax=1, origin='lower')
    # Save the title text object in a variable.
    title_text = ax.set_title(f"Chunk 0 - Distance: {total_distances[0]:.2f} meters")
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1])
    cbar.ax.set_yticklabels(['0 (Green)', '1 (Red)'])

    def update(frame):
        # Remove previous orange patches.
        for patch in ax.patches[:]:
            patch.remove()

        # Update the image data with the transposed prediction grid.
        im.set_data(prediction_grids[frame].T)

        # Update the title text with the corresponding total distance.
        title_text.set_text(f"Chunk {frame} - Distance: {total_distances[frame]:.2f} meters")

        # Overlay orange rectangles for ground truth positives.
        gt_transposed = ground_truth_grids[frame].T
        for i in range(num_elevation_points):
            for j in range(num_azimuth_points):
                if gt_transposed[i, j] == 1:
                    rect = Rectangle((j - 0.5, i - 0.5), 1, 1,
                                     linewidth=1.5, edgecolor='orange', facecolor='none')
                    ax.add_patch(rect)

        # Return the updated artists (image, title text, and all current patches).
        return [im, title_text] + ax.patches

    # Create the animation with blitting enabled.
    ani = animation.FuncAnimation(fig, update, frames=num_chunks, interval=200, blit=False)
    plt.show()

if __name__ == '__main__':
    main()

