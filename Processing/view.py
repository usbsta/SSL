#!/usr/bin/env python3
"""
animate_grid_labels.py

This script loads a grid-label dataset from an NPZ file and animates the labels.
The labels are assumed to be stored as a 3D array with shape:
    (num_frames, num_azimuth_points, num_elevation_points)

Author: [Your Name]
Date: [Current Date]
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Animate grid labels stored in an NPZ file with grid structure (num_frames x num_azimuth_points x num_elevation_points)."
    )
    parser.add_argument('--npz_file', type=str, default="11_10deg.npz",
                        help='Path to the NPZ file containing the grid labels.')
    parser.add_argument('--interval', type=int, default=5,
                        help='Interval between frames in milliseconds.')
    args = parser.parse_args()

    # Load the NPZ file; assume it contains key 'y' with the grid labels
    data = np.load(args.npz_file)
    y = data['y']  # Expected shape: (num_frames, num_azimuth_points, num_elevation_points)
    print("Loaded label grid shape:", y.shape)

    num_frames, num_azimuth_points, num_elevation_points = y.shape

    # Set up the figure for animation.
    fig, ax = plt.subplots()
    # Display the first frame; we use a transpose if necessary to orient the grid as desired.
    im = ax.imshow(y[0].T, cmap='viridis', vmin=0, vmax=1, origin='lower')
    ax.set_title("Grid Labels Animation")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Label value")

    def update(frame):
        # Update the image with the grid of labels for the current frame.
        im.set_data(y[frame].T)
        ax.set_title(f"Grid Labels Animation: Frame {frame}")
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=args.interval, blit=True)
    plt.show()

if __name__ == '__main__':
    main()


"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def main():
    # Load the dataset from NPZ file
    data = np.load('grid_dataset.npz')
    X = data['X']  # FFT features: shape (num_samples, num_features)
    y = data['y']  # Labels (if needed for display)

    num_samples, num_features = X.shape

    # Define a frequency axis if desired (adjust freq_range as used when generating FFT features)
    freq_range = (300, 2300)  # Example: frequency range from 300 Hz to 2300 Hz
    freqs = np.linspace(freq_range[0], freq_range[1], num_features)

    # Create figure and axis for the animation
    fig, ax = plt.subplots()
    # Initial line plot using the first frame
    line, = ax.plot(freqs, X[0], lw=2)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude')
    ax.set_title('FFT Features Animation')
    # Set y-limits based on the overall range in X
    ax.set_ylim(np.min(X), np.max(X))

    # Update function for the animation; frame is the current index
    def update(frame):
        # Update the line plot data
        line.set_ydata(X[frame])
        # Optionally, update the title to show frame number and label
        ax.set_title(f'Frame {frame} - Label: {y[frame]}')
        return line,

    # Create the animation: interval is in milliseconds
    ani = animation.FuncAnimation(fig, update, frames=num_samples, interval=100, blit=True)

    # Display the animation
    plt.show()

if __name__ == '__main__':
    main()

"""