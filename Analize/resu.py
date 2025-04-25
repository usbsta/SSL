import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# File paths and labels
files = {
    "Training-day": "test_2.csv",
    "Different-location": "test_Oct.csv",
    "No-drone": "test_No_drone.csv"
}

# Container for summary
summary = []

for name, path in files.items():
    df = pd.read_csv(path)
    # Only if both diff columns exist
    if 'az_diff' in df.columns and 'el_diff' in df.columns:
        # Compute combined angular error: RMS of beamforming diffs
        df['err_beam'] = np.sqrt(df['az_diff']**2 + df['el_diff']**2)
        # Combined error of UNet centroids
        df['err_unet'] = np.sqrt(df['az_diff_unet']**2 + df['el_diff_unet']**2)
        # Summary stats
        summary.append({
            'Dataset': name,
            'Mean Err Beam (°)': df['err_beam'].mean(),
            'Median Err Beam (°)': df['err_beam'].median(),
            'Mean Err UNet (°)': df['err_unet'].mean(),
            'Median Err UNet (°)': df['err_unet'].median()
        })
        # Histogram of errors
        plt.figure()
        plt.hist(df['err_beam'], bins=30, alpha=0.6, label='Beamforming')
        plt.hist(df['err_unet'], bins=30, alpha=0.6, label='UNet')
        plt.title(f"Combined Angular Error Distribution ({name})")
        plt.xlabel("Error (°)")
        plt.ylabel("Count")
        plt.legend()
        plt.show()
        # Scatter error vs distance if available
        if 'total_distance' in df.columns:
            plt.figure()
            plt.scatter(df['total_distance'], df['err_unet'], s=5)
            plt.title(f"UNet Combined Error vs Distance ({name})")
            plt.xlabel("Distance (m)")
            plt.ylabel("Error (°)")
            plt.show()
            # Compute SNR
            df['snr'] = 70 - 20 * np.log10(df['total_distance']) - 55
            plt.figure()
            plt.scatter(df['snr'], df['err_unet'], s=5)
            plt.title(f"UNet Combined Error vs SNR ({name})")
            plt.xlabel("SNR (dB)")
            plt.ylabel("Error (°)")
            plt.show()
    else:
        print(f"Skipping combined error for {name}: no ground truth diffs.")

# Display summary table
if summary:
    summary_df = pd.DataFrame(summary)
    display_dataframe_to_user("Combined Angular Error Summary", summary_df)
else:
    print("No datasets with both az_diff and el_diff found for combined error.")
