import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


# Define SNR computation
def compute_snr(distance, spl_source=70, spl_noise=55):
    return spl_source - 20 * np.log10(distance) - spl_noise


# File paths and friendly names
files = {
    "Training-day (test_2.csv)": "test_2.csv",
    "Different-location (test_Oct.csv)": "test_Oct.csv",
    "No-drone (test_No_drone.csv)": "test_No_drone.csv"
}

# Prepare a summary DataFrame
summary_list = []

for name, path in files.items():
    df = pd.read_csv(path)
    # Check availability of columns
    has_distance = 'total_distance' in df.columns
    has_errors = all(c in df.columns for c in ['az_diff', 'el_diff', 'az_diff_unet', 'el_diff_unet'])

    # Compute SNR if possible
    if has_distance:
        df['snr'] = compute_snr(df['total_distance'])
    else:
        df['snr'] = np.nan

    # Build stats dict
    stats = {'Dataset': name, 'Samples': len(df)}
    # Errors
    if has_errors:
        stats.update({
            'Mean az_diff': df['az_diff'].mean(),
            'STD az_diff': df['az_diff'].std(),
            'Mean el_diff': df['el_diff'].mean(),
            'STD el_diff': df['el_diff'].std(),
            'Mean az_diff_unet': df['az_diff_unet'].mean(),
            'STD az_diff_unet': df['az_diff_unet'].std(),
            'Mean el_diff_unet': df['el_diff_unet'].mean(),
            'STD el_diff_unet': df['el_diff_unet'].std(),
        })
    else:
        stats.update({k: np.nan for k in [
            'Mean az_diff', 'STD az_diff', 'Mean el_diff', 'STD el_diff',
            'Mean az_diff_unet', 'STD az_diff_unet', 'Mean el_diff_unet', 'STD el_diff_unet'
        ]})
    # unet_loss
    stats['Mean unet_loss'] = df['unet_loss'].mean()
    stats['STD unet_loss'] = df['unet_loss'].std()
    # SNR
    if has_distance:
        stats['Mean SNR'] = df['snr'].mean()
        stats['STD SNR'] = df['snr'].std()
    else:
        stats['Mean SNR'] = np.nan
        stats['STD SNR'] = np.nan
    # valid predictions
    vp_col = 'valid_prediction' if 'valid_prediction' in df.columns else 'valid_predict'
    stats['Valid pred. rate'] = df[vp_col].mean()

    summary_list.append(stats)

    # Plots
    # 1) Histogram of unet_loss
    plt.figure()
    plt.hist(df['unet_loss'], bins=30)
    plt.title(f"Unet Loss Distribution ({name})")
    plt.xlabel("Tversky Loss")
    plt.ylabel("Count")
    plt.show()

    # 2) Histogram of valid predictions
    plt.figure()
    plt.hist(df[vp_col], bins=[-0.1, 0.1, 0.9, 1.1])
    plt.xticks([0, 1])
    plt.title(f"Valid Prediction Rate ({name})")
    plt.xlabel("Valid Prediction (<0=no drone>)")
    plt.ylabel("Count")
    plt.show()

    # 3) Histogram of SNR if available
    if has_distance:
        plt.figure()
        plt.hist(df['snr'], bins=30)
        plt.title(f"SNR Distribution ({name})")
        plt.xlabel("SNR (dB)")
        plt.ylabel("Count")
        plt.show()

    # 4) Scatter plots if error and distance available
    if has_distance and has_errors:
        plt.figure()
        plt.scatter(df['total_distance'], df['az_diff_unet'], s=5)
        plt.title(f"Azimuth Error vs Distance ({name})")
        plt.xlabel("Distance (m)")
        plt.ylabel("Unet Azimuth Error (°)")
        plt.show()

        plt.figure()
        plt.scatter(df['total_distance'], df['el_diff_unet'], s=5)
        plt.title(f"Elevation Error vs Distance ({name})")
        plt.xlabel("Distance (m)")
        plt.ylabel("Unet Elevation Error (°)")
        plt.show()

        plt.figure()
        plt.scatter(df['snr'], df['az_diff_unet'], s=5)
        plt.title(f"Azimuth Error vs SNR ({name})")
        plt.xlabel("SNR (dB)")
        plt.ylabel("Unet Azimuth Error (°)")
        plt.show()

        plt.figure()
        plt.scatter(df['snr'], df['el_diff_unet'], s=5)
        plt.title(f"Elevation Error vs SNR ({name})")
        plt.xlabel("SNR (dB)")
        plt.ylabel("Unet Elevation Error (°)")
        plt.show()

# Display summary table
summary_df = pd.DataFrame(summary_list)

