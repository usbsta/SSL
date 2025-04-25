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
    if 'az_diff' in df.columns and 'el_diff' in df.columns:
        # Compute combined angular errors
        df['err_beam'] = np.sqrt(df['az_diff'] ** 2 + df['el_diff'] ** 2)
        df['err_unet'] = np.sqrt(df['az_diff_unet'] ** 2 + df['el_diff_unet'] ** 2)

        # Percentiles
        perc_beam = np.percentile(df['err_beam'], [25, 50, 75, 90])
        perc_unet = np.percentile(df['err_unet'], [25, 50, 75, 90])

        # Summary stats
        summary.append({
            'Dataset': name,
            'Mean Err Beam (°)': df['err_beam'].mean(),
            'Median Err Beam (°)': perc_beam[1],
            '90th Perc. Err Beam (°)': perc_beam[3],
            'Mean Err UNet (°)': df['err_unet'].mean(),
            'Median Err UNet (°)': perc_unet[1],
            '90th Perc. Err UNet (°)': perc_unet[3]
        })

        # Boxplot
        plt.figure(figsize=(6, 4))
        plt.boxplot([df['err_beam'], df['err_unet']], labels=['Beam', 'UNet'])
        plt.title(f"Combined Error Boxplot ({name})")
        plt.ylabel("Error (°)")
        plt.show()

        # Error vs Distance & SNR scatter for both
        if 'total_distance' in df.columns:
            df['snr'] = 70 - 20 * np.log10(df['total_distance']) - 55
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            ax[0].scatter(df['total_distance'], df['err_beam'], s=5, alpha=0.5, label='Beam')
            ax[0].scatter(df['total_distance'], df['err_unet'], s=5, alpha=0.5, label='UNet')
            ax[0].set_title("Error vs Distance")
            ax[0].set_xlabel("Distance (m)")
            ax[0].set_ylabel("Error (°)")
            ax[0].legend()

            ax[1].scatter(df['snr'], df['err_beam'], s=5, alpha=0.5, label='Beam')
            ax[1].scatter(df['snr'], df['err_unet'], s=5, alpha=0.5, label='UNet')
            ax[1].set_title("Error vs SNR")
            ax[1].set_xlabel("SNR (dB)")
            ax[1].set_ylabel("Error (°)")
            ax[1].legend()
            plt.show()

        # Improvement histogram
        df['improvement'] = df['err_beam'] - df['err_unet']
        plt.figure(figsize=(6, 4))
        plt.hist(df['improvement'], bins=30, color='green', alpha=0.7)
        plt.title(f"Error Improvement (Beam-Unet) ({name})")
        plt.xlabel("Improvement (°)")
        plt.ylabel("Count")
        plt.axvline(0, color='black', linestyle='--')
        plt.show()
    else:
        print(f"Skipping {name}: no ground-truth diffs")

# Display summary table
if summary:
    summary_df = pd.DataFrame(summary)


