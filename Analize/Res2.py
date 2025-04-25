import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Updated file paths
files = {
    "Training-day": "test_2.csv",
    "Different-location": "test_Oct.csv",
    "No-drone": "test_No_drone.csv"
}

summary = []

for name, path in files.items():
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"File not found: {path}")
        continue

    required_cols = {
        'az_diff', 'el_diff', 'az_diff_unet', 'el_diff_unet',
        'az_csv', 'el_csv', 'az_est', 'el_est', 'az_centroid', 'el_centroid',
        'total_distance'
    }
    if required_cols.issubset(df.columns):
        df['err_beam'] = np.sqrt(df['az_diff'] ** 2 + df['el_diff'] ** 2)
        df['err_unet'] = np.sqrt(df['az_diff_unet'] ** 2 + df['el_diff_unet'] ** 2)

        # Compute summary statistics
        stats = {
            'Dataset': name,
            'Mean Beam (°)': df['err_beam'].mean(),
            'Median Beam (°)': df['err_beam'].median(),
            '25th Beam (°)': df['err_beam'].quantile(0.25),
            '75th Beam (°)': df['err_beam'].quantile(0.75),
            'Mean UNet (°)': df['err_unet'].mean(),
            'Median UNet (°)': df['err_unet'].median(),
            '25th UNet (°)': df['err_unet'].quantile(0.25),
            '75th UNet (°)': df['err_unet'].quantile(0.75)
        }
        summary.append(stats)

        # Boxplot
        plt.figure()
        plt.boxplot([df['err_beam'], df['err_unet']], labels=['Beam', 'UNet'])
        plt.title(f"Error Distribution ({name})")
        plt.ylabel("Combined Error (°)")
        plt.show()

        # Scatter GT vs Predictions (sampled)
        sample = df.sample(frac=0.1, random_state=0)
        plt.figure()
        plt.scatter(sample['az_csv'], sample['el_csv'], s=5, alpha=0.3, label='Ground Truth')
        plt.scatter(sample['az_est'], sample['el_est'], s=5, alpha=0.3, label='Beamforming')
        plt.scatter(sample['az_centroid'], sample['el_centroid'], s=5, alpha=0.3, label='UNet')
        plt.title(f"Spatial Predictions vs Ground Truth ({name})")
        plt.xlabel("Azimuth (°)")
        plt.ylabel("Elevation (°)")
        plt.legend()
        plt.show()

        # Binned mean error vs distance
        bins = np.arange(0, df['total_distance'].max() + 5, 5)
        df['dist_bin'] = pd.cut(df['total_distance'], bins, include_lowest=True)
        grouped = df.groupby('dist_bin').agg(
            mean_beam=('err_beam', 'mean'), std_beam=('err_beam', 'std'),
            mean_unet=('err_unet', 'mean'), std_unet=('err_unet', 'std')
        ).reset_index()
        grouped['bin_center'] = grouped['dist_bin'].apply(lambda x: x.mid)

        plt.figure()
        plt.errorbar(grouped['bin_center'], grouped['mean_beam'],
                     yerr=grouped['std_beam'], marker='o', linestyle='-',
                     label='Beamforming')
        plt.errorbar(grouped['bin_center'], grouped['mean_unet'],
                     yerr=grouped['std_unet'], marker='s', linestyle='-',
                     label='UNet')
        plt.title(f"Binned Mean Error vs Distance ({name})")
        plt.xlabel("Distance (m)")
        plt.ylabel("Mean Error (°)")
        plt.legend()
        plt.show()
    else:
        print(f"Skipping {name}: missing required columns")

if summary:
    summary_df = pd.DataFrame(summary)
    display_dataframe_to_user("Error Statistics by Method and Dataset", summary_df)
