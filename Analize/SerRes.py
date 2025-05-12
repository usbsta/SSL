import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Cargar archivos CSV
summary_test2 = pd.read_csv("test_2_unet_20250426_001310_summary 2.csv")
summary_oct = pd.read_csv("test_Oct_unet_20250426_001310_summary 1.csv")

# Extraer etiquetas y valores
x_labels = summary_test2["dist_bin"]
x = np.arange(len(x_labels))
width = 0.2

# -------------------
# Plot 1: Angular Error
# -------------------
fig1, ax1 = plt.subplots(figsize=(10, 5))
ax1.bar(x - 1.5 * width, summary_test2['ang_bf_mean'], width, label='Beamforming (Test 2)')
ax1.bar(x - 0.5 * width, summary_test2['ang_un_mean'], width, label='U-Net (Test 2)')
ax1.bar(x + 0.5 * width, summary_oct['ang_bf_mean'], width, label='Beamforming (Test 1)')
ax1.bar(x + 1.5 * width, summary_oct['ang_un_mean'], width, label='U-Net (Test 1)')
ax1.set_ylabel("Mean angular error (°)")
ax1.set_xlabel("Distance range (m)")
ax1.set_title("Angular Error vs Distance")
ax1.set_xticks(x)
ax1.set_xticklabels(x_labels)
ax1.legend()
plt.tight_layout()
plt.savefig("angular_error_vs_distance_combined.png")
plt.show()

# -------------------
# Plot 2: False Negative Rate
# -------------------
fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.bar(x - 1.5 * width, summary_test2['fnr_bf'], width, label='Beamforming (Test 2)')
ax2.bar(x - 0.5 * width, summary_test2['fnr_unet'], width, label='U-Net (Test 2)')
ax2.bar(x + 0.5 * width, summary_oct['fnr_bf'], width, label='Beamforming (Test 1)')
ax2.bar(x + 1.5 * width, summary_oct['fnr_unet'], width, label='U-Net (Test 1)')
ax2.set_ylabel("False negative rate (%)")
ax2.set_xlabel("Distance range (m)")
ax2.set_title("False Negative Rate vs Distance")
ax2.set_xticks(x)
ax2.set_xticklabels(x_labels)
ax2.legend()
plt.tight_layout()
plt.savefig("fnr_vs_distance_combined.png")
plt.show()
