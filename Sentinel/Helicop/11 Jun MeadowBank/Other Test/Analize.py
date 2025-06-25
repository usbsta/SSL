import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("triangulation_debug_4arrays2.csv")
avg=40

# Apply 5-sample moving average to raw azimuth and elevation for all arrays
for array in ['N', 'S', 'E', 'W']:
    df[f"smooth_raw_el_{array}"] = df[f"raw_el_{array}"].rolling(window=avg, center=True).median()
    df[f"smooth_raw_az_{array}"] = df[f"raw_az_{array}"].rolling(window=avg, center=True).median()

# Define time window
start_time = 0.0
end_time = 104
df_section = df[(df["time_s"] >= start_time) & (df["time_s"] <= end_time)]



# Elevation plot
plt.figure(figsize=(12, 6))
plt.plot(df_section["time_s"], df_section["smooth_raw_el_N"], label="North")
plt.plot(df_section["time_s"], df_section["smooth_raw_el_S"], label="South")
plt.plot(df_section["time_s"], df_section["smooth_raw_el_E"], label="East")
plt.plot(df_section["time_s"], df_section["smooth_raw_el_W"], label="West")
plt.xlabel("Time [s]")
plt.ylabel("Raw Elevation [°]")
plt.title(f"Raw Elevation vs Time ({start_time}s–{end_time}s)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Azimuth plot
plt.figure(figsize=(12, 6))
plt.plot(df_section["time_s"], df_section["smooth_raw_az_N"], label="North")
plt.plot(df_section["time_s"], df_section["smooth_raw_az_S"], label="South")
plt.plot(df_section["time_s"], df_section["smooth_raw_az_E"], label="East")
plt.plot(df_section["time_s"], df_section["smooth_raw_az_W"], label="West")
plt.xlabel("Time [s]")
plt.ylabel("Raw Azimuth [°]")
plt.title(f"Raw Azimuth vs Time ({start_time}s–{end_time}s)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
