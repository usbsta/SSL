import pandas as pd
import numpy as np
from pyproj import Transformer

# Load CSV files
ref_df = pd.read_csv('/Users/30068385/OneDrive - Western Sydney University/FlightRecord/DJI Air 3/CSV/18 Mar 25/Ref/Mar-18th-2025-10-31AM-Flight-Airdata.csv')
flight_df = pd.read_csv('/Users/30068385/OneDrive - Western Sydney University/FlightRecord/DJI Air 3/CSV/18 Mar 25/1/Mar-18th-2025-11-19AM-Flight-Airdata.csv')

# Normalize column names
ref_df.columns = [col.strip().lower() for col in ref_df.columns]
flight_df.columns = [col.strip().lower() for col in flight_df.columns]

# Identify GPS columns
lat_col_ref = [col for col in ref_df.columns if 'lat' in col][0]
lon_col_ref = [col for col in ref_df.columns if 'lon' in col][0]
alt_col_ref = [col for col in ref_df.columns if 'alt' in col][0]

lat_col_flight = [col for col in flight_df.columns if 'lat' in col][0]
lon_col_flight = [col for col in flight_df.columns if 'lon' in col][0]
alt_col_flight = [col for col in flight_df.columns if 'alt' in col][0]

# Extract reference GPS point
lat0 = ref_df[lat_col_ref].dropna().values[0]
lon0 = ref_df[lon_col_ref].dropna().values[0]
alt0 = ref_df[alt_col_ref].dropna().values[0]

# Initialize UTM projection (zone 18N for your region)
transformer = Transformer.from_crs("EPSG:4326", "EPSG:32618", always_xy=True)

# Reference point in UTM
x0, y0 = transformer.transform(lon0, lat0)

# Convert flight data to relative positions
relative_positions = []
for _, row in flight_df.iterrows():
    try:
        lat = float(row[lat_col_flight])
        lon = float(row[lon_col_flight])
        alt = float(row[alt_col_flight])
        x, y = transformer.transform(lon, lat)
        dx = x - x0  # East (Unity X)
        dz = y - y0  # North (Unity Z)
        dy = alt - alt0  # Up (Unity Y)
        relative_positions.append([dx, dy, dz])
    except:
        continue  # Skip invalid rows

# Save to CSV
relative_positions_np = np.array(relative_positions)
np.savetxt('flight_positions_unity.csv', relative_positions_np, delimiter=',', header='X,Y,Z', comments='')
