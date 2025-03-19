import pandas as pd
import numpy as np

# Load the CSV file. Adjust 'your_file.csv' to the actual filename.
#df = pd.read_csv('/Users/30068385/OneDrive - Western Sydney University/FlightRecord/Holybro X500/CSV/25 Nov/21.csv')
df = pd.read_csv('/Users/30068385/OneDrive - Western Sydney University/FlightRecord/DJI Inspire 1/CSV/03 Mar 25/4/Mar-3rd-2025-02-40PM-Flight-Airdata.csv')


# Asegurarse que la columna de tiempo no tenga duplicados.
df = df.drop_duplicates(subset='time(millisecond)', keep='first')
#df = df.drop_duplicates(subset='OSD.flyTime [s]', keep='first')

# Ahora estableceremos el índice.
df = df.set_index('time(millisecond)')
#df = df.set_index('OSD.flyTime [s]')

# Convert index to numeric type if needed
df.index = pd.to_numeric(df.index, errors='coerce')

# Sort by index
df = df.sort_index()

# Identify the start and end times
start_time = int(df.index.min())
end_time = int(df.index.max())

# Crear un nuevo índice con pasos de 100 ms
new_index = range(start_time, end_time + 1, 100)

# Reindex the DataFrame to have a continuous time vector at 100 ms intervals
df = df.reindex(new_index)

# Separar columnas numéricas y no numéricas
numeric_cols = df.select_dtypes(include=[np.number]).columns
non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns

# Interpolar valores numéricos
df[numeric_cols] = df[numeric_cols].interpolate(method='linear')

# Llenar valores no numéricos hacia adelante
df[non_numeric_cols] = df[non_numeric_cols].ffill()

# Restablecer el índice
df = df.reset_index().rename(columns={'index': 'time(millisecond)'})
#df = df.reset_index().rename(columns={'index': 'OSD.flyTime [s]'})

# Guardar el DataFrame procesado
#df.to_csv('X500_19.csv', index=False)
df.to_csv('Mar-3rd-2025-02-40PM-Flight-Airdata2.csv', index=False)
