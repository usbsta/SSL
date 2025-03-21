import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyproj import Proj, Transformer

# Cargar los archivos CSV
ref_file_path = '/Users/30068385/OneDrive - Western Sydney University/flight records/DJIFlightRecord_2024-09-24_[13-07-49].csv'
file_path_flight = '/Users/30068385/OneDrive - Western Sydney University/flight records/DJIFlightRecord_2024-09-24_[13-23-48].csv'

# Leer el archivo de referencia y de vuelo
ref_data = pd.read_csv(ref_file_path, skiprows=1, delimiter=',', low_memory=False)
flight_data = pd.read_csv(file_path_flight, skiprows=1, delimiter=',', low_memory=False)

# Extraer la posición de referencia (promedio de los valores válidos de latitud y longitud)
reference_latitude = ref_data['OSD.latitude'].dropna().astype(float).mean()
reference_longitude = ref_data['OSD.longitude'].dropna().astype(float).mean()

# Extraer las columnas necesarias: latitud, longitud, altura y tiempo
latitude_col = 'OSD.latitude'
longitude_col = 'OSD.longitude'
altitude_col = 'OSD.altitude [ft]'
time_col = 'CUSTOM.updateTime [local]'

# Filtrar las filas con datos válidos
flight_data = flight_data[[latitude_col, longitude_col, altitude_col, time_col]].dropna()

# Convertir la altitud de pies a metros
flight_data[altitude_col] = flight_data[altitude_col] * 0.3048

# Obtener la altitud inicial del vuelo para usarla en la referencia de elevación
initial_altitude = flight_data[altitude_col].iloc[0]

# Configurar la proyección UTM para convertir las coordenadas geográficas a metros
transformer = Transformer.from_crs('epsg:4326', 'epsg:32756', always_xy=True)  # Ajustar la zona UTM 56 south

# Convertir las coordenadas de referencia a metros
ref_x, ref_y = transformer.transform(reference_longitude, reference_latitude)

# Crear nuevas columnas para las coordenadas en metros en el dataframe de vuelo
flight_data['X_meters'], flight_data['Y_meters'] = transformer.transform(
    flight_data[longitude_col].values,
    flight_data[latitude_col].values
)

# Función para calcular la distancia horizontal entre dos puntos en coordenadas cartesianas
def calculate_horizontal_distance_meters(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Función para calcular el azimuth usando coordenadas en metros
def calculate_azimuth_meters(x1, y1, x2, y2):
    azimuth = np.degrees(np.arctan2(y2 - y1, x2 - x1))
    return azimuth

# Función para calcular la elevación utilizando la distancia horizontal en metros
def calculate_elevation_meters(altitude, x1, y1, x2, y2, reference_altitude):
    horizontal_distance = calculate_horizontal_distance_meters(x1, y1, x2, y2)
    relative_altitude = altitude - reference_altitude  # Calcular la altura relativa respecto a la referencia
    return np.degrees(np.arctan2(relative_altitude, horizontal_distance))

# Calcular los valores iniciales de azimuth y elevación
initial_azimuth = calculate_azimuth_meters(ref_x, ref_y,
                                           flight_data.iloc[0]['X_meters'],
                                           flight_data.iloc[0]['Y_meters'])

initial_elevation = calculate_elevation_meters(flight_data.iloc[0][altitude_col], ref_x, ref_y,
                                               flight_data.iloc[0]['X_meters'], flight_data.iloc[0]['Y_meters'],
                                               initial_altitude)

# Configurar la gráfica en tiempo real
plt.ion()
fig, ax = plt.subplots(figsize=(12, 3))
point, = ax.plot([], [], 'bo', markersize=5)  # Crear un punto en lugar de una línea

ax.set_xlim(-180, 180)  # Rango en el eje X (azimuth en grados de -180 a 180)
ax.set_ylim(0, 90)  # Rango en el eje Y (elevación en grados de 0 a 90)
ax.set_xlabel('Azimuth (degrees)')
ax.set_ylabel('Elevation (degrees)')
ax.set_title('Real-Time Drone Azimuth and Elevation')

# Función de actualización para la animación
def update(frame):
    x = flight_data.iloc[frame]['X_meters']
    y = flight_data.iloc[frame]['Y_meters']
    altitude = flight_data.iloc[frame][altitude_col]

    # Calcular el azimuth y la elevación relativos al punto de referencia
    azimuth = calculate_azimuth_meters(ref_x, ref_y, x, y) - initial_azimuth
    elevation = calculate_elevation_meters(altitude, ref_x, ref_y, x, y, initial_altitude) - initial_elevation

    point.set_data([azimuth], [elevation])

    plt.draw()
    #plt.pause(0.05)
    plt.pause(0.01)

# Animación en tiempo real
for i in range(len(flight_data)):
    update(i)

# Desactivar el modo interactivo
plt.ioff()
plt.show()