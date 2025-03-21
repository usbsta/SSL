import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#csv_file_path = '/Users/30068385/OneDrive - Western Sydney University/FlightRecord/Holybro X500/CSV/25 Nov/4 Dataset/X500_19.csv'
csv_file_path = '/Users/30068385/OneDrive - Western Sydney University/FlightRecord/DJI Air 3/CSV/18 Mar 25/1/Mar-18th-2025-11-19AM-Flight-Airdata.csv'

#csv_data = pd.read_csv(csv_file_path, encoding='utf-16', engine='python')
csv_data = pd.read_csv(csv_file_path, encoding='utf-8', engine='python')

print(csv_data.head())

missing_data = csv_data.isnull().sum()
print("\nDatos faltantes por columna:")
print(missing_data)

missing_percentage = (missing_data / len(csv_data)) * 100
print("\nPorcentaje de datos faltantes por columna:")
print(missing_percentage)

# Estadísticas descriptivas
print("\nEstadísticas descriptivas:")
print(csv_data.describe(include='all'))

# Seleccionar columnas numéricas
numeric_cols = csv_data.select_dtypes(include=[np.number])

# Verificar si hay columnas numéricas
if numeric_cols.empty:
    print("\nNo se encontraron columnas numéricas para análisis.")
else:
    # Detectar outliers con IQR en columnas numéricas
    Q1 = numeric_cols.quantile(0.25)
    Q3 = numeric_cols.quantile(0.75)
    IQR = Q3 - Q1
    outliers = (numeric_cols < (Q1 - 1.5 * IQR)) | (numeric_cols > (Q3 + 1.5 * IQR))
    print("\nNúmero de outliers detectados en cada columna numérica:")
    print(outliers.sum())

    # Graficar boxplots sólo si hay datos numéricos
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=numeric_cols)
    plt.title('Boxplots de columnas numéricas')
    plt.show()

# Filas completamente vacías
empty_rows = csv_data.isnull().all(axis=1).sum()
print(f"\nNúmero de filas completamente vacías: {empty_rows}")

# Columnas con datos no numéricos
non_numeric_columns = csv_data.select_dtypes(exclude=[np.number])
print("\nColumnas con datos no numéricos:")
print(non_numeric_columns.head())

# Visualización de la matriz de correlación si hay datos numéricos suficientes
if not numeric_cols.empty:
    plt.figure(figsize=(10, 8))
    corr_matrix = numeric_cols.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Mapa de calor de las correlaciones')
    plt.show()
