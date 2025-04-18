import pdfplumber
import pandas as pd

# Ruta al archivo PDF
pdf_path = 'historiaLaboral-1.pdf'

# Lista para almacenar las tablas extraídas
all_tables = []

with pdfplumber.open(pdf_path) as pdf:
    for i, page in enumerate(pdf.pages):
        # Extrae tablas de la página
        tables = page.extract_tables()
        for table in tables:
            df = pd.DataFrame(table)
            df['page'] = i + 1  # Opcional: agrega número de página
            all_tables.append(df)

# Combina todas las tablas en un solo DataFrame
if all_tables:
    final_df = pd.concat(all_tables, ignore_index=True)

    # Guarda en Excel
    final_df.to_excel('tabla_extraida.xlsx', index=False)
    print("Tabla exportada a 'tabla_extraida.xlsx'")
else:
    print("No se encontraron tablas en el PDF.")
