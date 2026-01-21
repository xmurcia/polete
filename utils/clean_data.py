import pandas as pd
import glob
import os

def procesar_datos_csv(directorio_entrada, archivo_salida):
    """
    Lee todos los CSVs, elimina duplicados por fecha y genera un dataset limpio.
    """
    # 1. Buscar todos los archivos CSV en el directorio
    ruta_busqueda = os.path.join(directorio_entrada, "*.csv")
    archivos = glob.glob(ruta_busqueda)
    
    if not archivos:
        print(f"❌ No se encontraron archivos CSV en: {directorio_entrada}")
        return

    print(f"📂 Encontrados {len(archivos)} archivos. Procesando...")

    lista_dfs = []
    
    # 2. Leer cada archivo
    for archivo in archivos:
        try:
            df_temp = pd.read_csv(archivo)
            # Aseguramos que la columna de fecha sea datetime
            # Ajusta 'Date/Time' si tus CSV tienen otro nombre de columna
            if 'Date/Time' in df_temp.columns:
                df_temp['Date/Time'] = pd.to_datetime(df_temp['Date/Time'])
                lista_dfs.append(df_temp)
            else:
                print(f"⚠️ Saltando {os.path.basename(archivo)}: No tiene columna 'Date/Time'")
        except Exception as e:
            print(f"❌ Error leyendo {os.path.basename(archivo)}: {e}")

    if not lista_dfs:
        print("No hay datos válidos para procesar.")
        return

    # 3. Unir todos los datos (Concatenación)
    df_raw = pd.concat(lista_dfs, ignore_index=True)
    total_filas_originales = len(df_raw)

    # 4. LIMPIEZA CRÍTICA: Eliminar duplicados exactos de fecha
    # Nos quedamos con la primera aparición y borramos el resto
    df_clean = df_raw.drop_duplicates(subset='Date/Time', keep='first')
    
    # 5. Ordenar cronológicamente
    df_clean = df_clean.sort_values('Date/Time').reset_index(drop=True)

    # (Opcional) Recalcular el acumulativo si existe la columna 'Posts'
    # Esto es vital porque al unir trozos, el acumulativo original pierde sentido
    if 'Posts' in df_clean.columns:
        print("🔄 Recalculando columna 'Cumulative' para mantener coherencia...")
        df_clean['Cumulative'] = df_clean['Posts'].cumsum()

    filas_limpias = len(df_clean)
    duplicados_borrados = total_filas_originales - filas_limpias

    # 6. Guardar
    df_clean.to_csv(archivo_salida, index=False)

    print("-" * 30)
    print(f"✅ ¡Éxito! Archivo guardado en: {archivo_salida}")
    print(f"📊 Estadísticas:")
    print(f"   - Filas procesadas (con solapamiento): {total_filas_originales}")
    print(f"   - Duplicados eliminados: {duplicados_borrados}")
    print(f"   - Filas finales (datos reales): {filas_limpias}")
    print("-" * 30)

# --- CONFIGURACIÓN ---
# Pon aquí la ruta donde tienes tus CSVs sucios
CARPETA_CON_CSVS = "./mis_datos_sucios" 
# Nombre del archivo final
ARCHIVO_FINAL = "dataset_entrenamiento_limpio.csv"

if __name__ == "__main__":
    # Crear carpeta dummy para el ejemplo si no existe (puedes borrar esto)
    if not os.path.exists(CARPETA_CON_CSVS):
        os.makedirs(CARPETA_CON_CSVS)
        print(f"ℹ️ Crea la carpeta '{CARPETA_CON_CSVS}' y mete tus csv ahí.")
    
    procesar_datos_csv(CARPETA_CON_CSVS, ARCHIVO_FINAL)