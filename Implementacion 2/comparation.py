import os
import pandas as pd

def encontrar_no_transcritos(carpeta_audios, carpeta_transcripciones):
    # Obtener los nombres de archivos (sin extensiones) de ambas carpetas
    audios = {os.path.splitext(f)[0] for f in os.listdir(carpeta_audios) if f.endswith('.npy')}
    transcripciones = {os.path.splitext(f)[0] for f in os.listdir(carpeta_transcripciones) if f.endswith('.txt')}
    
    # Encontrar los audios que no tienen transcripción
    no_transcritos = audios - transcripciones
    
    return no_transcritos


def eliminar_no_transcritos(carpeta_audios, carpeta_transcripciones):
    # Obtener los nombres de archivos (sin extensiones) de ambas carpetas
    audios = {os.path.splitext(f)[0] for f in os.listdir(carpeta_audios) if f.endswith('.npy')}
    transcripciones = {os.path.splitext(f)[0] for f in os.listdir(carpeta_transcripciones) if f.endswith('.txt')}
    
    # Encontrar los audios que no tienen transcripción
    no_transcritos = audios - transcripciones

    # Eliminar los archivos .npy correspondientes a los audios no transcritos
    for archivo in no_transcritos:
        ruta_archivo = os.path.join(carpeta_audios, f"{archivo}.npy")
        try:
            os.remove(ruta_archivo)
            print(f"Archivo eliminado: {ruta_archivo}")
        except Exception as e:
            print(f"No se pudo eliminar {ruta_archivo}: {e}")

def eliminar_csv_sin_audio(carpeta_audios, carpeta_csv):
    # Obtener los nombres de los archivos de audio sin la extensión
    audios = {os.path.splitext(f)[0] for f in os.listdir(carpeta_audios) if f.endswith('.npy')}  # Cambiar a .wav si es necesario

    # Obtener los nombres de los archivos CSV sin la extensión
    csv_files = {os.path.splitext(f)[0] for f in os.listdir(carpeta_csv) if f.endswith('.csv')}

    # Encontrar los archivos CSV que no tienen un archivo de audio correspondiente
    csv_sin_audio = csv_files - audios

    # Eliminar los archivos CSV que no tienen un audio correspondiente
    for archivo in csv_sin_audio:
        ruta_csv = os.path.join(carpeta_csv, f"{archivo}.csv")
        try:
            os.remove(ruta_csv)
            print(f"Archivo CSV eliminado: {ruta_csv}")
        except Exception as e:
            print(f"No se pudo eliminar {ruta_csv}: {e}")



""" # Uso de la función
carpeta_audios = "/media/chega/Nuevo vol/Implementacion 2/test2/features_audio"
carpeta_transcripciones = "/media/chega/Nuevo vol/Implementacion 2/test2/transcripciones/ingles"

faltantes = encontrar_no_transcritos(carpeta_audios, carpeta_transcripciones)
if faltantes:
    print("Los siguientes archivos de audio no tienen transcripción:")
    for archivo in faltantes:
        print(archivo)
else:
    print("Todos los archivos de audio tienen transcripción.")
 """

""" carpeta_audios = "/media/chega/Nuevo vol/Implementacion 2/test2/features_audio_test"
carpeta_transcripciones = "/media/chega/Nuevo vol/Implementacion 2/test2/transcripciones_test/ingles"

eliminar_no_transcritos(carpeta_audios, carpeta_transcripciones)  """

carpeta_audios = "/media/chega/Nuevo vol/Implementacion 2/test2/features_audio_test" # Carpeta donde están los audios
archivo_csv = "/media/chega/Nuevo vol/Implementacion 2/data/labels/test"  # Ruta al archivo CSV con las etiquetas
eliminar_csv_sin_audio(carpeta_audios, archivo_csv)