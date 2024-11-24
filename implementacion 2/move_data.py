import os
import shutil

def move_files(source_dir, destination_dir, file_extension=None):
    """
    Mueve archivos de una carpeta a otra.

    Args:
        source_dir (str): Carpeta origen donde se encuentran los archivos.
        destination_dir (str): Carpeta destino donde se moverán los archivos.
        file_extension (str, opcional): Extensión de archivo para filtrar (por ejemplo, ".wav"). 
                                        Si es None, se moverán todos los archivos.
    """
    # Verificar si las carpetas existen
    if not os.path.exists(source_dir):
        print(f"La carpeta de origen no existe: {source_dir}")
        return
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)  # Crear la carpeta destino si no existe

    # Iterar sobre los archivos en la carpeta origen
    for file_name in os.listdir(source_dir):
        # Filtrar por extensión, si es necesario
        if file_extension and not file_name.endswith(file_extension):
            continue
        
        # Rutas completa del archivo origen y destino
        source_path = os.path.join(source_dir, file_name)
        destination_path = os.path.join(destination_dir, file_name)

        # Mover el archivo
        try:
            shutil.move(source_path, destination_path)
            print(f"Movido: {file_name} -> {destination_dir}")
        except Exception as e:
            print(f"Error al mover {file_name}: {e}")

# Ejemplo de uso
source_folder = "/media/chega/Nuevo vol/mmer_pfc/MMER/data/iemocap_augmented/out (2)"
destination_folder = "/media/chega/Nuevo vol/mmer_pfc/MMER/data/iemocap_augmented"
file_extension_filter = ".wav"  # Cambia a None si deseas mover todos los archivos

move_files(source_folder, destination_folder, file_extension_filter)
