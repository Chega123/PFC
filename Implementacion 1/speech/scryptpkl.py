import subprocess
import os
from datetime import datetime

# Configuración base
base_dir = "/media/chega/Nuevo vol/PFC1/multimodal_emotion_recognition"
script_path = os.path.join(base_dir, "speech", "make_pkl_data.py")
fold_base_dir = os.path.join(base_dir, "speech", "folds")
dict_file = os.path.join(base_dir, "aligner", "phone_dict.csv")
labels_dir = os.path.join(base_dir, "data", "labels")
wav_dir = os.path.join(base_dir, "data", "speech")
txt_dir = os.path.join(base_dir, "data", "text")
phone_dir = os.path.join(base_dir, "aligner", "align_results")

# Folds a procesar
folds = [f"fold{i}" for i in range(1, 6)]  # Cambia el rango según el número de folds que tengas

# Log de ejecución
now = datetime.now()
log_path = os.path.join(base_dir, f"execution_log_{now.strftime('%d-%m-%Y_%H-%M-%S')}.log")

def logger(message, highlight=False):
    """ Registro de mensajes en consola y archivo de log """
    with open(log_path, 'a') as log_file:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_message = f"{timestamp} - {message}"
        log_file.write(full_message + "\n")
        print(full_message) if not highlight else print("\033[1;32m" + full_message + "\033[0m")

def execute_make_pkl_data(mode, fold):
    """ Ejecuta make_pkl_data.py para un fold y modo específicos """
    spmel_dir = os.path.join(fold_base_dir, fold, mode)
    command = [
        "python3", script_path,
        "--mode", mode,
        "--spmel_dir", spmel_dir,
        "--dict_file", dict_file,
        "--labels_dir", labels_dir,
        "--wav_dir", wav_dir,
        "--txt_dir", txt_dir,
        "--phone_dir", phone_dir
    ]
    try:
        subprocess.run(command, check=True)
        logger(f"[INFO] Completado: {mode} para {fold}")
    except subprocess.CalledProcessError as e:
        logger(f"[ERROR] Fallo en {mode} para {fold}: {str(e)}", highlight=True)

def main():
    logger("Inicio de ejecución para todos los folds y modos", highlight=True)
    for fold in folds:
        logger(f"Procesando {fold}", highlight=True)
        # Modo 'train'
        execute_make_pkl_data("train", fold)
        # Modo 'test'
        execute_make_pkl_data("test", fold)

    logger("Ejecución completa de todos los modos y folds", highlight=True)

if __name__ == "__main__":
    main()
