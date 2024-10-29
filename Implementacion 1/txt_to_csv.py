import os
import csv

# Directorio base donde están los archivos .txt
txt_dir = '/media/chega/Nuevo vol/PFC1/multimodal_emotion_recognition/data/labels/Session5'
csv_dir = '/media/chega/Nuevo vol/PFC1/multimodal_emotion_recognition/data/csv_labels/Session5'  # Carpeta base para guardar los archivos .csv

# Crear directorios de salida para Male y Female si no existen
male_csv_dir = os.path.join(csv_dir, 'Male')
female_csv_dir = os.path.join(csv_dir, 'Female')
os.makedirs(male_csv_dir, exist_ok=True)
os.makedirs(female_csv_dir, exist_ok=True)

# Función para convertir .txt a .csv
def convert_txt_to_csv(txt_file, csv_file):
    with open(txt_file, 'r') as infile, open(csv_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        # Leer la línea del archivo .txt
        line = infile.readline().strip()
        # Dividir por tabulaciones
        parts = line.split("\t")
        # Escribir los datos en formato CSV
        writer.writerow(parts)

# Recorrer los archivos en las carpetas Male y Female dentro de txt_dir
for gender in ['Male', 'Female']:
    gender_txt_dir = os.path.join(txt_dir, gender)
    gender_csv_dir = os.path.join(csv_dir, gender)

    # Asegurarse de que la carpeta de salida existe para cada género
    os.makedirs(gender_csv_dir, exist_ok=True)

    # Procesar los archivos .txt en la carpeta correspondiente a cada género
    for file_name in os.listdir(gender_txt_dir):
        if file_name.endswith('.txt'):
            txt_file_path = os.path.join(gender_txt_dir, file_name)
            csv_file_path = os.path.join(gender_csv_dir, file_name.replace('.txt', '.csv'))
            # Convertir el archivo .txt a .csv
            convert_txt_to_csv(txt_file_path, csv_file_path)
            print(f"Convertido: {file_name} a CSV en {gender}.")
