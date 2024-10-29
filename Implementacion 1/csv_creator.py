import os
import pandas as pd

# Directorios base
speech_base_dir = "/media/chega/Nuevo vol/PFC1/multimodal_emotion_recognition/data/speech"
text_base_dir = "/media/chega/Nuevo vol/PFC1/multimodal_emotion_recognition/data/text"
sessions = [f"Session{i}" for i in range(1, 6)]  # Sesiones 1 a 5

output_csv = "/media/chega/Nuevo vol/PFC1/multimodal_emotion_recognition/wav2vec/utils/dataset.csv"
data = []

# Función para procesar el texto
def process_text(text):
    # Eliminar tokens especiales y convertir a mayúsculas
    return text.replace("[LAUGHER]", "").replace("[BREATHING]", "").strip().upper()

# Recorrer todas las sesiones, géneros y archivos
for session in sessions:
    for gender in ["Female", "Male"]:
        speech_dir = os.path.join(speech_base_dir, session, gender)
        text_dir = os.path.join(text_base_dir, session, gender)

        # Asegurarse de que las carpetas existen
        if os.path.exists(speech_dir) and os.path.exists(text_dir):
            for wav_file in os.listdir(speech_dir):
                if wav_file.endswith(".wav"):
                    # Nombre del archivo sin extensión
                    base_filename = os.path.splitext(wav_file)[0]

                    # Ruta del archivo de audio
                    wav_path = os.path.join(speech_dir, wav_file)

                    # Archivo de texto correspondiente
                    txt_file = f"{base_filename}.txt"
                    txt_path = os.path.join(text_dir, txt_file)

                    # Verificar que el archivo de texto existe
                    if os.path.exists(txt_path):
                        # Leer el archivo de texto
                        with open(txt_path, 'r') as f:
                            text_line = f.readline().strip()
                            # Eliminar la parte de la etiqueta de tiempo y el texto extra
                            utterance = text_line.split("]:")[-1].strip()

                            # Procesar el texto
                            processed_text = process_text(utterance)

                            # Añadir la fila al dataset
                            data.append({
                                "audio_path": wav_path,
                                "utterance": processed_text
                            })

# Guardar los datos en el CSV
df = pd.DataFrame(data)
df.to_csv(output_csv, sep=";", index=False)

print(f"CSV generado en {output_csv} con {len(df)} filas.")
