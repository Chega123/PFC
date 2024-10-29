import os
import subprocess

# Definir directorios de texto y audio
txt_dir = "/media/chega/Nuevo vol/PFC1/multimodal_emotion_recognition/data/text"
audio_dir = "/media/chega/Nuevo vol/PFC1/multimodal_emotion_recognition/data/speech"

# Lista de hablantes para las 5 sesiones (M1, M2, M3, M4, M5 y sus variantes)
speakers = [
    "M2i", "M2s", "F2i", "F2s",
    "M3i", "M3s", "F3i", "F3s",
    "M4i", "M4s", "F4i", "F4s",
    "M5i", "M5s", "F5i", "F5s"
]

# Iterar sobre cada hablante y ejecutar el comando
for speaker in speakers:
    command = ["python", "extract_phonemes.py", "--speaker", speaker, "--txt_dir", txt_dir, "--audio_dir", audio_dir]
    print(f"Ejecutando: {' '.join(command)}")  # Imprimir el comando para seguimiento
    subprocess.run(command)  # Ejecutar el comando
