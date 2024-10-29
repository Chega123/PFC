from pydub import AudioSegment
import os
import re

# Definir rutas de entrada y salida
base_dir = "/media/chega/Nuevo vol/IEMOCAP_full_release"
output_speech_dir = "/media/chega/Nuevo vol/PFC1/multimodal_emotion_recognition/data/speech"
output_text_dir = "/media/chega/Nuevo vol/PFC1/multimodal_emotion_recognition/data/text"
output_labels_dir = "/media/chega/Nuevo vol/PFC1/multimodal_emotion_recognition/data/labels"
os.makedirs(output_speech_dir, exist_ok=True)
os.makedirs(output_text_dir, exist_ok=True)
os.makedirs(output_labels_dir, exist_ok=True)

# Emociones de interés
target_emotions = {"ang", "hap", "exc", "neu", "sad"}

# Función para cargar etiquetas de emoción válidas (con al menos dos anotadores)
def load_valid_emotions(session_num):
    valid_utterances = {}
    session_path = os.path.join(base_dir, f"Session{session_num}", "dialog", "EmoEvaluation")
    for emo_file in os.listdir(session_path):
        if emo_file.endswith(".txt"):
            with open(os.path.join(session_path, emo_file), "r") as file:
                for line in file:
                    match = re.match(r"^\[(\d+\.\d+)\s-\s(\d+\.\d+)\]\s(\S+)\s(\w+)", line)
                    if match:
                        start_time, end_time, utterance_id, emotion = match.groups()
                        if emotion in target_emotions:
                            # Combinar "exc" con "hap"
                            if emotion == "exc":
                                emotion = "hap"
                            # Guardar solo si hay dos o más anotaciones con la misma emoción
                            if utterance_id not in valid_utterances:
                                valid_utterances[utterance_id] = (start_time, end_time, emotion)
    return valid_utterances

# Función para convertir el audio a mono y 16 kHz
def convert_audio_to_mono_16kHz(audio_segment):
    return audio_segment.set_channels(1).set_frame_rate(16000)


# Función para dividir el audio y guardar el texto y la etiqueta según las emociones válidas
def split_and_save_data(wav_file, trans_file, session_num, valid_utterances):
    audio = AudioSegment.from_wav(wav_file)
    with open(trans_file, "r") as file:
        for line in file:
            match = re.match(r"^(\S+)\s\[(\d+\.\d+)-(\d+\.\d+)\]:\s(.+)", line)
            if match:
                utterance_id, start_time, end_time, text = match.groups()
                
                # Procesar solo si la frase tiene una emoción válida
                if utterance_id in valid_utterances:
                    start_time, end_time, emotion = valid_utterances[utterance_id]
                    start_ms = int(float(start_time) * 1000)  # Convertir a milisegundos
                    end_ms = int(float(end_time) * 1000)

                    # Extraer y guardar el segmento de audio
                    gender = "Male" if utterance_id[5] == "M" else "Female"
                    session_output_speech_dir = os.path.join(output_speech_dir, f"Session{session_num}", gender)
                    os.makedirs(session_output_speech_dir, exist_ok=True)
                    segment = audio[start_ms:end_ms]
                    
                    # Convertir a mono y 16 kHz
                    segment = convert_audio_to_mono_16kHz(segment)
                    
                    # Exportar el segmento convertido
                    segment.export(os.path.join(session_output_speech_dir, f"{utterance_id}.wav"), format="wav")

                    # Guardar la transcripción en mayúsculas sin tokens especiales
                    session_output_text_dir = os.path.join(output_text_dir, f"Session{session_num}", gender)
                    os.makedirs(session_output_text_dir, exist_ok=True)
                    text = text.replace("[LAUGHTER]", "").replace("[BREATHING]", "").upper().strip()
                    with open(os.path.join(session_output_text_dir, f"{utterance_id}.txt"), "w") as out_file:
                        out_file.write(text)

                    # Guardar la etiqueta en el formato específico en el directorio adecuado
                    session_output_labels_dir = os.path.join(output_labels_dir, f"Session{session_num}", gender)
                    os.makedirs(session_output_labels_dir, exist_ok=True)
                    with open(os.path.join(session_output_labels_dir, f"{utterance_id}.txt"), "w") as out_file:
                        out_file.write(f"[{start_time} - {end_time}]\t{utterance_id}\t{emotion}\n")

# Procesar cada sesión con las emociones válidas
for session_num in range(1, 6):
    # Cargar emociones válidas para la sesión
    valid_utterances = load_valid_emotions(session_num)
    
    # Procesar archivos de audio y transcripción
    session_path = os.path.join(base_dir, f"Session{session_num}", "dialog")
    wav_path = os.path.join(session_path, "wav")
    trans_path = os.path.join(session_path, "transcriptions")

    for trans_file in os.listdir(trans_path):
        if trans_file.endswith(".txt"):
            wav_file = os.path.join(wav_path, trans_file.replace(".txt", ".wav"))
            trans_file_path = os.path.join(trans_path, trans_file)

            if os.path.exists(wav_file):
                split_and_save_data(wav_file, trans_file_path, session_num, valid_utterances)
