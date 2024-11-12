import os
import torch
import whisper
from langdetect import detect
from transformers import MarianMTModel, MarianTokenizer

# Selección del dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

class AudioProcessor:
    def __init__(self, whisper_model_name="base", translation_model_fr_to_en="Helsinki-NLP/opus-mt-fr-en", translation_model_en_to_fr="Helsinki-NLP/opus-mt-en-fr"):
        """
        Inicializa los modelos necesarios para transcribir y traducir audio.

        Args:
            whisper_model_name (str): Nombre del modelo Whisper para la transcripción.
            translation_model_fr_to_en (str): Modelo de traducción de francés a inglés.
            translation_model_en_to_fr (str): Modelo de traducción de inglés a francés.
        """
        print(f"Inicializando modelos en {device_name}...\n")
        
        # Carga el modelo Whisper
        print("Cargando modelo Whisper...")
        self.whisper_model = whisper.load_model(whisper_model_name).to(device)
        
        # Carga los modelos de traducción
        print("Cargando modelo de traducción (Francés a Inglés)...")
        self.translator_fr_to_en = MarianMTModel.from_pretrained(translation_model_fr_to_en).to(device)
        self.tokenizer_fr_to_en = MarianTokenizer.from_pretrained(translation_model_fr_to_en)
        
        print("Cargando modelo de traducción (Inglés a Francés)...")
        self.translator_en_to_fr = MarianMTModel.from_pretrained(translation_model_en_to_fr).to(device)
        self.tokenizer_en_to_fr = MarianTokenizer.from_pretrained(translation_model_en_to_fr)

        print("Modelos cargados correctamente.\n")

    def transcribe(self, audio_path):
        """
        Realiza la transcripción de un archivo de audio.

        Args:
            audio_path (str): Ruta del archivo de audio.

        Returns:
            str: Texto transcrito del audio.
        """
        try:
            print(f"Transcribiendo archivo: {audio_path}")
            result = self.whisper_model.transcribe(audio_path)
            print("Transcripción completada.")
            return result["text"]
        except Exception as e:
            print(f"Error transcribiendo audio: {e}")
            return None

    def detect_language(self, text):
        """
        Detecta el idioma de un texto.

        Args:
            text (str): Texto cuyo idioma se quiere detectar.

        Returns:
            str: Código del idioma detectado (e.g., 'fr' para francés, 'en' para inglés).
        """
        try:
            print("Detectando idioma del texto...")
            language = detect(text)
            print(f"Idioma detectado: {language}")
            return language
        except Exception as e:
            print(f"Error detectando idioma: {e}")
            return None

    def translate(self, text, source_lang, target_lang):
        """
        Traduce un texto entre francés e inglés.

        Args:
            text (str): Texto a traducir.
            source_lang (str): Idioma de origen ('fr' o 'en').
            target_lang (str): Idioma de destino ('en' o 'fr').

        Returns:
            str: Texto traducido.
        """
        try:
            print(f"Traduciendo texto de {source_lang} a {target_lang}...")
            if source_lang == 'fr' and target_lang == 'en':
                model = self.translator_fr_to_en
                tokenizer = self.tokenizer_fr_to_en
            elif source_lang == 'en' and target_lang == 'fr':
                model = self.translator_en_to_fr
                tokenizer = self.tokenizer_en_to_fr
            else:
                return text  # No se necesita traducción

            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
            translated_tokens = model.generate(**inputs)
            translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
            print("Traducción completada.")
            return translated_text
        except Exception as e:
            print(f"Error traduciendo texto: {e}")
            return None

def save_to_file(text, folder_path, file_name):
    """
    Guarda el texto en un archivo dentro de una carpeta específica.

    Args:
        text (str): Texto a guardar.
        folder_path (str): Ruta de la carpeta donde se guardará el archivo.
        file_name (str): Nombre del archivo.
    """
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, file_name)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Archivo guardado en: {file_path}")

if __name__ == "__main__":
    print(f"Procesando en {device_name}...\n")

    # Ruta a la carpeta con archivos de audio
    input_folder = input("Ingresa la ruta de la carpeta con archivos .wav: ")
    base_output_folder = "transcripciones_test"

    # Crear el procesador
    processor = AudioProcessor()

    # Iterar sobre todos los archivos .wav en la carpeta
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".wav"):
            audio_path = os.path.join(input_folder, file_name)
            print(f"\nProcesando archivo: {file_name}")

            # Transcribir el archivo de audio
            transcribed_text = processor.transcribe(audio_path)

            if transcribed_text:
                print("\nTexto transcrito:")
                print(transcribed_text)

                # Detectar el idioma y traducir según corresponda
                detected_language = processor.detect_language(transcribed_text)

                if detected_language == 'fr':
                    save_to_file(transcribed_text, os.path.join(base_output_folder, "frances"), file_name.replace(".wav", ".txt"))
                    translated_text = processor.translate(transcribed_text, source_lang='fr', target_lang='en')
                    save_to_file(translated_text, os.path.join(base_output_folder, "ingles"), file_name.replace(".wav", ".txt"))
                elif detected_language == 'en':
                    save_to_file(transcribed_text, os.path.join(base_output_folder, "ingles"), file_name.replace(".wav", ".txt"))
                    translated_text = processor.translate(transcribed_text, source_lang='en', target_lang='fr')
                    save_to_file(translated_text, os.path.join(base_output_folder, "frances"), file_name.replace(".wav", ".txt"))
                else:
                    print("\nNo se pudo determinar el idioma o no es compatible para traducción.")
