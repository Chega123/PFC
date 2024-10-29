import soundfile as sf

# Ruta al archivo .wav
file_path = '/media/chega/Nuevo vol/PFC1/multimodal_emotion_recognition/data/speech/Session1/Female/Ses01F_impro01_F000_6.2901_8.2357.wav'

# Leer el archivo y obtener información
data, samplerate = sf.read(file_path)
print(f'Tamaño de audio: {data.shape}')  # Tamaño de la matriz de audio
print(f'Tasa de muestreo: {samplerate}') 