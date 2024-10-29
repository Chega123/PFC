import soundfile as sf

# Ruta al archivo .wav
data, samplerate = sf.read('/media/chega/Nuevo vol/PFC1/multimodal_emotion_recognition/data/speech/Session1/Female/Ses01F_impro01_F000_6.2901_8.2357.wav')


# Convertir a mono si es estéreo
if data.ndim == 2:
    data = data.mean(axis=1)  # Promedio de los dos canales

# Guardar el audio en un nuevo archivo mono
sf.write('/media/chega/Nuevo vol/PFC1/multimodal_emotion_recognition/data/speech/Session1/Female/Ses01F_impro01_F000_6.2901_8.2357.wav', data, samplerate)

