import os
import torch
import soundfile as sf
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from torch import nn

# Configuración de dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Configuración del modelo y parámetros
MODEL_TYPE = "facebook/wav2vec2-large-960h"
processor = Wav2Vec2Processor.from_pretrained(MODEL_TYPE)
model = Wav2Vec2Model.from_pretrained(MODEL_TYPE).to(device)

# Parámetros de procesamiento
output_dim = 1024    # Dimensión deseada de salida
num_tokens = 300     # Número de tokens que queremos tener

# Directorios de entrada y salida
audio_dir = "/media/chega/Nuevo vol/Implementacion 2/data/speech/train"
out_dir = "/media/chega/Nuevo vol/Implementacion 2/test2/features_audio"
os.makedirs(out_dir, exist_ok=True)

# Proyección de características a la dimensión deseada
projection_layer = nn.Linear(model.config.hidden_size, output_dim).to(device)

def process_audio(audio_path):
    """Procesa el audio para extraer características con wav2vec y aplica recorte/padding."""
    # Carga del audio
    speech, _ = sf.read(audio_path)
    inputs = processor(speech, return_tensors="pt", sampling_rate=16000).to(device)

    # Extrae características con wav2vec
    with torch.no_grad():
        hidden_states = model(**inputs).last_hidden_state  # Características originales

    # Proyección a la dimensión deseada si es necesario
    if hidden_states.size(-1) != output_dim:
        hidden_states = projection_layer(hidden_states)

    # Ajuste del número de tokens
    if hidden_states.size(1) > num_tokens:
        indices = torch.randperm(hidden_states.size(1))[:num_tokens]
        features = hidden_states[:, indices, :]
    elif hidden_states.size(1) < num_tokens:
        pad_size = num_tokens - hidden_states.size(1)
        padding = torch.zeros(hidden_states.size(0), pad_size, hidden_states.size(2), device=device)
        features = torch.cat((hidden_states, padding), dim=1)
    else:
        features = hidden_states

    return features.squeeze(0).cpu().numpy()

def save_features(audio_path):
    """Guarda las características procesadas en un archivo .npy."""
    features = process_audio(audio_path)
    file_name = os.path.basename(audio_path).replace(".wav", ".npy")
    np.save(os.path.join(out_dir, file_name), features)
    print(f"Características guardadas en {file_name}")

# Procesa cada archivo de audio en el directorio
for file_name in os.listdir(audio_dir):
    if file_name.endswith(".wav"):
        audio_path = os.path.join(audio_dir, file_name)
        save_features(audio_path)
