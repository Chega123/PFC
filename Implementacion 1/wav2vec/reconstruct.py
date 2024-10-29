import numpy as np
import os

def reconstruct_hidden_states(audio_dir):
    """Recargar y concatenar las 12 capas guardadas para un archivo de audio"""
    # Ordenar archivos de capas en el orden original
    layer_files = sorted([os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.startswith("layer_")])
    
    # Cargar y concatenar cada capa
    layers = [np.load(layer_file) for layer_file in layer_files]
    full_embedding = np.concatenate(layers, axis=0)
    return full_embedding

def main(out_dir):
    for root, dirs, files in os.walk(out_dir):
        if files:
            # Concatenar las capas de cada archivo de audio en el formato final
            full_embedding = reconstruct_hidden_states(root)
            
            # Guardar el archivo concatenado en la misma estructura que el código original
            subdir = os.path.dirname(root).replace(out_dir, "").strip("/")
            out_file_path = os.path.join(out_dir, "all_hidden_states", subdir + ".npy")
            np.save(out_file_path, full_embedding)

if __name__ == "__main__":
    # Ruta raíz donde se guardaron los embeddings
    out_dir = "wav2vec/features/large-lv60"
    main(out_dir)
