import os
import torch
import numpy as np
from transformers import CamembertTokenizer, BertTokenizer
from transformers import CamembertModel, BertModel
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TextProcessor:
    def __init__(self, num_tokens=120, dim=1024):
        self.num_tokens = num_tokens
        self.dim = dim
        self.camembert = CamembertModel.from_pretrained("camembert-base").to(device)
        self.camembert_tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
        self.bert = BertModel.from_pretrained("bert-base-uncased").to(device)
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def _process_text(self, text, model, tokenizer):
        # Tokenizar texto
        tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        tokens = {key: value.to(device) for key, value in tokens.items()}

        # Pasar texto por el modelo
        output = model(**tokens)
        embeddings = output.last_hidden_state  # [batch_size, seq_length, hidden_dim]

        # Proyectar a la dimensión deseada si no coincide con self.dim
        if embeddings.size(-1) != self.dim:
            projection_layer = torch.nn.Linear(embeddings.size(-1), self.dim).to(device)
            embeddings = projection_layer(embeddings)

        # Ajustar número de tokens a self.num_tokens
        if embeddings.size(1) > self.num_tokens:
            indices = list(range(1, embeddings.size(1)))
            random.shuffle(indices)
            selected_indices = [0] + indices[:self.num_tokens - 1]  # Incluir token de clase
            embeddings = embeddings[:, selected_indices, :]
        elif embeddings.size(1) < self.num_tokens:
            pad_size = self.num_tokens - embeddings.size(1)
            padding = embeddings[:, -1:, :].repeat(1, pad_size, 1)  # Duplicar el último token
            embeddings = torch.cat((embeddings, padding), dim=1)

        return embeddings

    def process_language_folder(self, input_folder, output_folder, model, tokenizer):
        os.makedirs(output_folder, exist_ok=True)

        for file_name in os.listdir(input_folder):
            if file_name.endswith(".txt"):
                file_path = os.path.join(input_folder, file_name)

                # Leer contenido del archivo
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read().strip()

                # Procesar texto
                embeddings = self._process_text(text, model, tokenizer)

                # Convertir a CPU y a numpy
                embeddings_np = embeddings.detach().cpu().numpy()

                # Guardar como .npy
                output_path = os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}.npy")
                np.save(output_path, embeddings_np)
                print(f"Processed and saved {file_name} to {output_path}")


# Ejemplo de uso
if __name__ == "__main__":
    processor = TextProcessor(num_tokens=120, dim=1024)

    # Carpeta de entrada para cada idioma
    input_folder_fr = "/media/chega/Nuevo vol/Implementacion 2/test2/transcripciones_test/frances"
    input_folder_en = "/media/chega/Nuevo vol/Implementacion 2/test2/transcripciones_test/ingles"

    # Carpetas de salida para cada idioma
    output_folder_fr = "./processed_texts_test/frances"
    output_folder_en = "./processed_texts_test/ingles"

    # Procesar archivos de texto francés
    print("Processing French texts...")
    processor.process_language_folder(input_folder_fr, output_folder_fr, processor.camembert, processor.camembert_tokenizer)

    # Procesar archivos de texto inglés
    print("Processing English texts...")
    processor.process_language_folder(input_folder_en, output_folder_en, processor.bert, processor.bert_tokenizer)
