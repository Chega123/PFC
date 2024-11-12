import torch
from torch import nn
from einops import rearrange
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import torch.nn as nn
from sklearn.model_selection import train_test_split


class MultimodalDataset(Dataset):
    def __init__(self, eng_dir, fr_dir, audio_dir, labels_dir, labels_dict, transform=None):
        self.eng_dir = eng_dir
        self.fr_dir = fr_dir
        self.audio_dir = audio_dir
        self.labels_dir = labels_dir
        self.labels_dict = labels_dict
        self.transform = transform

        # Obtener lista de archivos de características (asumiendo que tienen la misma estructura en cada directorio)
        self.files = [f.replace('.npy', '') for f in os.listdir(audio_dir) if f.endswith('.npy')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]

        # Cargar etiqueta desde el archivo CSV correspondiente
        label_file = os.path.join(self.labels_dir, f"{filename}.csv")
        label_data = pd.read_csv(label_file, header=None)
        label = self.labels_dict[label_data.iloc[0, 2]]  # Asume que la etiqueta está en la primera fila, segunda columna

        # Cargar características
        eng_path = os.path.join(self.eng_dir, f"{filename}.npy")
        fr_path = os.path.join(self.fr_dir, f"{filename}.npy")
        audio_path = os.path.join(self.audio_dir, f"{filename}.npy")

        eng_features = np.load(eng_path)
        fr_features = np.load(fr_path)
        audio_features = np.load(audio_path)

        # Convertir a tensores
        eng_features = torch.tensor(eng_features, dtype=torch.float32)
        fr_features = torch.tensor(fr_features, dtype=torch.float32)
        audio_features = torch.tensor(audio_features, dtype=torch.float32)


        eng_features = eng_features.clone().detach().squeeze(0)
        fr_features = fr_features.clone().detach().squeeze(0)
        # Concatenar las características
        combined_features = torch.cat([eng_features, fr_features, audio_features], dim=0)

        return combined_features, label
"""         print(f"eng_features shape: {eng_features.shape}")
        print(f"fr_features shape: {fr_features.shape}")
        print(f"audio_features shape: {audio_features.shape}") """
        

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, q, **kwargs):
        return self.fn(self.norm(x), q, **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, q=None):  # q is passed only for easier code
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, q):
        kv = self.to_kv(x).chunk(2, dim=-1)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), kv)
        q = rearrange(self.to_q(q), 'b n (h d) -> b h n d', h=self.heads)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, q):
        for attn, ff in self.layers:
            x = attn(x, q) + x
            x = ff(x, q) + x
        return x


class CascadedCrossModalTransformer(nn.Module):
    def __init__(self, num_classes, num_patches, dim, depth, heads, mlp_dim, dim_head=64, dropout=0.10):
        super().__init__()
        
        # Actualiza self.ppm de acuerdo con num_patches
        self.ppm = num_patches // 3  # Número de patches por modalidad
        assert num_patches % 3 == 0, "The number of patches must be divisible by 3 to split evenly across modalities!"

        # Asegúrate de que los embeddings de posición coincidan con el nuevo self.ppm
        self.pos_embedding_text = nn.Parameter(torch.randn(1, self.ppm, dim))
        self.pos_embedding_text_en = nn.Parameter(torch.randn(1, self.ppm, dim))
        self.pos_embedding_audio = nn.Parameter(torch.randn(1, self.ppm, dim))

        # Módulos de transformadores cruzados para cada modalidad
        self.cross_tr_language = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.cross_tr_speech = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        # Capa de salida
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Prints de diagnóstico
        """ print(f"x shape: {x.shape}")
        print(f"pos_embedding_text shape: {self.pos_embedding_text.shape}")
        print(f"pos_embedding_text_en shape: {self.pos_embedding_text_en.shape}")
        print(f"pos_embedding_audio shape: {self.pos_embedding_audio.shape}") """
        
        # División en modalidades y suma de embeddings de posición
        text1_tokens = x[:, :self.ppm] + self.pos_embedding_text
        #print(f"text1_tokens shape after addition: {text1_tokens.shape}")
        
        text2_tokens = x[:, self.ppm:2*self.ppm] + self.pos_embedding_text_en
        #print(f"text2_tokens shape after addition: {text2_tokens.shape}")
        
        audio_tokens = x[:, 2*self.ppm:3*self.ppm] + self.pos_embedding_audio
        #print(f"audio_tokens shape after addition: {audio_tokens.shape}")

        # Transformadores cruzados
        tokens_text_cross = self.cross_tr_language(text1_tokens, text2_tokens)
        tokens_cross = self.cross_tr_speech(tokens_text_cross, audio_tokens)

        x = tokens_cross[:, 0]
        return self.mlp_head(x)


def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0

    for features, labels in dataloader:
        features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / len(dataloader.dataset)
    return total_loss / len(dataloader), accuracy


def evaluate(model, dataloader, criterion, device, num_classes):
    model.eval()
    total_loss = 0
    correct = 0
    class_correct = [0] * num_classes  # Correctas por clase
    class_total = [0] * num_classes    # Totales por clase

    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

            # Actualización de aciertos y totales por clase
            for i in range(len(labels)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1

    # Cálculo de precisión general
    accuracy = 100 * correct / len(dataloader.dataset)
    
    # Cálculo de precisión por clase
    class_accuracy = [100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0 for i in range(num_classes)]

    return total_loss / len(dataloader), accuracy, class_accuracy


def main():
    # Configuración
    train_eng_dir = '/media/chega/Nuevo vol/Implementacion 2/test2/processed_texts_train/ingles'
    train_fr_dir = '/media/chega/Nuevo vol/Implementacion 2/test2/processed_texts_train/frances'
    train_audio_dir = '/media/chega/Nuevo vol/Implementacion 2/test2/features_audio_train'
    train_labels_dir = '/media/chega/Nuevo vol/Implementacion 2/data/labels/train'

    test_eng_dir = '/media/chega/Nuevo vol/Implementacion 2/test2/processed_texts_test/ingles'
    test_fr_dir = '/media/chega/Nuevo vol/Implementacion 2/test2/processed_texts_test/frances'
    test_audio_dir = '/media/chega/Nuevo vol/Implementacion 2/test2/features_audio_test'
    test_labels_dir = '/media/chega/Nuevo vol/Implementacion 2/data/labels/test'

    labels_dict = {
        "Neutral": 0,
        "Angry": 1,
        "Happy": 2,
        "Sad": 3,
    }

    batch_size = 16
    num_epochs = 400
    learning_rate = 5e-5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    # Datasets y DataLoaders para entrenamiento y prueba
    train_dataset = MultimodalDataset(train_eng_dir, train_fr_dir, train_audio_dir, train_labels_dir, labels_dict)
    test_dataset = MultimodalDataset(test_eng_dir, test_fr_dir, test_audio_dir, test_labels_dir, labels_dict)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Modelo
    model = CascadedCrossModalTransformer(num_classes=len(labels_dict), num_patches=120, dim=1024, depth=6, heads=6, mlp_dim=512)
    model.to(device)

    # Pérdida y optimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # Entrenamiento y evaluación
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs}")
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc, class_acc = evaluate(model, test_loader, criterion, device, len(labels_dict))
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
        
        # Imprime la precisión por clase
        for emotion, acc in zip(labels_dict.keys(), class_acc):
            print(f"Test Accuracy for {emotion}: {acc:.2f}%")
    
    # Guardar el modelo
    torch.save(model.state_dict(), "ccmt_model.pth")
    print("Modelo guardado en ccmt_model.pth")

if __name__ == "__main__":
    main()
