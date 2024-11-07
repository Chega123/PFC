import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, random_split
from ccmt import CascadedCrossModalTransformer
from dataset import AudioTextDataset
from train_eval import train, evaluate
import torchaudio

# Configuración de datos y modelo
audio_dir = "/media/chega/Nuevo vol/Implementacion 2/data/speech/test"
label_dir = "/media/chega/Nuevo vol/Implementacion 2/data/labels/test"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = AudioTextDataset(audio_dir, label_dir)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
model = CascadedCrossModalTransformer(num_classes=3, dim=1024, depth=6, heads=6, mlp_dim=128).to(device)
criterion = CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
print("########################################################################################################################")
# Entrenamiento
epochs = 10
for epoch in range(epochs):
    train_loss, train_accuracy = train(model, train_loader, optimizer, criterion, device)
    val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
