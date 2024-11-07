import torch
from torch.nn import CrossEntropyLoss
import torch.optim as optim
import torchaudio
from torch.cuda.amp import autocast, GradScaler
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    scaler = torch.amp.GradScaler()  # Updated API
     # Updated API
    total_loss, total_correct = 0, 0  # Escalador para precisión mixta
    
    for audio_path, labels in dataloader:
        audio_path = audio_path[0]  
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = waveform.to(device)
        
        if isinstance(labels, tuple):
            labels = torch.tensor(labels[0], dtype=torch.long).to(device)
        else:
            labels = labels.to(device)

        optimizer.zero_grad()
        print(audio_path)
        
        # Usar precisión mixta para reducir el uso de memoria
        with torch.amp.autocast('cuda'): 
            outputs = model(audio_path)
            loss = criterion(outputs, labels)
        print(audio_path)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        total_correct += (outputs.argmax(1) == labels).sum().item()

    accuracy = total_correct / len(dataloader.dataset)
    return total_loss / len(dataloader), accuracy

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_correct = 0, 0
    
    with torch.no_grad():
        for audio_path, labels in dataloader:
            # Cargar y procesar el audio
            audio_path = audio_path[0]  # Obtener la primera (y única) ruta en el batch
            waveform, sample_rate = torchaudio.load(audio_path)
            waveform = waveform.to(device)
            
            # Convertir labels a tensor y mover al dispositivo
            if isinstance(labels, tuple):
                labels = torch.tensor(labels[0], dtype=torch.long).to(device)
            else:
                labels = labels.to(device)

            # Pasar el audio y texto (francés e inglés) al modelo
            outputs = model(audio_path)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            total_correct += (outputs.argmax(1) == labels).sum().item()

    accuracy = total_correct / len(dataloader.dataset)
    return total_loss / len(dataloader), accuracy
