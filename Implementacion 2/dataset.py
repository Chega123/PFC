import os
from torch.utils.data import Dataset

class AudioTextDataset(Dataset):
    def __init__(self, audio_dir, label_dir, max_length=16000):
        self.audio_dir = audio_dir
        self.label_dir = label_dir
        self.max_length = max_length
        self.data = self.load_data()

    def load_data(self):
        data = []
        # Itera sobre cada archivo CSV en la carpeta de etiquetas
        for label_file in os.listdir(self.label_dir):
            if label_file.endswith('.csv'):
                full_label_path = os.path.join(self.label_dir, label_file)
                base_filename = label_file[:-4]  # Remove ".csv" extension

                audio_file = f"{base_filename}.wav"
                audio_path = os.path.join(self.audio_dir, audio_file)
                print(audio_path)
                if os.path.exists(audio_path):  # Verifica si el archivo de audio existe
                    with open(full_label_path, 'r') as file:
                        for line in file:
                            time_range, filename, label = line.strip().split(',')
                            print(label)
                            data.append((audio_path, label))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path, label = self.data[idx]
        label = self.convert_label_to_int(label)   
        return audio_path, label


    def convert_label_to_int(self, label):
        # Mapea las etiquetas a índices numéricos
        label_map = {'Neutral': 0, 'Happy': 1, 'Sad': 2, 'Angry': 3}  # Ejemplo, ajusta según tus etiquetas
        return label_map.get(label, -1)
