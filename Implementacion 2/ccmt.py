import torch
import whisper
from torch import nn
import tempfile
import torchaudio
from transformers import MarianMTModel, MarianTokenizer
from transformers import CamembertTokenizer, BertTokenizer

from transformers import Wav2Vec2Model, CamembertModel, BertModel
from einops import rearrange
from langdetect import detect



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Importa módulos de procesamiento del CCMT
class PreNorm(nn.Module):
    # Define PreNorm (sin cambios)
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, q, **kwargs):
        return self.fn(self.norm(x), q, **kwargs)

class FeedForward(nn.Module):
    # Define FeedForward (sin cambios)
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, q=None):
        return self.net(x)

class Attention(nn.Module):
    # Define Attention (sin cambios)
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
    # Define Transformer (sin cambios)
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
    def __init__(self, num_classes, dim, depth, heads, mlp_dim, dim_head=64, dropout=0.20, num_tokens=200):
        super().__init__()

        # Cargar los modelos de traducción y moverlos al dispositivo adecuado
        model_name_fr_to_en = 'Helsinki-NLP/opus-mt-fr-en'
        model_name_en_to_fr = 'Helsinki-NLP/opus-mt-en-fr'
        self.translator_fr_to_en = MarianMTModel.from_pretrained(model_name_fr_to_en).to(device)
        self.tokenizer_fr_to_en = MarianTokenizer.from_pretrained(model_name_fr_to_en)
        self.translator_en_to_fr = MarianMTModel.from_pretrained(model_name_en_to_fr).to(device)
        self.tokenizer_en_to_fr = MarianTokenizer.from_pretrained(model_name_en_to_fr)

        # Cargar modelos preentrenados y moverlos al dispositivo adecuado
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h").to(device)
        self.camembert = CamembertModel.from_pretrained("camembert-base").to(device)
        self.bert = BertModel.from_pretrained("bert-base-uncased").to(device)
        self.whisper_model = whisper.load_model("base").to(device) 

        # Parámetro para el número de tokens
        self.num_tokens = num_tokens

        # Transformadores cruzados
        self.cross_tr_language = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.cross_tr_speech = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        # Capa de clasificación
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes),
            nn.Sigmoid()
        )

    def is_audio_in_french(self, audio_text):
        try:
            language = detect(audio_text)
            return language == 'fr'  # 'fr' es el código para francés
        except Exception as e:
            print(f"Error detecting language: {e}")
            return False

    def transcribe_audio(self, audio_path, sample_rate=16000):
        result = self.whisper_model.transcribe(audio_path)
        return result["text"]

    def process_audio(self, audio_path):
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = waveform.to(device)
        audio_tokens = self.wav2vec(waveform).last_hidden_state  # Genera las características de audio

        # Proyectar a la dimensión correcta si es necesario
        if audio_tokens.size(-1) != 1024:
            projection_layer = nn.Linear(audio_tokens.size(-1), 1024).to(device)
            audio_tokens = projection_layer(audio_tokens)

        # Asegurarse de que el número de tokens sea igual a `self.num_tokens`
        if audio_tokens.size(1) > self.num_tokens:
            indices = torch.randperm(audio_tokens.size(1))[:self.num_tokens]
            audio_tokens = audio_tokens[:, indices, :]
        elif audio_tokens.size(1) < self.num_tokens:
            pad_size = self.num_tokens - audio_tokens.size(1)
            padding = torch.zeros(audio_tokens.size(0), pad_size, audio_tokens.size(2), device=device)
            audio_tokens = torch.cat((audio_tokens, padding), dim=1)

        return audio_tokens

    def process_text(self, text_input, model):
        if isinstance(model, CamembertModel):
            tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
        elif isinstance(model, BertModel):
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        else:
            raise ValueError("Unsupported model type")

        text_tokens = tokenizer(text_input, return_tensors='pt', padding=True, truncation=True, max_length=512)
        text_tokens = {key: value.to(device) for key, value in text_tokens.items()}
        output = model(**text_tokens)
        text_tokens = output.last_hidden_state

        # Proyectar a la dimensión correcta si es necesario
        if text_tokens.size(-1) != 1024:
            projection_layer = nn.Linear(text_tokens.size(-1), 1024).to(device)
            text_tokens = projection_layer(text_tokens)

        if text_tokens.size(1) > self.num_tokens:
            indices = torch.randperm(text_tokens.size(1))[:self.num_tokens]
            text_tokens = text_tokens[:, indices, :]
        elif text_tokens.size(1) < self.num_tokens:
            pad_size = self.num_tokens - text_tokens.size(1)
            padding = torch.zeros(text_tokens.size(0), pad_size, text_tokens.size(2), device=device)
            text_tokens = torch.cat((text_tokens, padding), dim=1)

        return text_tokens



    def translate_text(self, text, source_lang='fr', target_lang='en'):
        if source_lang == 'fr' and target_lang == 'en':
            model = self.translator_fr_to_en
            tokenizer = self.tokenizer_fr_to_en
        elif source_lang == 'en' and target_lang == 'fr':
            model = self.translator_en_to_fr
            tokenizer = self.tokenizer_en_to_fr
        else:
            return text  # No se necesita traducción

        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        
        translated = model.generate(**inputs)
        translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
        return translated_text

    def forward(self, audio_path):
        # Transcribir el audio
        audio_transcription = self.transcribe_audio(audio_path)
        
        # Detectar el idioma de la transcripción
        if self.is_audio_in_french(audio_transcription):  
            text_fr = audio_transcription  
            text_en = self.translate_text(text_fr, source_lang='fr', target_lang='en')
        else:
            text_en = audio_transcription  
            text_fr = self.translate_text(text_en, source_lang='en', target_lang='fr')

        # Procesar tokens de texto en francés y en inglés
        text_fr_tokens = self.process_text(text_fr, self.camembert)
        text_en_tokens = self.process_text(text_en, self.bert)

        # Cruzar los tokens de texto
        tokens_text_cross = self.cross_tr_language(text_fr_tokens, text_en_tokens)
        
        # Procesar el audio con el transformador de audio
        audio_tokens = self.process_audio(audio_path)
        tokens_cross = self.cross_tr_speech(tokens_text_cross, audio_tokens)

        x = tokens_cross[:, 0]
        return self.mlp_head(x)
