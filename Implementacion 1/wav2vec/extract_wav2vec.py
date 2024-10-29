"""
    Name: eval_and_extract_wav2vec.py

    Description: Script to extract wav2vec embeddings from the model's
                 last hidden layer and store these embeddings as npy
"""

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from datasets import load_dataset
import soundfile as sf
import torch
import sys, os
import numpy as np
from spectrogram_helpers import get_spec
import tempfile

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')
device=torch.device('cpu')
print("Device: ", device)

MODEL_TYPE = "facebook/wav2vec2-large-lv60"

processor = Wav2Vec2Processor.from_pretrained(MODEL_TYPE)

model = Wav2Vec2ForCTC.from_pretrained(MODEL_TYPE).to(device)


temp_audio_dir = tempfile.mkdtemp(dir="/media/chega/Nuevo vol/PFC1/multimodal_emotion_recognition/wav2vec/tmp")


def map_to_array(batch):
    # Read the audio file
    speech, _ = sf.read(batch["file"])
    remainder = len(speech) % 320
    if remainder != 0:
        padding = 320 - remainder
        speech = np.pad(speech, (0, padding), mode='constant')
    
    # Flatten to ensure 1D input
    speech = speech.flatten()
    print("Speech shape after padding and flattening:", speech.shape)

    # Save processed audio as .npz on disk
    file_name = os.path.basename(batch["file"]).replace(".wav", ".npz")
    temp_file_path = os.path.join(temp_audio_dir, file_name)
    np.savez_compressed(temp_file_path, speech=speech)
    
    # Store the temporary file path instead of data
    batch["temp_speech_file"] = temp_file_path
    return batch


"""     # Define padding for spectrogram alignment
    padding = 160 if len(speech) % 320 >= 80 else 319
    speech = np.pad(speech, padding, mode='constant') """
    
    


  


def save_hidden_states(speech_files, out_dir, hidden_states, speech_file_paths):
    """Save each hidden layer state in separate folders by audio file."""
    for speech_file, speech_path in zip(speech_files, speech_file_paths):
        file_name = os.path.basename(speech_file).replace(".wav", "")
        subdir1 = speech_file.split("/")[-2]
        subdir2 = speech_file.split("/")[-3]
        
        # Create the save path for current audio file layers
        path = os.path.join(out_dir, "all_hidden_states", subdir2, subdir1, file_name)
        os.makedirs(path, exist_ok=True)

        # Save each layer in a compressed .npz file
        for i, layer in enumerate(hidden_states):
            layer_file_path = os.path.join(path, f"layer_{i}.npz")
            np.savez_compressed(layer_file_path, layer=layer.detach().cpu().numpy())
            
            # Clear memory
            del layer
            torch.cuda.empty_cache()
            
        # Remove the temporary audio file after processing
        os.remove(speech_path)


def map_to_pred(batch):
    # Load the processed audio data from the temporary file
    speech = np.load(batch["temp_speech_file"][0])['speech']
    print("Speech shape before processor:", speech.shape)  # For debugging

    # Process the audio data using the Wav2Vec2Processor
    inputs = processor(speech, return_tensors="pt", sampling_rate=16000, padding=True)
    print("Input values shape after processor:", inputs.input_values.shape)  # For debugging

    # Ensure inputs are the correct dimensions: (batch_size, sequence_length)
    input_values = inputs.input_values.squeeze().to(device)  # Squeeze to remove extra dimensions
    attention_mask = inputs.attention_mask.to(device)

    out_dir = "wav2vec/features/large-lv60"

    # Perform inference to get hidden states
    with torch.no_grad():
        output = model(input_values.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0), output_hidden_states=True)

    # Save hidden states
    save_hidden_states(
        speech_files=batch["file"],
        out_dir=out_dir,
        hidden_states=output.hidden_states,
        speech_file_paths=batch["temp_speech_file"]
    )

    # Remove temporary audio file
    os.remove(batch["temp_speech_file"][0])

    ## get the text results from the decoding
    # logits = output.logits
    # predicted_ids = torch.argmax(logits, dim=-1)
    # transcription = tokenizer.batch_decode(predicted_ids)
    # transcription = processor.batch_decode(predicted_ids)
    # batch["transcription"] = transcription

def eval(data_file):
    """Evaluate the model on the librispeech dataset, printing the WER."""
    # Load the dataset
    librispeech_eval = load_dataset("csv", data_files=data_file, column_names=["file","text"], delimiter=";", quoting=3, split="train").select(range(1)) #para test
    librispeech_eval = librispeech_eval.map(map_to_array)

    # Cambiar "speech" a "temp_speech_file" en remove_columns
    librispeech_eval.map(map_to_pred, batched=True, batch_size=1, remove_columns=["temp_speech_file"])


def main():
    # samples that have text transcript
    data_file = "/media/chega/Nuevo vol/PFC1/multimodal_emotion_recognition/wav2vec/utils/dataset.csv"
    eval(data_file)

    # samples that have no text transcript
    no_content_files = "wav2vec/utils/no_content_files.csv"
    eval(no_content_files)

if __name__ == '__main__':
    main()
