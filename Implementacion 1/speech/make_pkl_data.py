import os, sys
import numpy as np
import pandas as pd
import gc
from pathlib import Path
from os.path import dirname, join, abspath

sys.path.insert(0, abspath(join(dirname(__file__), '..')))
import speech.make_data_helper as helper

# how many utterances to average to compute the speaker id embedding
num_uttrs = 100
batch_size_to_save = 10  # Tamaño del lote para guardar en CSV

def save_data_to_csv(utterances_batch, csv_filename):
    """Guardar el batch de utterances en un archivo CSV de manera incremental."""
    df = pd.DataFrame(utterances_batch, columns=[
        "utterance_file", "phones_and_durations", "main_phones", "emotion_class", 
        "word_seq", "word_intervals", "gender_class", "speaker_embedding", "speaker_id"
    ])
    # Guardar en CSV, añadiendo las filas al archivo existente
    df.to_csv(csv_filename, mode='a', header=not os.path.exists(csv_filename), index=False)
    print(f"Guardado batch en CSV temporal: {csv_filename}")

def convert_csv_to_pkl(csv_filename, pkl_filename):
    """Convertir el archivo CSV final en un archivo PKL."""
    df = pd.read_csv(csv_filename)
    df.to_pickle(pkl_filename)
    print(f"Archivo PKL final guardado en: {pkl_filename}")

def get_data(config):
    """ Main loop to get the data for the IEMOCAP dataset """

    dir_name, subdir_list, _ = next(os.walk(config.root_dir))
    helper.logger("info", "[INFO] Found directory: " + str(dir_name))

    all_success = 0
    all_unsuccess = 0

    # Obtener el nombre del fold desde config.spmel_dir
    fold_name = Path(config.spmel_dir).parts[-2]  # Extraer el nombre del fold, por ejemplo, 'fold1'

    # Crear nombres de archivo basados en el fold y el modo (train/test)
    base_name = f"{config.mode}_{fold_name}"  # Por ejemplo: 'train_fold1' o 'test_fold1'
    csv_filename = f"{base_name}.csv"
    pkl_filename = f"{base_name}.pkl"

    # Procesamiento de sesiones y parlantes
    for session in sorted(subdir_list):
        helper.logger("info", "[INFO] Processing session: " + str(session))
        session_dir, spks_gender, _ = next(os.walk(os.path.join(dir_name, session)))
        
        # Para cada speaker (Male y Female)
        for gender in spks_gender:
            phone_seq_file_path = config.phone_dir + "/" + session + "/" + gender
            pathlib_phone_path = Path(phone_seq_file_path)
            
            if pathlib_phone_path.exists():
                spk_id_to_append = session + "_" + gender
                print("spk: ", spk_id_to_append)
                _, _, file_list = next(os.walk(os.path.join(session_dir, gender)))

                assert len(file_list) >= num_uttrs
                helper.logger("info", "[INFO] len(file_list): " + str(len(file_list)))

                idx_uttrs = np.random.choice(len(file_list), size=num_uttrs, replace=False)
                helper.logger("info", "[INFO] idx_uttrs: " + str(idx_uttrs))
                embs = helper.get_speaker_embeddings(
                    dir_name=config.wav_dir,
                    speaker=session + "/" + gender,
                    file_list=file_list,
                    idx_uttrs=idx_uttrs,
                    num_uttrs=num_uttrs
                )
                
                spk_emb_to_append = np.mean(embs, axis=0)
                helper.logger("info", "[INFO] Speaker embedding mean shape: " + str(spk_emb_to_append.shape))

                utterances_batch = []
                content_gen = helper.get_content_list(file_list=file_list, speaker_dir=session + "/" + gender, config=config)

                for element in content_gen:
                    tmp_element = element.copy()
                    gender_class = 0 if gender == "Female" else 1
                    tmp_element.append(gender_class)
                    tmp_element.append(spk_emb_to_append)
                    tmp_element.append(spk_id_to_append)
                    utterances_batch.append(tmp_element)

                    # Guardar en CSV si se alcanza el tamaño del lote
                    if len(utterances_batch) >= batch_size_to_save:
                        save_data_to_csv(utterances_batch, csv_filename)
                        del utterances_batch
                        utterances_batch = []
                        gc.collect()

                # Guardar el resto si el lote no estaba lleno
                if len(utterances_batch) > 0:
                    save_data_to_csv(utterances_batch, csv_filename)
                    del utterances_batch
                    gc.collect()

            else:
                helper.logger("warning", "[WARNING] The speaker does not have any phone sequence")

    helper.logger("info", "[INFO] Total number of successful spec transformations: " + str(all_success))
    helper.logger("info", "[INFO] Total number of unsuccessful spec transformations: " + str(all_unsuccess))

    # Convertir el archivo CSV a PKL al final
    convert_csv_to_pkl(csv_filename, pkl_filename)


def main():
    config = helper.get_config()
    get_data(config)

if __name__ == '__main__':
    main()
