import os, glob, argparse, random, sys
from pathlib import Path
import subprocess

def create_dir(save_path):
    """ Test if the directory exists and, if it does not, it creates it """
    path = Path(save_path)
    if not path.exists():
        try:
            path.mkdir(parents=True)
        except FileExistsError:
            sys.exit(3)

def filter_txt_files(txt_fs, json_fs, txt_dir):
    """ Returns the txt files that were still not converted to json files """
    filtered_txt_fs = []
    for txt_f in txt_fs:
        has_json = False
        splits = txt_f.split(txt_dir + "/")
        splits = splits[1].split("/")
        txt_name = splits[-1]
        f_name = txt_name[:-4]
        for json_f in json_fs:
            if f_name in json_f:
                has_json = True
                break
        if not has_json:
            filtered_txt_fs.append(txt_f)
    random.shuffle(filtered_txt_fs)
    return filtered_txt_fs

def filter_script_improv(file_list, scripted):
    """
        Returns the list with only the scripted
        files if scripted=True or with only the improvised
        files if scripted=False
    """
    filtered_list = []
    for path in file_list:
        if scripted and "_script" in str(path):
            filtered_list.append(path)
        elif not scripted and "_impro" in str(path):
            filtered_list.append(path)
    return filtered_list

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--speaker", type=str, required=True, help="Speaker id (e.g., M1i, M1s, F2i, F2s)")
    ap.add_argument("--txt_dir", type=str, required=True, help="Directory for the text files")
    ap.add_argument("--audio_dir", type=str, required=True, help="Directory for the speech files")
    config = ap.parse_args()

    speaker = config.speaker
    txt_dir = config.txt_dir
    audio_dir = config.audio_dir

    allowed_speakers = {
        "M1i", "M1s", "F1i", "F1s", "M2i", "M2s", "F2i", "F2s",
        "M3i", "M3s", "F3i", "F3s", "M4i", "M4s", "F4i", "F4s",
        "M5i", "M5s", "F5i", "F5s"
    }

    if speaker not in allowed_speakers:
        sys.exit(0)

    print(f"Executing alignment for speaker: {speaker}")
    output_main = "align_results"

    gender = "Female" if "F" in speaker else "Male"
    session = f"Session{speaker[1]}"
    scripted = "s" in speaker

    txt_fs = glob.glob(f"{txt_dir}/{session}/{gender}/*.txt")
    json_fs = glob.glob(f"{output_main}/{session}/{gender}/*.json")
    txt_fs = filter_script_improv(file_list=txt_fs, scripted=scripted)
    if json_fs:
        json_fs = filter_script_improv(file_list=json_fs, scripted=scripted)

    filtered_txt_fs = txt_fs
    if json_fs:
        filtered_txt_fs = filter_txt_files(txt_fs=txt_fs, json_fs=json_fs, txt_dir=txt_dir)
        if not (len(filtered_txt_fs) == (len(txt_fs) - len(json_fs))):
            sys.exit(1)

    for txt_f in filtered_txt_fs:
        splits = txt_f.split(f"{txt_dir}/")
        splits = splits[1].split("/")
        folders = splits[:-1]
        folder_name = "/".join(folders)
        txt_name = splits[-1]
        f_name = txt_name[:-4]
        audio_f = f"{audio_dir}/{folder_name}/{f_name}.wav"
        create_dir(f"{output_main}/{folder_name}")
        output_f = f"{output_main}/{folder_name}/{f_name}.json"

        audio_path = Path(audio_f)
        if audio_path.exists():
            command = ["python3", "align.py", "-o", output_f, "--nthreads", "12", audio_f, txt_f]
            subprocess.run(command, check=True)
