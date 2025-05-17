import os
import librosa
import soundfile as sf
import pandas as pd

def preprocess_audio_from_metadata(metadata_path, audio_base_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    metadata = pd.read_csv(metadata_path)

    for i, row in metadata.iterrows():
        filename, speaker_id = row['filename'], row['speaker_id']
        audio_path = os.path.join(audio_base_path, filename)
        try:
            y, sr = librosa.load(audio_path, sr=22050)
            speaker_folder = os.path.join(output_path, speaker_id)
            os.makedirs(speaker_folder, exist_ok=True)

            output_file = os.path.join(speaker_folder, os.path.basename(filename))
            sf.write(output_file, y, sr)
            print(f"Processed {filename} -> {output_file}")
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")


metadata_path = "data/speaker_dataset/metadata.csv"
audio_base_path = "data/speaker_dataset/50_speakers_audio_data"
output_path = "data/processed"

preprocess_audio_from_metadata(metadata_path, audio_base_path, output_path)
