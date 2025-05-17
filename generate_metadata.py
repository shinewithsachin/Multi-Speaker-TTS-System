import os
import csv

base_path = 'dataset/speaker_dataset/50_speakers_audio_data'
metadata_path = 'dataset/metadata.csv'

with open(metadata_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['filename', 'speaker_id'])

    for speaker_folder in os.listdir(base_path):
        speaker_path = os.path.join(base_path, speaker_folder)
        if os.path.isdir(speaker_path):
            for file_name in os.listdir(speaker_path):
                if file_name.endswith('.wav'):
                    full_path = os.path.join(speaker_path, file_name)
                    writer.writerow([full_path, speaker_folder])
