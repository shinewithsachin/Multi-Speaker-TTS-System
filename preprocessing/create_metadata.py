import os
import csv

def create_metadata(dataset_path, output_csv):
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['filename', 'speaker_id'])
        for speaker_id in os.listdir(dataset_path):
            speaker_dir = os.path.join(dataset_path, speaker_id)
            if os.path.isdir(speaker_dir):
                for audio_file in os.listdir(speaker_dir):
                    if audio_file.endswith('.wav'):
                        writer.writerow([os.path.join(speaker_dir, audio_file), speaker_id])

if __name__ == "__main__":
    dataset_path = "data/speaker_dataset"
    output_csv = "data/metadata.csv"
    create_metadata(dataset_path, output_csv)
