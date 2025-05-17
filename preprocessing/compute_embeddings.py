import os
import torch
import torchaudio
from tqdm import tqdm
import numpy as np
from speechbrain.inference.speaker import EncoderClassifier

def compute_speaker_embeddings(audio_dir, output_path):
    
    classifier = EncoderClassifier.from_hparams(
        source="D:/6th Sem Project/TexttoSpeech/pretrained_models/spkrec",
        run_opts={"device": "cpu"}  
    )

    embeddings = {}

    for root, _, files in os.walk(audio_dir):
        for file in tqdm(files):
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                signal, fs = torchaudio.load(file_path)
                embedding = classifier.encode_batch(signal).squeeze().detach().cpu().numpy()
                embeddings[file_path] = embedding

    
    np.save(output_path, embeddings)
    print(f"✅ Saved {len(embeddings)} speaker embeddings to {output_path}")


audio_dir = "data/processed"
output_path = "data/speaker_embeddings.npy"

compute_speaker_embeddings(audio_dir, output_path)
