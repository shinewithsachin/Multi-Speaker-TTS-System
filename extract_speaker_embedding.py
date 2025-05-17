import os
import torch
import torchaudio
from transformers import SpeechT5Processor, SpeechT5SpeakerEncoder

# Paths
wav_path = "dataset/speaker_dataset/50_speakers_audio_data/Speaker_0000/Speaker_0000_00000.wav"
embedding_save_path = "speaker_embedding.pt"

# Load processor and model
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
speaker_encoder = SpeechT5SpeakerEncoder.from_pretrained("microsoft/speecht5_speaker_encoder")

# Load and resample audio
speech_waveform, sampling_rate = torchaudio.load(wav_path)
if sampling_rate != 16000:
    resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
    speech_waveform = resampler(speech_waveform)

# Get speaker embedding
with torch.no_grad():
    speaker_embedding = speaker_encoder(speech_waveform[0].unsqueeze(0))

# Save embedding
torch.save(speaker_embedding, embedding_save_path)
print(f"✅ Speaker embedding saved to: {embedding_save_path}")
