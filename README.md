# 🗣️ Multi-Speaker Text-to-Speech System

## 🎯 Project Summary
This project implements a **natural voice synthesis system** capable of **voice cloning** using **pre-computed speaker embeddings**. Built using **Python**, **PyTorch**, **Coqui TTS**, **SpeechBrain**, and **Streamlit**, it allows users to input text and synthesize speech in the voice of different speakers.

---

## 🧠 Key Features

- 🔊 Text-to-speech synthesis with **Coqui TTS (Glow-TTS)**
- 🧬 Speaker embedding extraction using **SpeechBrain**
- 🗂️ 50-speaker dataset with metadata and preprocessing
- 💻 Streamlit GUI for real-time interaction and audio playback
- 🎯 Supports basic **voice cloning** via speaker embeddings

---

## 🧪 Objectives

- Preprocess audio data: resampling and metadata creation
- Extract speaker embeddings from 50-speaker dataset
- Integrate embeddings into the TTS inference engine
- Build a functional GUI using Streamlit
- Evaluate generated speech for intelligibility and speaker similarity

---

## 🛠️ Project Structure
multi-speaker-tts-system/
│
├── data/
│ └── speaker_dataset/ # Raw audio files + metadata.csv
│
├── preprocessing/
│ ├── create_metadata.py # Metadata generation
│ ├── preprocess_audio.py # Audio resampling (22050Hz)
│ └── compute_embeddings.py # Extracts & stores speaker embeddings
│
├── inference/
│ ├── tts_engine.py # Core TTS engine using Coqui
│ └── infer.py # Wrapper for inference functions
│
├── ui/
│ └── app.py # Streamlit frontend
│
├── requirements.txt # Project dependencies
├── runtime.txt # Python version pinning for deployment
└── main.py # Entry point

---

## 🖼️ User Interface (Streamlit)

- 🔤 Input text
- 👤 Select speaker (from dropdown)
- ▶️ Generate and play audio
- 💾 Download synthesized audio

---

## 🧪 Results

- Functional pipeline validated with listening tests.
- Speech is intelligible and voice cloning reflects speaker characteristics.
- Perceived speaker similarity varies across the dataset.

---

## 🚧 Limitations

- Some speaker clones lack strong perceptual similarity.
- Coqui Glow-TTS model produces moderately natural speech by default.
- Python 3.12 is **not supported** by `TTS`. Use **Python 3.10 or 3.9**.

---

## 🔧 Setup Instructions

### 1. Clone the Repository

git clone https://github.com/your-username/multi-speaker-tts-system.git
cd multi-speaker-tts-system

### 2. Create a Virtual Environment with Python 3.10

conda create -n tts-env python=3.10
conda activate tts-env

### 3. Install Dependencies

pip install -r requirements.txt

### 4. Run the App

streamlit run main.py


## 📚 Credits

Sachin Kumar – Project Developer

IIIT Guwahati – Academic Institution

Libraries Used:

Coqui TTS

SpeechBrain

Streamlit

## 📷 Screenshots



## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


