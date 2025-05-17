import torch

class Tacotron2TTS:
    def __init__(self, tacotron_path="models/tacotron2.pth", waveglow_path="models/waveglow.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tacotron = self.load_model(tacotron_path)
        self.waveglow = self.load_model(waveglow_path)

    def load_model(self, model_path):
        try:
            model = torch.load(model_path, map_location=self.device)
            model.eval()
            print(f"✅ Model Loaded: {model_path}")
            return model
        except FileNotFoundError:
            print(f"❌ ERROR: Model file '{model_path}' not found.")
            return None
        except Exception as e:
            print(f"❌ ERROR: Unable to load model. Details: {e}")
            return None
