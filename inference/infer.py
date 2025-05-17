import torch
import numpy as np
from TTS.utils.synthesizer import Synthesizer
from TTS.utils.manage import ModelManager
from scipy.io.wavfile import write
from pathlib import Path
import logging
import os
import traceback

class TTSInfer:
    def __init__(self, config):
        requested_device = config.get('device', 'auto')
        valid_devices = ['cpu', 'cuda']

        if requested_device == 'auto' or requested_device is None:
            if torch.cuda.is_available(): self.device = 'cuda'
            else: self.device = 'cpu'
        elif requested_device in valid_devices:
            if requested_device == 'cuda' and not torch.cuda.is_available():
                self.device = 'cpu'; print("WARNING: CUDA requested but not available, using CPU.")
            else: self.device = requested_device
        else:
            print(f"WARNING: Invalid device '{requested_device}'. Auto-detecting.")
            if torch.cuda.is_available(): self.device = 'cuda'
            else: self.device = 'cpu'
        print(f"INFO: Using device: '{self.device}'")

        self.model_name = config.get('model_name', 'tts_models/en/vctk/vits')
        print(f"INFO: Preparing to load TTS model: '{self.model_name}'")

        self.synthesizer = None
        self.speakers = None
        self.manager = None

        try:
            print(f"INFO: Initializing Synthesizer for model '{self.model_name}' on device '{self.device}'...")
            self.manager = ModelManager(models_file=None, progress_bar=True, verbose=False)
            model_path, config_path, model_item = self.manager.download_model(self.model_name)

            if model_path is None or config_path is None:
                raise RuntimeError(f"Could not download or find model files for {self.model_name}")

            self.synthesizer = Synthesizer(
                tts_checkpoint=model_path,
                tts_config_path=config_path,
                use_cuda=(self.device == 'cuda'),
            )
            print("INFO: Synthesizer initialized successfully.")

            if hasattr(self.synthesizer, 'tts_model') and hasattr(self.synthesizer.tts_model, 'speaker_manager') and self.synthesizer.tts_model.speaker_manager is not None:
                 if hasattr(self.synthesizer.tts_model.speaker_manager, 'speaker_names') and isinstance(self.synthesizer.tts_model.speaker_manager.speaker_names, list):
                     self.speakers = self.synthesizer.tts_model.speaker_manager.speaker_names
                     print(f"INFO: Found {len(self.speakers)} speakers via SpeakerManager.")
                 else:
                     print("WARNING: SpeakerManager found, but 'speaker_names' attribute is missing or not a list.")
            elif hasattr(self.synthesizer, 'speakers') and self.synthesizer.speakers and isinstance(self.synthesizer.speakers, list):
                 self.speakers = self.synthesizer.speakers
                 print(f"INFO: Found {len(self.speakers)} speakers via synthesizer.speakers attribute.")

            if self.speakers:
                 print(f"INFO: Available speakers (first 10): {self.speakers[:10]}...")
            else:
                 print("WARNING: Could not retrieve speaker list from synthesizer.")

        except Exception as e:
            print(f"ERROR: Failed to initialize Synthesizer: {e}")
            traceback.print_exc()
            self.synthesizer = None
            self.speakers = None
            raise e

        self.custom_model = self._load_model(config.get('model_path'))
        if self.custom_model:
            print("WARNING: Custom model loaded, but direct integration with Synthesizer is not implemented in this script.")
        self.speaker_embedding_data = None


    def _load_model(self, model_path):
        if model_path:
            model_path_obj = Path(model_path)
            if model_path_obj.exists() and model_path_obj.is_file():
                try:
                    print(f"INFO: Loading custom model from: {model_path}")
                    
                    print("INFO: Custom model loaded (placeholder).")
                    
                except Exception as e: print(f"ERROR: Failed to load custom model from {model_path}: {e}")
            else: print(f"WARNING: Custom model path not found: {model_path}")
        return None

    def synthesize(self, text, speaker_id=None, **kwargs):
        if not self.synthesizer:
             print("ERROR: Synthesizer not initialized. Cannot synthesize.")
             return None

        params_to_pass = {}
        speaker_name_to_use = None
        parameter_used = None

        if speaker_id is not None:
            print(f"INFO: Validating speaker name: '{speaker_id}'")
            if self.speakers and isinstance(self.speakers, list):
                if speaker_id in self.speakers:
                    speaker_name_to_use = speaker_id
                    parameter_used = 'speaker_name'
                    print(f"INFO: Validated speaker name '{speaker_name_to_use}' found in model list.")
                else:
                    print(f"ERROR: Speaker name '{speaker_id}' not found in model's speaker list: {self.speakers}")
            else:
                print(f"WARNING: Cannot validate speaker name. Model speaker list missing or invalid.")

        try:
            if speaker_name_to_use is not None:
                 print(f"INFO: Starting synthesis for text: '{text[:50]}...' using speaker_name: '{speaker_name_to_use}'")
                 wav = self.synthesizer.tts(text, speaker_name=speaker_name_to_use, language_name=None, speaker_idx=None, **params_to_pass)
                 parameter_used_log = f"speaker_name='{speaker_name_to_use}'"
            else:
                 print(f"INFO: Starting synthesis for text: '{text[:50]}...' using default voice (no speaker specified).")
                 wav = self.synthesizer.tts(text, speaker_name=None, language_name=None, speaker_idx=None, **params_to_pass)
                 parameter_used_log = "None"

            print(f"INFO: Synthesizer.tts() successful. Output type: {type(wav)}")

            processed_audio = None
            if isinstance(wav, list):
                print("INFO: Output is list, converting to float32 NumPy array.")
                processed_audio = np.array(wav, dtype=np.float32)
                max_abs_val = np.max(np.abs(processed_audio))
                if max_abs_val > 1.0:
                     print(f"INFO: Scaling int samples (max abs: {max_abs_val}) to float range.")
                     processed_audio = processed_audio / 32768.0
                print(f"INFO: Converted list to array. Shape: {processed_audio.shape}")
            elif isinstance(wav, np.ndarray):
                print("INFO: Output is NumPy array.")
                processed_audio = wav.astype(np.float32)
            else:
                print(f"ERROR: Unexpected output type from Synthesizer.tts(): {type(wav)}")
                return None

            if not isinstance(processed_audio, np.ndarray): print(f"ERROR: Failed process output. Got {type(processed_audio)}"); return None
            print(f"INFO: Processed audio data. Shape: {processed_audio.shape}, dtype: {processed_audio.dtype}")
            if processed_audio.ndim > 1: processed_audio = processed_audio.squeeze()
            if processed_audio.ndim == 0: print("ERROR: Audio is scalar."); return None
            if processed_audio.ndim != 1: print(f"ERROR: Audio not 1D. Shape: {processed_audio.shape}"); return None
            if processed_audio.size == 0: print("ERROR: Audio empty."); return None
            print(f"INFO: Final audio ready. Shape: {processed_audio.shape}, dtype: {processed_audio.dtype}")
            return processed_audio

        except ValueError as e:
             if "need to define either" in str(e) or "speaker_idx must be specified" in str(e) or "speaker_wav must be specified" in str(e):
                  print(f"ERROR: TTS synthesis failed. The multi-speaker model requires speaker info. Attempted with parameter '{parameter_used_log}' but failed.")
             else:
                  print(f"ERROR: TTS synthesis failed with ValueError: {e}")
                  traceback.print_exc()
             return None
        except Exception as e:
            print(f"ERROR: TTS synthesis failed with unexpected error: {e}")
            traceback.print_exc()
            return None


    def save_to_wav(self, audio, output_path, sample_rate=None):
        if audio is None: print("ERROR: Cannot save None audio to WAV."); return
        if not isinstance(audio, np.ndarray): print(f"ERROR: Audio data not np.ndarray, got {type(audio)}"); return
        if audio.ndim != 1:
             print(f"INFO: Audio not 1D (shape: {audio.shape}), squeezing.")
             audio = audio.squeeze()
             if audio.ndim == 0: print(f"ERROR: Audio became scalar. Cannot save."); return
             if audio.ndim != 1: print(f"ERROR: Cannot reduce audio to 1D. Shape: {audio.shape}"); return
        if audio.size == 0: print("ERROR: Cannot save empty audio array."); return

        sr_to_use = sample_rate
        if sr_to_use is None:
            sr_found = None
            if self.synthesizer:
                 if hasattr(self.synthesizer, 'output_sample_rate'): sr_found = self.synthesizer.output_sample_rate
                 elif hasattr(self.synthesizer, 'tts_config') and hasattr(self.synthesizer.tts_config, 'audio') and 'sample_rate' in self.synthesizer.tts_config.audio: sr_found = self.synthesizer.tts_config.audio['sample_rate']

            if sr_found: sr_to_use = sr_found; print(f"INFO: Using sample rate from synthesizer: {sr_to_use}")
            else: sr_to_use = 22050; print(f"WARNING: Cannot get sample rate from synthesizer. Using default: {sr_to_use}")

        try:
            audio_to_save = None
            if np.issubdtype(audio.dtype, np.floating):
                max_val = np.max(np.abs(audio)); audio_normalized = audio
                if max_val > 1e-6:
                    if max_val > 1.0: print(f"WARNING: Max abs audio {max_val:.4f} > 1.0. Normalizing."); audio_normalized = audio / max_val
                audio_normalized = np.clip(audio_normalized, -1.0, 1.0); audio_to_save = (audio_normalized * 32767).astype(np.int16)
            elif np.issubdtype(audio.dtype, np.int16): audio_to_save = audio
            elif np.issubdtype(audio.dtype, np.integer): print(f"WARNING: Int type {audio.dtype} not int16."); audio_to_save = audio.astype(np.int16)
            else: print(f"ERROR: Unsupported dtype {audio.dtype}"); return

            output_path_obj = Path(output_path); output_path_obj.parent.mkdir(parents=True, exist_ok=True)
            write(str(output_path_obj), int(sr_to_use), audio_to_save)
            print(f"INFO: Audio saved successfully to: {output_path}")
        except Exception as e: print(f"ERROR: Failed to save audio to {output_path}: {e}"); traceback.print_exc()