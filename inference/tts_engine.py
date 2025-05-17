from .infer import TTSInfer  
from dataclasses import dataclass
from pathlib import Path

@dataclass
class TTSConfig:
    model_name: str = 'tts_models/en/ljspeech/glow-tts'
    model_path: str = None
    embeddings_path: str = None
    device: str = 'auto'

class TTSEngine(TTSInfer):
    """Enhanced interface with quality-of-life improvements"""
    
    def __init__(self, config=None):
        """
        Initialize with either:
        - Path to config file (YAML/JSON)
        - Dictionary of parameters
        - TTSConfig object
        """
        if config is None:
            config = TTSConfig()
        elif isinstance(config, (str, Path)):
            config = self._load_config(config)
        elif isinstance(config, dict):
            config = TTSConfig(**config)
            
        super().__init__(config.__dict__)

    def _load_config(self, config_path):
        """Load configuration from file"""
        
        raise NotImplementedError("Config loading not implemented")