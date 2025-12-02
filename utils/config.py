import os
import yaml
import random
import torch
import numpy as np
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables (HF_TOKEN)
load_dotenv()

# Default fallback if no argument is provided
DEFAULT_CONFIG_PATH = "config_distilroberta.yaml" 

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@dataclass
class TrainCfg:
    # --- Fields populated from YAML ---
    model_name: str = ""
    use_qlora: bool = False
    max_length: int = 512
    train_batch_size: int = 4
    valid_batch_size: int = 4
    learning_rate: float = 2e-5
    epochs: int = 3

    # --- Static Paths & Auth ---
    seed: int = 42
    train_path: str = "data/lmsys-chatbot-arena/train.csv"
    train_path_extend: str = "data/lmsys-33k-deduplicated.csv"
    test_path: str = "data/lmsys-chatbot-arena/test.csv"
    sub_path: str = "data/ours_submission.csv"
    hf_token: str = os.getenv("HF_TOKEN")
    
    # Device (set automatically)
    device: torch.device = field(init=False)

    def __post_init__(self):
        # 1. Set Hardware
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 2. Load Default Config initially (so imports don't fail)
        if os.path.exists(DEFAULT_CONFIG_PATH):
            self.load_config(DEFAULT_CONFIG_PATH, verbose=False)

    def load_config(self, config_path: str, verbose: bool = True):
        """Loads a YAML file and updates the configuration fields."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)

        # Apply YAML values to this class instance
        self.model_name = config_data.get("model_name", self.model_name)
        self.use_qlora = config_data.get("use_qlora", self.use_qlora)
        self.max_length = config_data.get("max_length", self.max_length)
        self.train_batch_size = config_data.get("train_batch_size", self.train_batch_size)
        self.valid_batch_size = config_data.get("valid_batch_size", self.valid_batch_size)
        self.learning_rate = float(config_data.get("learning_rate", self.learning_rate))
        self.epochs = config_data.get("epochs", self.epochs)

        if verbose:
            print(f"Loaded Config File: [{config_path}]")
            print(f"- Model: {self.model_name}")
            print(f"- QLoRA: {self.use_qlora}")
            print(f"- Batch Size: {self.train_batch_size}")

# Initialize global config
cfg = TrainCfg()