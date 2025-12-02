import ast
import re
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.metrics import log_loss
from utils.config import cfg

def strip_surrogates(text: str) -> str:
    """Remove characters that cannot be encoded in UTF-8 (surrogates etc)."""
    if not isinstance(text, str):
        text = str(text)
    return text.encode("utf-8", "replace").decode("utf-8")

def strip_control_chars(s: str) -> str:
    """Remove ASCII control chars except \n and \t."""
    return re.sub(r"[\x00-\x08\x0B-\x1F\x7F]", "", s)

def safe_literal_list(text: str):
    """Safely parse stringified list like '["a", "b"]', else return None."""
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            return None
    return None

def clean_text(x):
    """Complete cleaning pipeline."""
    # None / NaN
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""

    if isinstance(x, list):
        return " ".join(strip_surrogates(strip_control_chars(str(t))) for t in x)

    s = str(x).strip()

    parsed_list = safe_literal_list(s)
    if parsed_list is not None:
        return " ".join(strip_surrogates(strip_control_chars(str(t))) for t in parsed_list)

    # Strip unicode surrogates and control characters
    s = strip_surrogates(s)
    s = strip_control_chars(s)

    return s

def load_and_clean_data(cfg):
    print("Loading data...")
    train_df = pd.read_csv(cfg.train_path)
    train_ext_df = pd.read_csv(cfg.train_path_extend)
    test_df = pd.read_csv(cfg.test_path)

    print("Original train shape:", train_df.shape)
    print("Original extended train shape :", train_ext_df.shape)
    
    dfs = {"train": train_df, "test": test_df, "train_ext": train_ext_df}

    for df_name, df in dfs.items():
        # Ensure critical columns exist before dropping NA
        cols = ["prompt", "response_a", "response_b"]
        df.dropna(subset=cols, inplace=True)
        
        for col in cols:
            df[col] = df[col].apply(clean_text)

        mask_empty = (
            df["prompt"].str.strip().eq("") |
            df["response_a"].str.strip().eq("") |
            df["response_b"].str.strip().eq("")
        )
        to_drop = mask_empty.sum()
        if to_drop > 0:
            print(f"{df_name}: Dropping {to_drop} empty rows")
            df.drop(df[mask_empty].index, inplace=True)

        df.reset_index(drop=True, inplace=True)
        print(f"After cleaning, {df_name} shape:", df.shape)

    print("Concatenating main + extended train...")
    combined_train_df = pd.concat([train_df, train_ext_df], ignore_index=True)
    print("Final combined train shape:", combined_train_df.shape)
    
    return combined_train_df, test_df

# --- Dataset Class ---
class LMSYSDataset(Dataset):
    def __init__(self, df, tokenizer, max_length, is_test=False):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test
        
        # Convert to records for speed
        self.data = self.df[["prompt", "response_a", "response_b"]].to_dict("records")

        # Pre-compute labels
        if not self.is_test:
            self.labels = []
            for _, row in self.df.iterrows():
                if row["winner_model_a"] == 1:
                    label = 0
                elif row["winner_model_b"] == 1:
                    label = 1
                else:
                    label = 2
                self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        prompt = row["prompt"]
        response_a = row["response_a"]
        response_b = row["response_b"]

        if not self.is_test:
            if row["winner_model_a"] == 1:
                label = 0   # A wins
            elif row["winner_model_b"] == 1:
                label = 1   # B wins
            else:
                label = 2   # tie
        else:
            label = -1  # unused

        # ----- Data augmentation: random swap A/B -----\n"
        if not self.is_test:
            # 50% chance to swap
            if np.random.random() < 0.5:
                response_a, response_b = response_b, response_a
                if label == 0:
                    label = 1
                elif label == 1:
                    label = 0
        
        encoded = self.tokenizer(
            prompt,
            response_a + self.tokenizer.eos_token + response_b,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }

        if not self.is_test:
            item["labels"] = torch.tensor(label, dtype=torch.long)

        return item

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = F.softmax(torch.tensor(logits), dim=-1).numpy()

    num_classes = probs.shape[1]
    labels = np.array(labels, dtype=int)
    labels_oh = np.eye(num_classes)[labels]

    return {"log_loss": log_loss(labels_oh, probs)}