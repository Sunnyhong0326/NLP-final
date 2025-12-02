import argparse
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    BitsAndBytesConfig
)
from sklearn.model_selection import train_test_split
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

# Import from our local modules
from utils.config import cfg, set_seed
from utils.data_utils import LMSYSDataset, compute_metrics, load_and_clean_data

def get_roberta_model(cfg):
    """Load standard model for Full Fine-Tuning."""
    print(f"Loading Standard Model: {cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, token=cfg.hf_token)
    
    # Ensure pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=3,
        token=cfg.hf_token
    )
    
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
        
    return model, tokenizer

def get_qlora_model(cfg):
    """Load Quantized model + LoRA adapters."""
    print(f"Loading QLoRA Model: {cfg.model_name}")
    
    # 4-bit Quantization Config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, token=cfg.hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" 

    # Load Base Model in 4-bit
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        num_labels=3,
        trust_remote_code=True,
        token=cfg.hf_token
    )
    
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False 
    
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # LoRA Config
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def main():
    set_seed(cfg.seed)
    print(f"Mode: {'QLoRA' if cfg.use_qlora else 'Standard Fine-Tuning'}")
    print(f"Device: {cfg.device}")

    # 1. Load Data
    train_df, _ = load_and_clean_data(cfg)
    
    # 2. Split Data
    train_split, val_split = train_test_split(
        train_df,
        test_size=0.1,
        random_state=cfg.seed,
        shuffle=True,
    )

    # 3. Load Model & Tokenizer based on config
    if cfg.use_qlora:
        model, tokenizer = get_qlora_model(cfg)
    else:
        model, tokenizer = get_roberta_model(cfg)

    # 4. Create Datasets
    train_dataset = LMSYSDataset(train_split, tokenizer, cfg.max_length, is_test=False)
    valid_dataset = LMSYSDataset(val_split, tokenizer, cfg.max_length, is_test=False)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(valid_dataset)}")

    # 5. Define Training Arguments
    training_args = TrainingArguments(
        output_dir=f"./outputs/{cfg.model_name.replace('/', '_')}_{'qlora' if cfg.use_qlora else 'fft'}",
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.train_batch_size,
        per_device_eval_batch_size=cfg.valid_batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="log_loss",
        greater_is_better=False,
        logging_steps=50,
        save_total_limit=1,
        report_to="none",
        fp16=True, 
    )

    # QLoRA specific overrides
    if cfg.use_qlora:
        training_args.gradient_checkpointing = True
        training_args.optim = "paged_adamw_8bit" 

    # 6. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    # 7. Train
    print("Starting training...")
    trainer.train()
    print("Training done.")
    
    # 8. Save Model/Adapter
    adapter_save_path = f"./saved_models/{cfg.model_name.replace('/', '_')}_final"
    trainer.save_model(adapter_save_path)
    tokenizer.save_pretrained(adapter_save_path)
    print(f"Model/Adapter saved to: {adapter_save_path}")

if __name__ == "__main__":
    # Argument Parsing
    parser = argparse.ArgumentParser(description="Train LLM or RoBERTa for Chatbot Arena")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config_gemma_qlora.yaml", 
        help="Path to the YAML configuration file"
    )
    args = parser.parse_args()
    
    # Reload Config based on Argument
    cfg.load_config(args.config)
    
    main()