"""
train.py
========
QLoRA fine-tuning of StarCoder2-7B on the PyTorch instruction dataset.
Run this inside Google Colab with a GPU runtime.
"""

import os
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# ── Reproducibility ────────────────────────────────────────────────────────
set_seed(42)

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path("/content/pytorch-code-assistant")
DATA_PATH = ROOT / "data" / "processed" / "pytorch_dataset.jsonl"
OUTPUT_DIR = "/content/drive/MyDrive/pytorch-code-assistant/checkpoints"
MERGED_DIR = "/content/drive/MyDrive/pytorch-code-assistant/merged"

# ── Config ─────────────────────────────────────────────────────────────────
MODEL_NAME    = "bigcode/starcoder2-7b"
LORA_R        = 64
LORA_ALPHA    = 128
LORA_DROPOUT  = 0.05
MAX_SEQ_LEN   = 1024
BATCH_SIZE    = 2
GRAD_ACCUM    = 8        # effective batch = 16
EPOCHS        = 3
LR            = 2e-4

INSTRUCTION_TEMPLATE = "### Instruction:"
RESPONSE_TEMPLATE    = "### Response:"

# ── Prompt formatter ───────────────────────────────────────────────────────
def format_prompt(example):
    return {
        "text": (
            f"{INSTRUCTION_TEMPLATE}\n"
            f"{example['instruction'].strip()}\n\n"
            f"{RESPONSE_TEMPLATE}\n"
            f"{example['output'].strip()}"
        )
    }

# ── Load dataset ───────────────────────────────────────────────────────────
def load_data():
    ds = load_dataset("json", data_files=str(DATA_PATH), split="train")
    ds = ds.map(format_prompt, remove_columns=ds.column_names)
    split = ds.train_test_split(test_size=0.1, seed=42)
    print(f"Train: {len(split['train'])} | Val: {len(split['test'])}")
    return split["train"], split["test"]

# ── Load model ─────────────────────────────────────────────────────────────
def load_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        use_cache=False,
    )
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

# ── Attach LoRA ────────────────────────────────────────────────────────────
def attach_lora(model):
    config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    return model

# ── Train ──────────────────────────────────────────────────────────────────
def train():
    # Check GPU
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU found. Please enable GPU in Colab: Runtime > Change runtime type > T4 GPU")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    train_ds, val_ds = load_data()
    model, tokenizer = load_model()
    model = attach_lora(model)

    # Loss only on response tokens
    response_ids = tokenizer.encode(RESPONSE_TEMPLATE, add_special_tokens=False)
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_ids,
        tokenizer=tokenizer,
    )

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        bf16=True,
        max_grad_norm=0.3,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to="none",
        group_by_length=True,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        data_collator=collator,
        args=args,
        packing=False,
    )

    print("\nStarting training...")
    trainer.train()

    # Save final adapter
    adapter_path = os.path.join(OUTPUT_DIR, "final_adapter")
    trainer.save_model(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"\nAdapter saved → {adapter_path}")

if __name__ == "__main__":
    train()