#!/usr/bin/env python3
"""Fine-tune Qwen3-8B on finance Q&A data using Unsloth + QLoRA."""

from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import torch

BASE_MODEL = "unsloth/Qwen3-8B-unsloth-bnb-4bit"
MAX_SEQ_LENGTH = 4096
OUTPUT_DIR = "/srv/llama/training/output"
TRAIN_FILE = "/srv/llama/training/dataset/train.jsonl"
VAL_FILE = "/srv/llama/training/dataset/val.jsonl"

print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    BASE_MODEL, max_seq_length=MAX_SEQ_LENGTH, load_in_4bit=True, dtype=None,
)

print("Applying LoRA...")
model = FastLanguageModel.get_peft_model(
    model, r=32, lora_alpha=32, lora_dropout=0.0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    bias="none", use_gradient_checkpointing="unsloth", random_state=42,
)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

dataset = load_dataset("json", data_files={"train": TRAIN_FILE, "validation": VAL_FILE})
print(f"Train: {len(dataset['train'])} | Val: {len(dataset['validation'])}")


def formatting_func(examples):
    return {"text": [
        tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
        for m in examples["messages"]
    ]}


trainer = SFTTrainer(
    model=model, tokenizer=tokenizer,
    train_dataset=dataset["train"], eval_dataset=dataset["validation"],
    args=SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        eval_strategy="steps", eval_steps=100,
        save_strategy="steps", save_steps=200, save_total_limit=3,
        max_seq_length=MAX_SEQ_LENGTH,
        seed=42, report_to="none",
    ),
    formatting_func=formatting_func,
)

stats = trainer.train()
print(f"\nDone! Steps: {stats.global_step} | Loss: {stats.training_loss:.4f}")

model.save_pretrained(f"{OUTPUT_DIR}/lora-adapter")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/lora-adapter")
print(f"Saved ? {OUTPUT_DIR}/lora-adapter")