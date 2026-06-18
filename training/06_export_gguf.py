#!/usr/bin/env python3
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "/srv/llama/training/output/lora-adapter",
    max_seq_length=4096, load_in_4bit=True,
)
model.save_pretrained_gguf(
    "/srv/llama/models/qwen3-8b-finance", tokenizer, quantization_method="q4_k_m",
)
print("Done ? /srv/llama/models/qwen3-8b-finance/")