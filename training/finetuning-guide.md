# Fine-Tuning Local LLMs — Complete Guide (v2)

**Target setup:** RTX 4090 (24GB) · AMD 7950X · 64GB DDR5 · Arch Linux
**Base model:** Qwen3-8B (for fine-tuning) · Qwen3.5-27B (for Q&A generation)
**Goal:** Fine-tune on finance training manuals across multiple courses (CIIA, CIWM, AWM, etc.)

---

## Table of Contents

1. Folder structure & PDF organization
2. Environment setup
3. PDF text extraction (course/module-aware)
4. Data cleaning
5. Q&A generation with local model
6. Training data formatting
7. Fine-tuning with Unsloth (QLoRA)
8. Export to GGUF
9. Deploy with llama-server
10. Evaluation
11. Tips and troubleshooting

---

## 1. Folder Structure & PDF Organization

### Recommended structure

```
/srv/llama/training/
├── pdfs/
│   ├── CIIA/
│   │   ├── accounting/
│   │   │   ├── CIIA_ACC_1.pdf
│   │   │   ├── CIIA_ACC_2.pdf
│   │   │   └── ...
│   │   ├── economics/
│   │   │   ├── CIIA_ECO_1.pdf
│   │   │   └── ...
│   │   ├── equity_analysis/
│   │   ├── fixed_income/
│   │   ├── derivatives/
│   │   ├── portfolio_management/
│   │   └── regulation/
│   ├── CIWM/
│   │   ├── client_advisory/
│   │   ├── compliance/
│   │   └── ...
│   ├── AWM/
│   │   └── ...
│   └── SHARED/
│       └── corporate_finance/
│           └── CIIA_ACC_6_PARTLY_CIWM_CF.pdf
├── extracted/
├── cleaned/
├── qa-pairs/
├── dataset/
└── output/
```

### Rules

- **First level** = course (`CIIA`, `CIWM`, `AWM`, ...)
- **Second level** = module (use snake_case: `accounting`, `fixed_income`, `portfolio_management`, ...)
- **SHARED/** = for manuals that belong to multiple courses. Tag them with all relevant courses.
- PDF filenames don't matter for tagging — the folder path determines the course and module.

### Migration from your current structure

Your current layout `1- CIIA_ILPIP\2026\1_Accounting - linked` needs flattening.
Here's a helper script to set up the folder structure:

### setup_folders.sh

```bash
#!/bin/bash
# Create the folder structure. Adjust courses and modules to match yours.

BASE="/srv/llama/training/pdfs"

# CIIA modules
for mod in accounting economics equity_analysis fixed_income derivatives \
           portfolio_management regulation corporate_finance; do
    mkdir -p "$BASE/CIIA/$mod"
done

# CIWM modules
for mod in client_advisory compliance corporate_finance wealth_planning \
           portfolio_management taxation estate_planning; do
    mkdir -p "$BASE/CIWM/$mod"
done

# AWM modules (adjust as needed)
for mod in asset_management risk_management alternative_investments; do
    mkdir -p "$BASE/AWM/$mod"
done

# Shared (cross-course manuals)
mkdir -p "$BASE/SHARED/corporate_finance"

echo "Done. Now move your PDFs into the appropriate folders."
echo "Structure:"
find "$BASE" -type d | sort | sed 's|^|  |'
```

Then manually move (or copy/symlink) your PDFs into the right folders. For the shared manual:

```bash
# Example: a manual used by both CIIA and CIWM
cp "CIIA_ACC_6_PARTLY_CIWM_CF.pdf" /srv/llama/training/pdfs/SHARED/corporate_finance/
```

---

## 2. Environment Setup

```
/srv/llama/
├── llama.cpp/          # built from source (CUDA 89, znver4)
├── llm-tools/          # Python venv
├── models/             # GGUF models + HF cache
└── training/           # fine-tuning workspace
```

### Activate venv and install dependencies

```bash
source /srv/llama/llm-tools/bin/activate
pip install pymupdf sentencepiece protobuf
```

### Verify GPU

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```

---

## 3. PDF Text Extraction (Course/Module-Aware)

This script walks the folder tree and tags every extracted page with course and module metadata.

### extract_text.py

```python
#!/usr/bin/env python3
"""Extract text from PDFs, tagging each page with course and module from folder structure.

Expected layout:
    pdfs/<COURSE>/<module>/<file>.pdf
    pdfs/SHARED/<module>/<file>.pdf   ← tagged with all courses listed in SHARED_TAGS
"""

import json
import fitz  # pymupdf
from pathlib import Path

PDF_DIR = Path("/srv/llama/training/pdfs")
OUTPUT_FILE = Path("/srv/llama/training/extracted/raw_pages.jsonl")

# For PDFs in the SHARED folder, specify which courses they belong to.
# Key = module folder name, Value = list of courses
SHARED_TAGS = {
    "corporate_finance": ["CIIA", "CIWM"],
    # Add more as needed:
    # "risk_management": ["CIIA", "CIWM", "AWM"],
}

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

total_pages = 0
total_docs = 0
skipped = 0


def get_tags(pdf_path: Path) -> list[dict]:
    """Derive course and module tags from the folder path.

    Returns a list of {course, module} dicts.
    Most PDFs return one entry; SHARED PDFs return multiple.
    """
    # Path relative to PDF_DIR, e.g. "CIIA/accounting/CIIA_ACC_1.pdf"
    rel = pdf_path.relative_to(PDF_DIR)
    parts = rel.parts  # ("CIIA", "accounting", "CIIA_ACC_1.pdf")

    if len(parts) < 3:
        # PDF directly in a course folder without module subfolder
        # Treat the course folder as both course and "general" module
        course = parts[0].upper()
        module = "general"
        if course == "SHARED":
            return [{"course": "UNKNOWN", "module": "general"}]
        return [{"course": course, "module": module}]

    course_dir = parts[0]
    module_dir = parts[1]

    if course_dir.upper() == "SHARED":
        # Look up which courses this shared module belongs to
        courses = SHARED_TAGS.get(module_dir, ["UNKNOWN"])
        return [{"course": c, "module": module_dir} for c in courses]
    else:
        return [{"course": course_dir.upper(), "module": module_dir}]


with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for pdf_path in sorted(PDF_DIR.rglob("*.pdf")):
        tags = get_tags(pdf_path)
        print(f"Processing: {pdf_path.relative_to(PDF_DIR)}")
        print(f"  Tags: {tags}")

        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                text = page.get_text("text").strip()
                if len(text) > 50:  # skip near-empty pages
                    # Write one record per course tag (shared PDFs produce multiple)
                    for tag in tags:
                        record = {
                            "course": tag["course"],
                            "module": tag["module"],
                            "source": pdf_path.name,
                            "page": page.number + 1,
                            "text": text,
                        }
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        total_pages += 1
            doc.close()
            total_docs += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            skipped += 1

print(f"\nDone: {total_pages} tagged pages from {total_docs} documents ({skipped} errors)")
print(f"Output: {OUTPUT_FILE}")
```

### Run it

```bash
python /srv/llama/training/extract_text.py
```

### Verify tags

```bash
# Check distribution of courses and modules
python3 -c "
import json
from collections import Counter

courses = Counter()
modules = Counter()

with open('/srv/llama/training/extracted/raw_pages.jsonl') as f:
    for line in f:
        r = json.loads(line)
        courses[r['course']] += 1
        modules[f\"{r['course']}/{r['module']}\"] += 1

print('Pages per course:')
for k, v in courses.most_common():
    print(f'  {k}: {v}')

print()
print('Pages per module:')
for k, v in modules.most_common():
    print(f'  {k}: {v}')
"
```

---

## 4. Data Cleaning

Cleans text and splits into chunks, preserving course/module tags.

### clean_and_chunk.py

```python
#!/usr/bin/env python3
"""Clean extracted text and split into training-ready chunks.
Preserves course and module tags throughout."""

import json
import re
from pathlib import Path

INPUT_FILE = Path("/srv/llama/training/extracted/raw_pages.jsonl")
OUTPUT_FILE = Path("/srv/llama/training/cleaned/chunks.jsonl")

MAX_CHARS = 4000   # ~1000 tokens per chunk
OVERLAP_CHARS = 400


def clean_text(text: str) -> str:
    """Remove common PDF artifacts."""
    # Page numbers
    text = re.sub(r"^\s*\d{1,3}\s*$", "", text, flags=re.MULTILINE)

    # Common footer/header patterns (customize for your manuals)
    text = re.sub(r"(?i)^.*confidential.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"(?i)^.*all rights reserved.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"(?i)^.*AZEK.*Swiss Training Centre.*$", "", text, flags=re.MULTILINE)

    # Fix broken line wraps
    text = re.sub(r"([a-zäöüéèà,])\n([a-zäöüéèà])", r"\1 \2", text)

    # Collapse blank lines and whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


def chunk_text(text: str, metadata: dict) -> list:
    """Split text into overlapping chunks, carrying metadata."""
    if len(text) <= MAX_CHARS:
        return [{**metadata, "text": text}]

    chunks = []
    start = 0
    chunk_idx = 0

    while start < len(text):
        end = start + MAX_CHARS

        if end < len(text):
            break_point = text.rfind("\n\n", start + MAX_CHARS // 2, end)
            if break_point == -1:
                break_point = text.rfind(". ", start + MAX_CHARS // 2, end)
                if break_point != -1:
                    break_point += 2
            if break_point > start:
                end = break_point

        chunk = text[start:end].strip()
        if len(chunk) > 50:
            chunks.append({**metadata, "chunk": chunk_idx, "text": chunk})
            chunk_idx += 1

        start = end - OVERLAP_CHARS

    return chunks


# Process
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
total_chunks = 0

with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

    for line in fin:
        record = json.loads(line)
        cleaned = clean_text(record["text"])
        if cleaned:
            metadata = {
                "course": record["course"],
                "module": record["module"],
                "source": record["source"],
                "page": record["page"],
            }
            for chunk in chunk_text(cleaned, metadata):
                fout.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                total_chunks += 1

print(f"Done: {total_chunks} chunks → {OUTPUT_FILE}")
```

### Run it

```bash
python /srv/llama/training/clean_and_chunk.py
```

---

## 5. Q&A Generation with Local Model

Start llama-server first:

```bash
llama-server \
  -m /srv/llama/models/Qwen3.5-27B-Q4_K_M.gguf \
  --host 127.0.0.1 --port 8080 \
  -ngl 99 -c 8192 --jinja \
  --reasoning-format none \
  --temp 0.7 --top-k 20 --top-p 0.95
```

### generate_qa.py

```python
#!/usr/bin/env python3
"""Generate Q&A pairs from text chunks using local LLM.
Includes course and module context in the prompt for better relevance."""

import json
import time
from pathlib import Path
from openai import OpenAI

INPUT_FILE = Path("/srv/llama/training/cleaned/chunks.jsonl")
OUTPUT_FILE = Path("/srv/llama/training/qa-pairs/qa_raw.jsonl")

client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")

# Module display names for better prompts
MODULE_NAMES = {
    "accounting": "Accounting & Financial Reporting",
    "economics": "Economics",
    "equity_analysis": "Equity Analysis & Valuation",
    "fixed_income": "Fixed Income",
    "derivatives": "Derivatives & Structured Products",
    "portfolio_management": "Portfolio Management",
    "regulation": "Regulation & Compliance",
    "corporate_finance": "Corporate Finance",
    "client_advisory": "Client Advisory",
    "compliance": "Compliance & Legal",
    "wealth_planning": "Wealth Planning",
    "taxation": "Taxation",
    "estate_planning": "Estate Planning",
    "asset_management": "Asset Management",
    "risk_management": "Risk Management",
    "alternative_investments": "Alternative Investments",
}


def get_system_prompt(course: str, module: str) -> str:
    """Build a context-aware system prompt."""
    module_display = MODULE_NAMES.get(module, module.replace("_", " ").title())

    return f"""You are an expert in finance education, specifically for the {course} certification program, module: {module_display}.

Given a text passage from the {course} training manual on {module_display}, generate 3-5 high-quality question-answer pairs that test understanding of the key concepts.

Rules:
- Questions should vary: definitional, conceptual, applied/scenario-based
- Answers must be self-contained, accurate, and reference the {course} curriculum context
- Use the same language as the source text
- Output valid JSON only, no markdown, no preamble
- Format: [{{"question": "...", "answer": "..."}}]"""


def generate_qa(chunk: dict) -> list:
    """Generate Q&A pairs for a single chunk."""
    course = chunk["course"]
    module = chunk["module"]
    text = chunk["text"]

    try:
        response = client.chat.completions.create(
            model="local",
            messages=[
                {"role": "system", "content": get_system_prompt(course, module)},
                {"role": "user", "content": f"Text passage:\n\n{text}"}
            ],
            temperature=0.7,
            max_tokens=2048,
        )

        content = response.choices[0].message.content.strip()

        # Extract JSON from possible markdown wrapping
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        pairs = json.loads(content)

        # Tag each pair with metadata
        for pair in pairs:
            pair["course"] = course
            pair["module"] = module
            pair["source"] = chunk["source"]
            pair["page"] = chunk["page"]

        return pairs

    except json.JSONDecodeError as e:
        print(f"  JSON error [{course}/{module}] {chunk['source']} p{chunk['page']}: {e}")
        return []
    except Exception as e:
        print(f"  Error [{course}/{module}] {chunk['source']} p{chunk['page']}: {e}")
        return []


# Process
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

chunks = []
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    chunks = [json.loads(line) for line in f]

print(f"Processing {len(chunks)} chunks...")

total_pairs = 0
errors = 0

with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
    for i, chunk in enumerate(chunks):
        pairs = generate_qa(chunk)

        if pairs:
            for pair in pairs:
                fout.write(json.dumps(pair, ensure_ascii=False) + "\n")
            total_pairs += len(pairs)
        else:
            errors += 1

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(chunks)}] {total_pairs} pairs ({errors} errors)")

        time.sleep(0.1)

print(f"\nDone: {total_pairs} Q&A pairs ({errors} errors)")
print(f"Output: {OUTPUT_FILE}")
```

### Run it

```bash
python /srv/llama/training/generate_qa.py
```

---

## 6. Training Data Formatting

Converts Q&A pairs to chat format with course-specific system prompts.

### format_dataset.py

```python
#!/usr/bin/env python3
"""Convert Q&A pairs to chat-format training data.
Each example gets a course/module-specific system prompt."""

import json
import random
from pathlib import Path

INPUT_FILE = Path("/srv/llama/training/qa-pairs/qa_raw.jsonl")
TRAIN_FILE = Path("/srv/llama/training/dataset/train.jsonl")
VAL_FILE = Path("/srv/llama/training/dataset/val.jsonl")
STATS_FILE = Path("/srv/llama/training/dataset/stats.json")

MODULE_NAMES = {
    "accounting": "Accounting & Financial Reporting",
    "economics": "Economics",
    "equity_analysis": "Equity Analysis & Valuation",
    "fixed_income": "Fixed Income",
    "derivatives": "Derivatives & Structured Products",
    "portfolio_management": "Portfolio Management",
    "regulation": "Regulation & Compliance",
    "corporate_finance": "Corporate Finance",
    "client_advisory": "Client Advisory",
    "compliance": "Compliance & Legal",
    "wealth_planning": "Wealth Planning",
    "taxation": "Taxation",
    "estate_planning": "Estate Planning",
    "asset_management": "Asset Management",
    "risk_management": "Risk Management",
    "alternative_investments": "Alternative Investments",
}


def make_system_prompt(course: str, module: str) -> str:
    """Create a system prompt that teaches the model to associate
    knowledge with specific courses and modules."""
    module_display = MODULE_NAMES.get(module, module.replace("_", " ").title())
    return (
        f"You are a knowledgeable tutor for the {course} certification program. "
        f"You are answering questions about {module_display}. "
        f"Provide accurate, detailed answers based on the official {course} curriculum."
    )


# Load and format
pairs = []
stats = {"courses": {}, "modules": {}}

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        raw = json.loads(line)
        if not raw.get("question") or not raw.get("answer"):
            continue

        course = raw.get("course", "UNKNOWN")
        module = raw.get("module", "general")

        formatted = {
            "messages": [
                {"role": "system", "content": make_system_prompt(course, module)},
                {"role": "user", "content": raw["question"]},
                {"role": "assistant", "content": raw["answer"]},
            ],
            # Keep metadata for filtering (stripped before training if needed)
            "_course": course,
            "_module": module,
        }
        pairs.append(formatted)

        # Track stats
        stats["courses"][course] = stats["courses"].get(course, 0) + 1
        key = f"{course}/{module}"
        stats["modules"][key] = stats["modules"].get(key, 0) + 1

# Shuffle and split
random.seed(42)
random.shuffle(pairs)

split_idx = int(len(pairs) * 0.95)
train_data = pairs[:split_idx]
val_data = pairs[split_idx:]

# Write datasets
TRAIN_FILE.parent.mkdir(parents=True, exist_ok=True)

for filepath, data in [(TRAIN_FILE, train_data), (VAL_FILE, val_data)]:
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            # Remove metadata fields before writing
            output = {"messages": item["messages"]}
            f.write(json.dumps(output, ensure_ascii=False) + "\n")

# Write stats
stats["total"] = len(pairs)
stats["train"] = len(train_data)
stats["val"] = len(val_data)

with open(STATS_FILE, "w") as f:
    json.dump(stats, f, indent=2)

print(f"Train: {len(train_data)} examples → {TRAIN_FILE}")
print(f"Val:   {len(val_data)} examples → {VAL_FILE}")
print(f"\nDistribution by course:")
for course, count in sorted(stats["courses"].items()):
    print(f"  {course}: {count} examples")
print(f"\nDistribution by module:")
for module, count in sorted(stats["modules"].items()):
    print(f"  {module}: {count}")
```

### Run it

```bash
python /srv/llama/training/format_dataset.py
```

---

## 7. Fine-Tuning with Unsloth (QLoRA)

### finetune.py

```python
#!/usr/bin/env python3
"""Fine-tune Qwen3-8B on finance Q&A data using Unsloth + QLoRA."""

from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import torch

# ============================================================
# Configuration
# ============================================================
BASE_MODEL = "unsloth/Qwen3-8B-unsloth-bnb-4bit"
MAX_SEQ_LENGTH = 4096
OUTPUT_DIR = "/srv/llama/training/output"
TRAIN_FILE = "/srv/llama/training/dataset/train.jsonl"
VAL_FILE = "/srv/llama/training/dataset/val.jsonl"

LORA_R = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0.0

EPOCHS = 3
BATCH_SIZE = 2
GRAD_ACCUM = 4            # effective batch = 8
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01
LR_SCHEDULER = "cosine"

# ============================================================
# Load model
# ============================================================
print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    BASE_MODEL,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    dtype=None,
)

# ============================================================
# Apply LoRA
# ============================================================
print("Applying LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

# ============================================================
# Load dataset
# ============================================================
print("Loading dataset...")
dataset = load_dataset("json", data_files={
    "train": TRAIN_FILE,
    "validation": VAL_FILE,
})
print(f"Train: {len(dataset['train'])} | Val: {len(dataset['validation'])}")

# ============================================================
# Chat template formatting
# ============================================================
def formatting_func(examples):
    texts = []
    for messages in examples["messages"]:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        texts.append(text)
    return {"text": texts}

# ============================================================
# Training
# ============================================================
print("Starting training...")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    args=SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        max_seq_length=MAX_SEQ_LENGTH,
        seed=42,
        report_to="none",
    ),
    formatting_func=formatting_func,
)

stats = trainer.train()

print(f"\nTraining complete!")
print(f"  Steps: {stats.global_step} | Loss: {stats.training_loss:.4f}")

model.save_pretrained(f"{OUTPUT_DIR}/lora-adapter")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/lora-adapter")
print(f"LoRA adapter saved to {OUTPUT_DIR}/lora-adapter")
```

### Run it

```bash
python /srv/llama/training/finetune.py
```

---

## 8. Export to GGUF

### export_gguf.py

```python
#!/usr/bin/env python3
"""Export fine-tuned model to GGUF."""

from unsloth import FastLanguageModel

OUTPUT_DIR = "/srv/llama/training/output"
GGUF_DIR = "/srv/llama/models/qwen3-8b-finance"

print("Loading fine-tuned model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    f"{OUTPUT_DIR}/lora-adapter",
    max_seq_length=4096,
    load_in_4bit=True,
)

print("Exporting to GGUF (Q4_K_M)...")
model.save_pretrained_gguf(GGUF_DIR, tokenizer, quantization_method="q4_k_m")

print(f"\nDone → {GGUF_DIR}/")
```

```bash
python /srv/llama/training/export_gguf.py
```

---

## 9. Deploy with llama-server

```bash
llama-server \
  -m /srv/llama/models/qwen3-8b-finance/unsloth.Q4_K_M.gguf \
  --host 0.0.0.0 --port 8080 \
  -ngl 99 -c 4096 --jinja \
  --reasoning-format none \
  --temp 0.6 --top-k 20 --top-p 0.95
```

### Querying with course context

The model learned to associate knowledge with courses via system prompts. Use them at inference:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a tutor for the CIIA program. Answer questions about Accounting & Financial Reporting based on the CIIA curriculum."},
      {"role": "user", "content": "How is goodwill treated under IFRS vs US GAAP?"}
    ],
    "temperature": 0.6
  }'
```

Switch courses by changing the system prompt:

```json
{"role": "system", "content": "You are a tutor for the CIWM program. Answer questions about Wealth Planning."}
```

---

## 10. Evaluation

### eval_compare.py

```python
#!/usr/bin/env python3
"""Compare base vs fine-tuned model, per course."""

from openai import OpenAI

base = OpenAI(base_url="http://localhost:8080/v1", api_key="x")
ft = OpenAI(base_url="http://localhost:8081/v1", api_key="x")

# Test questions per course
TESTS = {
    "CIIA": {
        "system": "You are a tutor for the CIIA program.",
        "questions": [
            "Explain the difference between IFRS and US GAAP treatment of goodwill.",
            "What is the Sharpe ratio and how is it used?",
            "Describe the Black-Scholes model assumptions.",
        ],
    },
    "CIWM": {
        "system": "You are a tutor for the CIWM program.",
        "questions": [
            "What are the key elements of a client investment profile?",
            "How does Swiss estate law affect wealth planning?",
        ],
    },
}

for course, config in TESTS.items():
    print(f"\n{'='*70}")
    print(f"  COURSE: {course}")
    print(f"{'='*70}")

    for q in config["questions"]:
        msgs = [
            {"role": "system", "content": config["system"]},
            {"role": "user", "content": q},
        ]

        base_r = base.chat.completions.create(model="x", messages=msgs, temperature=0.3, max_tokens=500)
        ft_r = ft.chat.completions.create(model="x", messages=msgs, temperature=0.3, max_tokens=500)

        print(f"\nQ: {q}")
        print(f"\n--- BASE ---\n{base_r.choices[0].message.content}")
        print(f"\n--- FINE-TUNED ---\n{ft_r.choices[0].message.content}")
```

---

## 11. Tips and Troubleshooting

### Check your data balance

After running `format_dataset.py`, check `stats.json`. If one course dominates heavily (e.g. CIIA has 80% of the data), the model will be biased. Options:
- **Oversample** smaller courses (duplicate examples)
- **Undersample** the dominant course
- Add a weighting mechanism

### The SHARED folder

For manuals like `CIIA_ACC_6_PARTLY_CIWM_CF.pdf` that serve multiple courses, the extraction script creates separate tagged records for each course. This means the same text produces Q&A pairs tagged for both CIIA and CIWM, teaching the model that this knowledge is relevant to both programs.

### Complete file overview

```
/srv/llama/training/
├── setup_folders.sh
├── extract_text.py
├── clean_and_chunk.py
├── generate_qa.py
├── format_dataset.py
├── finetune.py
├── export_gguf.py
├── eval_compare.py
├── pdfs/
│   ├── CIIA/
│   │   ├── accounting/        ← CIIA_ACC_1.pdf ... CIIA_ACC_5.pdf
│   │   ├── economics/
│   │   └── .../
│   ├── CIWM/
│   │   ├── client_advisory/
│   │   └── .../
│   ├── AWM/
│   │   └── .../
│   └── SHARED/
│       └── corporate_finance/ ← CIIA_ACC_6_PARTLY_CIWM_CF.pdf
├── extracted/
│   └── raw_pages.jsonl        # tagged: {course, module, source, page, text}
├── cleaned/
│   └── chunks.jsonl           # tagged chunks
├── qa-pairs/
│   └── qa_raw.jsonl           # tagged Q&A pairs
├── dataset/
│   ├── train.jsonl            # 95% — course-aware system prompts
│   ├── val.jsonl              # 5%
│   └── stats.json             # distribution per course/module
└── output/
    └── lora-adapter/

/srv/llama/models/
└── qwen3-8b-finance/
    └── unsloth.Q4_K_M.gguf
```

---

*Generated for Christian's local LLM setup on Arch Linux.
RTX 4090 · AMD 7950X · 64GB DDR5 · /srv/llama/
Courses: CIIA, CIWM, AWM + shared modules*

