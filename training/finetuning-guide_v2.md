# Fine-Tuning Local LLMs — Complete Guide (v3)

**Target setup:** RTX 4090 (24GB) · AMD 7950X · 64GB DDR5 · Arch Linux
**Base model:** Qwen3-8B (for fine-tuning) · Qwen3.5-27B (for Q&A generation)
**Goal:** Fine-tune on finance training manuals across CIIA, AWM, CIWM, FMT, FMO

---

## Table of Contents

1. Folder structure (use as-is)
2. Environment setup
3. PDF text extraction (auto-parsing folder names)
4. Data cleaning
5. Q&A generation with local model
6. Training data formatting
7. Fine-tuning with Unsloth (QLoRA)
8. Export to GGUF
9. Deploy with llama-server
10. Evaluation
11. Tips and troubleshooting

---

## 1. Folder Structure

**No restructuring needed.** Copy your existing folder tree into `/srv/llama/training/pdfs/` as-is:

```
/srv/llama/training/pdfs/
├── 1- CIIA_ILPIP/
│   └── 2026/
│       ├── 1_Accounting - linked (SOME parts) 7 CIWM/
│       ├── 2_Corporate Finance - linked (SOME parts) 7 CIWM/
│       ├── 3_Derivatives/
│       ├── 4_Economics/
│       ├── 5_Equity -linked (SOME parts) 7 CIWM/
│       ├── 6_Fixed Income/
│       ├── 7_Portfolio Management/
│       ├── 8 Tax/
│       └── 9 Law/
├── 2- AWM/
│   └── 2026/
│       ├── 1_Financial_Instruments/
│       ├── 2_Wealth_Management/
│       ├── 3_Tax/
│       └── 4_Law/
├── 3- CIWM/
│   └── 2025/
│       ├── 1_Wealth Management/
│       ├── ...
│       └── 8_Financial Planning/
├── 4- FMT/
│   └── 2026/
│       ├── 1_Role and organisation of financial institutions/
│       ├── ...
│       └── 4_Financial Instruments (=AWM)/
└── 5- FMO/
    └── 2026/
        ├── 1 Operational process on financial instrument/
        ├── 2 Investment funds/
        └── 3 Management in operations/
```

The extraction script automatically:
- Parses the **course** from the top-level folder (e.g. `1- CIIA_ILPIP` → `CIIA`)
- Parses the **module** from the subfolder (e.g. `1_Accounting - linked (SOME parts) 7 CIWM` → `accounting`)
- Detects **linked/shared modules** from folder names containing "linked" or "=" and tags content with all relevant courses
- Ignores the year folder (2025, 2026) — it's just a pass-through

---

## 2. Environment Setup

```bash
source /srv/llama/llm-tools/bin/activate
pip install pymupdf sentencepiece protobuf
mkdir -p /srv/llama/training/{extracted,cleaned,qa-pairs,dataset,output}
```

---

## 3. PDF Text Extraction

### extract_text.py

```python
#!/usr/bin/env python3
"""Extract text from PDFs, auto-parsing course/module/links from folder names.

Expected layout:
    pdfs/<course_folder>/<year>/<module_folder>/<files>.pdf

Parses:
    "1- CIIA_ILPIP"  →  course: CIIA
    "2- AWM"         →  course: AWM
    "3- CIWM"        →  course: CIWM
    "4- FMT"         →  course: FMT
    "5- FMO"         →  course: FMO

    "1_Accounting - linked (SOME parts) 7 CIWM"  →  module: accounting, also_for: [CIWM]
    "4_Financial Instruments (=AWM)"              →  module: financial_instruments, also_for: [AWM]
"""

import json
import re
import fitz  # pymupdf
from pathlib import Path

PDF_DIR = Path("/srv/llama/training/pdfs")
OUTPUT_FILE = Path("/srv/llama/training/extracted/raw_pages.jsonl")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# ── Course name mapping ─────────────────────────────────────────────
# Maps keywords found in top-level folder names to canonical course IDs
COURSE_MAP = {
    "CIIA":  "CIIA",
    "ILPIP": "CIIA",
    "AWM":   "AWM",
    "CIWM":  "CIWM",
    "FMT":   "FMT",
    "FMO":   "FMO",
}

# Known course abbreviations used in link annotations
KNOWN_COURSES = {"CIIA", "AWM", "CIWM", "FMT", "FMO"}


def parse_course(folder_name: str) -> str:
    """Extract canonical course ID from top-level folder name.
    Examples:
        '1- CIIA_ILPIP' → 'CIIA'
        '2- AWM'        → 'AWM'
        '5- FMO'        → 'FMO'
    """
    upper = folder_name.upper()
    for key, course in COURSE_MAP.items():
        if key in upper:
            return course
    return folder_name.strip()


def parse_module(folder_name: str) -> tuple[str, list[str]]:
    """Extract module name and linked courses from module folder name.

    Returns:
        (module_name, linked_courses)

    Examples:
        '1_Accounting - linked (SOME parts) 7 CIWM'
            → ('accounting', ['CIWM'])

        '4_Financial Instruments (=AWM)'
            → ('financial_instruments', ['AWM'])

        '3_Derivatives'
            → ('derivatives', [])

        '1 Operational process on financial instrument'
            → ('operational_process_on_financial_instrument', [])
    """
    name = folder_name

    # Detect linked courses from "(=XXX)" pattern
    linked = []
    eq_match = re.search(r'\(=\s*(\w+)\)', name)
    if eq_match:
        candidate = eq_match.group(1).upper()
        if candidate in KNOWN_COURSES:
            linked.append(candidate)
        name = name[:eq_match.start()]  # remove the (=XXX) part

    # Detect linked courses from "linked ... 7 CIWM" or "linked ... CIWM" pattern
    link_match = re.search(r'[-–]\s*linked.*?(\d\s+)?([A-Z]{2,})', name, re.IGNORECASE)
    if link_match:
        candidate = link_match.group(2).upper()
        if candidate in KNOWN_COURSES:
            linked.append(candidate)
        # Remove everything from " - linked" onwards
        name = re.sub(r'\s*[-–]\s*linked.*$', '', name, flags=re.IGNORECASE)

    # Remove leading number + separator: "1_", "1 ", "8 "
    name = re.sub(r'^\d+[_\s]+', '', name)

    # Clean up: lowercase, replace spaces/special chars with underscore
    name = name.strip().lower()
    name = re.sub(r'[^a-z0-9äöüéèà]+', '_', name)
    name = name.strip('_')

    return name, list(set(linked))


def is_year_folder(name: str) -> bool:
    """Check if a folder name is just a year (2024, 2025, 2026, etc.)."""
    return bool(re.match(r'^\d{4}$', name.strip()))


def find_pdf_tags(pdf_path: Path) -> list[dict]:
    """Walk up from the PDF to determine course, module, and any links.

    Expected path structure:
        pdfs / <course_folder> / <year> / <module_folder> / file.pdf
    """
    rel = pdf_path.relative_to(PDF_DIR)
    parts = rel.parts  # e.g. ("1- CIIA_ILPIP", "2026", "1_Accounting - linked...", "file.pdf")

    if len(parts) < 3:
        return [{"course": "UNKNOWN", "module": "general"}]

    # Find course (first non-year directory)
    course_folder = parts[0]
    primary_course = parse_course(course_folder)

    # Find module (skip year folders)
    module_folder = None
    for part in parts[1:-1]:  # exclude filename
        if not is_year_folder(part):
            module_folder = part
            break

    if module_folder is None:
        # PDF directly under year folder, no module subfolder
        return [{"course": primary_course, "module": "general"}]

    module_name, linked_courses = parse_module(module_folder)

    # Build tag list: primary course + any linked courses
    tags = [{"course": primary_course, "module": module_name}]
    for linked in linked_courses:
        if linked != primary_course:
            tags.append({"course": linked, "module": module_name})

    return tags


# ── Main extraction ──────────────────────────────────────────────────
total_pages = 0
total_docs = 0
all_tags_seen = set()

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for pdf_path in sorted(PDF_DIR.rglob("*.pdf")):
        tags = find_pdf_tags(pdf_path)
        rel_path = pdf_path.relative_to(PDF_DIR)

        tag_str = ", ".join(f"{t['course']}/{t['module']}" for t in tags)
        print(f"{rel_path}")
        print(f"  → {tag_str}")

        for t in tags:
            all_tags_seen.add(f"{t['course']}/{t['module']}")

        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                text = page.get_text("text").strip()
                if len(text) > 50:
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

print(f"\n{'='*60}")
print(f"Extracted {total_pages} tagged pages from {total_docs} PDFs")
print(f"Output: {OUTPUT_FILE}")
print(f"\nAll course/module combinations found:")
for tag in sorted(all_tags_seen):
    print(f"  {tag}")
```

### Run it

```bash
python /srv/llama/training/extract_text.py
```

### Expected output

```
1- CIIA_ILPIP/2026/1_Accounting - linked (SOME parts) 7 CIWM/CIIA_ACC_1.pdf
  → CIIA/accounting, CIWM/accounting
1- CIIA_ILPIP/2026/3_Derivatives/CIIA_DER_1.pdf
  → CIIA/derivatives
4- FMT/2026/4_Financial Instruments (=AWM)/FMT_FI_1.pdf
  → FMT/financial_instruments, AWM/financial_instruments
...

All course/module combinations found:
  AWM/financial_instruments
  AWM/law
  AWM/tax
  AWM/wealth_management
  CIIA/accounting
  CIIA/corporate_finance
  CIIA/derivatives
  CIIA/economics
  CIIA/equity
  CIIA/fixed_income
  CIIA/law
  CIIA/portfolio_management
  CIIA/tax
  CIWM/accounting
  CIWM/behavioural_finance
  CIWM/corporate_finance
  CIWM/equity
  CIWM/financial_instruments
  CIWM/financial_planning
  CIWM/law
  CIWM/relationship_management
  CIWM/tax
  CIWM/wealth_management
  FMO/investment_funds
  FMO/management_in_operations
  FMO/operational_process_on_financial_instrument
  FMT/custodian_activities
  FMT/financial_instruments
  FMT/role_and_organisation_of_financial_institutions
  FMT/trade_and_post_trade_functions
```

### Verify the tag distribution

```bash
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
for k, v in modules.most_common(20):
    print(f'  {k}: {v}')
"
```

---

## 4. Data Cleaning

### clean_and_chunk.py

```python
#!/usr/bin/env python3
"""Clean extracted text and split into chunks. Preserves course/module tags."""

import json
import re
from pathlib import Path

INPUT_FILE = Path("/srv/llama/training/extracted/raw_pages.jsonl")
OUTPUT_FILE = Path("/srv/llama/training/cleaned/chunks.jsonl")

MAX_CHARS = 4000
OVERLAP_CHARS = 400


def clean_text(text: str) -> str:
    """Remove common PDF artifacts."""
    text = re.sub(r"^\s*\d{1,3}\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"(?i)^.*confidential.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"(?i)^.*all rights reserved.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"(?i)^.*AZEK.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"(?i)^.*Swiss Training Centre.*$", "", text, flags=re.MULTILINE)
    # Fix broken line wraps (letter-newline-lowercase letter)
    text = re.sub(r"([a-zäöüéèà,])\n([a-zäöüéèà])", r"\1 \2", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def chunk_text(text: str, metadata: dict) -> list:
    """Split text into overlapping chunks, carrying metadata."""
    if len(text) <= MAX_CHARS:
        return [{**metadata, "text": text}]

    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = start + MAX_CHARS
        if end < len(text):
            bp = text.rfind("\n\n", start + MAX_CHARS // 2, end)
            if bp == -1:
                bp = text.rfind(". ", start + MAX_CHARS // 2, end)
                if bp != -1:
                    bp += 2
            if bp and bp > start:
                end = bp
        chunk = text[start:end].strip()
        if len(chunk) > 50:
            chunks.append({**metadata, "chunk": idx, "text": chunk})
            idx += 1
        start = end - OVERLAP_CHARS
    return chunks


OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
total = 0

with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
    for line in fin:
        record = json.loads(line)
        cleaned = clean_text(record["text"])
        if cleaned:
            meta = {k: record[k] for k in ("course", "module", "source", "page")}
            for chunk in chunk_text(cleaned, meta):
                fout.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                total += 1

print(f"Done: {total} chunks → {OUTPUT_FILE}")
```

```bash
python /srv/llama/training/clean_and_chunk.py
```

---

## 5. Q&A Generation with Local Model

Start llama-server:

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
"""Generate Q&A pairs from chunks using local LLM. Course/module-aware prompts."""

import json
import time
from pathlib import Path
from openai import OpenAI

INPUT_FILE = Path("/srv/llama/training/cleaned/chunks.jsonl")
OUTPUT_FILE = Path("/srv/llama/training/qa-pairs/qa_raw.jsonl")

client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")

# Human-readable module names for better prompts
MODULE_DISPLAY = {
    "accounting": "Accounting & Financial Reporting",
    "corporate_finance": "Corporate Finance",
    "derivatives": "Derivatives & Structured Products",
    "economics": "Economics",
    "equity": "Equity Analysis & Valuation",
    "fixed_income": "Fixed Income",
    "portfolio_management": "Portfolio Management",
    "tax": "Taxation",
    "law": "Law & Regulation",
    "financial_instruments": "Financial Instruments",
    "wealth_management": "Wealth Management",
    "behavioural_finance": "Behavioural Finance",
    "relationship_management": "Relationship Management",
    "financial_planning": "Financial Planning",
    "client_advisory": "Client Advisory",
    "compliance": "Compliance",
    "estate_planning": "Estate Planning",
    "risk_management": "Risk Management",
    "alternative_investments": "Alternative Investments",
    "role_and_organisation_of_financial_institutions": "Role & Organisation of Financial Institutions",
    "trade_and_post_trade_functions": "Trade & Post-Trade Functions",
    "custodian_activities": "Custodian Activities",
    "operational_process_on_financial_instrument": "Operational Processes on Financial Instruments",
    "investment_funds": "Investment Funds",
    "management_in_operations": "Management in Operations",
}

COURSE_FULLNAMES = {
    "CIIA": "Certified International Investment Analyst (CIIA)",
    "CIWM": "Certified International Wealth Manager (CIWM)",
    "AWM": "Advanced Wealth Management (AWM)",
    "FMT": "Financial Market Technician (FMT)",
    "FMO": "Financial Market Operations (FMO)",
}


def get_system_prompt(course: str, module: str) -> str:
    course_full = COURSE_FULLNAMES.get(course, course)
    module_full = MODULE_DISPLAY.get(module, module.replace("_", " ").title())

    return f"""You are an expert in finance education for the {course_full} program, specifically the module: {module_full}.

Given a text passage from the {course} training manual on {module_full}, generate 3-5 high-quality question-answer pairs.

Rules:
- Vary question types: definitional, conceptual, applied/scenario-based
- Answers must be self-contained, accurate, and detailed
- Use the same language as the source text
- Output valid JSON only, no markdown, no preamble
- Format: [{{"question": "...", "answer": "..."}}]"""


def generate_qa(chunk: dict) -> list:
    course = chunk["course"]
    module = chunk["module"]

    try:
        response = client.chat.completions.create(
            model="local",
            messages=[
                {"role": "system", "content": get_system_prompt(course, module)},
                {"role": "user", "content": f"Text passage:\n\n{chunk['text']}"}
            ],
            temperature=0.7,
            max_tokens=2048,
        )

        content = response.choices[0].message.content.strip()
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        pairs = json.loads(content)
        for pair in pairs:
            pair["course"] = course
            pair["module"] = module
            pair["source"] = chunk["source"]
            pair["page"] = chunk["page"]
        return pairs

    except json.JSONDecodeError:
        return []
    except Exception as e:
        print(f"  Error [{course}/{module}] {chunk['source']} p{chunk['page']}: {e}")
        return []


# ── Main ──
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

chunks = []
with open(INPUT_FILE) as f:
    chunks = [json.loads(line) for line in f]

print(f"Processing {len(chunks)} chunks...")
total_pairs = 0
errors = 0

with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
    for i, chunk in enumerate(chunks):
        pairs = generate_qa(chunk)
        if pairs:
            for p in pairs:
                fout.write(json.dumps(p, ensure_ascii=False) + "\n")
            total_pairs += len(pairs)
        else:
            errors += 1

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(chunks)}] {total_pairs} pairs ({errors} errors)")

        time.sleep(0.1)

print(f"\nDone: {total_pairs} Q&A pairs ({errors} errors) → {OUTPUT_FILE}")
```

```bash
python /srv/llama/training/generate_qa.py
```

---

## 6. Training Data Formatting

### format_dataset.py

```python
#!/usr/bin/env python3
"""Convert Q&A pairs to chat-format training data with course-aware system prompts."""

import json
import random
from pathlib import Path

INPUT_FILE = Path("/srv/llama/training/qa-pairs/qa_raw.jsonl")
TRAIN_FILE = Path("/srv/llama/training/dataset/train.jsonl")
VAL_FILE = Path("/srv/llama/training/dataset/val.jsonl")
STATS_FILE = Path("/srv/llama/training/dataset/stats.json")

MODULE_DISPLAY = {
    "accounting": "Accounting & Financial Reporting",
    "corporate_finance": "Corporate Finance",
    "derivatives": "Derivatives & Structured Products",
    "economics": "Economics",
    "equity": "Equity Analysis & Valuation",
    "fixed_income": "Fixed Income",
    "portfolio_management": "Portfolio Management",
    "tax": "Taxation",
    "law": "Law & Regulation",
    "financial_instruments": "Financial Instruments",
    "wealth_management": "Wealth Management",
    "behavioural_finance": "Behavioural Finance",
    "relationship_management": "Relationship Management",
    "financial_planning": "Financial Planning",
    "role_and_organisation_of_financial_institutions": "Role & Organisation of Financial Institutions",
    "trade_and_post_trade_functions": "Trade & Post-Trade Functions",
    "custodian_activities": "Custodian Activities",
    "operational_process_on_financial_instrument": "Operational Processes on Financial Instruments",
    "investment_funds": "Investment Funds",
    "management_in_operations": "Management in Operations",
}


def make_system_prompt(course: str, module: str) -> str:
    mod = MODULE_DISPLAY.get(module, module.replace("_", " ").title())
    return (
        f"You are a knowledgeable tutor for the {course} certification program. "
        f"You are answering questions about {mod}. "
        f"Provide accurate, detailed answers based on the official {course} curriculum."
    )


pairs = []
stats = {"courses": {}, "modules": {}}

with open(INPUT_FILE) as f:
    for line in f:
        raw = json.loads(line)
        if not raw.get("question") or not raw.get("answer"):
            continue

        course = raw.get("course", "UNKNOWN")
        module = raw.get("module", "general")

        pairs.append({
            "messages": [
                {"role": "system", "content": make_system_prompt(course, module)},
                {"role": "user", "content": raw["question"]},
                {"role": "assistant", "content": raw["answer"]},
            ]
        })

        stats["courses"][course] = stats["courses"].get(course, 0) + 1
        key = f"{course}/{module}"
        stats["modules"][key] = stats["modules"].get(key, 0) + 1

random.seed(42)
random.shuffle(pairs)

split = int(len(pairs) * 0.95)
train_data, val_data = pairs[:split], pairs[split:]

TRAIN_FILE.parent.mkdir(parents=True, exist_ok=True)

for path, data in [(TRAIN_FILE, train_data), (VAL_FILE, val_data)]:
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

stats["total"] = len(pairs)
stats["train"] = len(train_data)
stats["val"] = len(val_data)
with open(STATS_FILE, "w") as f:
    json.dump(stats, f, indent=2)

print(f"Train: {len(train_data)} | Val: {len(val_data)}")
print(f"\nPer course:")
for c, n in sorted(stats["courses"].items()):
    print(f"  {c}: {n}")
print(f"\nPer module:")
for m, n in sorted(stats["modules"].items()):
    print(f"  {m}: {n}")
```

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
print(f"Saved → {OUTPUT_DIR}/lora-adapter")
```

```bash
python /srv/llama/training/finetune.py
```

---

## 8. Export to GGUF

### export_gguf.py

```python
#!/usr/bin/env python3
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "/srv/llama/training/output/lora-adapter",
    max_seq_length=4096, load_in_4bit=True,
)
model.save_pretrained_gguf(
    "/srv/llama/models/qwen3-8b-finance", tokenizer, quantization_method="q4_k_m",
)
print("Done → /srv/llama/models/qwen3-8b-finance/")
```

```bash
python /srv/llama/training/export_gguf.py
```

---

## 9. Deploy

```bash
llama-server \
  -m /srv/llama/models/qwen3-8b-finance/unsloth.Q4_K_M.gguf \
  --host 0.0.0.0 --port 8080 \
  -ngl 99 -c 4096 --jinja \
  --reasoning-format none \
  -fa on --temp 0.6 --top-k 20 --top-p 0.95
```

Query with course context:

```bash
# CIIA question
curl -s http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "messages": [
    {"role": "system", "content": "You are a tutor for the CIIA program. Answer questions about Fixed Income."},
    {"role": "user", "content": "Explain modified duration and its relationship to bond price sensitivity."}
  ]
}' | python -m json.tool

# FMO question
curl -s http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "messages": [
    {"role": "system", "content": "You are a tutor for the FMO program. Answer questions about Investment Funds."},
    {"role": "user", "content": "What are the key operational risks in fund administration?"}
  ]
}'
```

---

## 10. Evaluation

### eval_compare.py

```python
#!/usr/bin/env python3
"""Compare base vs fine-tuned model across courses."""

from openai import OpenAI

base = OpenAI(base_url="http://localhost:8080/v1", api_key="x")
ft = OpenAI(base_url="http://localhost:8081/v1", api_key="x")

TESTS = {
    "CIIA": [
        ("fixed_income", "Explain the concept of duration in bond pricing."),
        ("derivatives", "What is the put-call parity relationship?"),
        ("accounting", "How is goodwill treated under IFRS?"),
    ],
    "CIWM": [
        ("wealth_management", "What are the key elements of a client investment profile?"),
        ("behavioural_finance", "Explain the disposition effect in investor behavior."),
    ],
    "AWM": [
        ("financial_instruments", "Compare ETFs and index funds for portfolio construction."),
    ],
    "FMT": [
        ("trade_and_post_trade_functions", "What is the role of a central counterparty (CCP)?"),
    ],
    "FMO": [
        ("investment_funds", "What are the key operational risks in fund administration?"),
    ],
}

for course, questions in TESTS.items():
    print(f"\n{'='*70}\n  {course}\n{'='*70}")
    for module, q in questions:
        sys_msg = f"You are a tutor for the {course} program. Answer about {module.replace('_',' ').title()}."
        msgs = [{"role": "system", "content": sys_msg}, {"role": "user", "content": q}]

        b = base.chat.completions.create(model="x", messages=msgs, temperature=0.3, max_tokens=500)
        f = ft.chat.completions.create(model="x", messages=msgs, temperature=0.3, max_tokens=500)

        print(f"\nQ: {q}")
        print(f"--- BASE ---\n{b.choices[0].message.content[:300]}...")
        print(f"--- FINE-TUNED ---\n{f.choices[0].message.content[:300]}...")
```

---

## 11. Tips

### How linked modules work

When the folder name contains "linked ... 7 CIWM" or "(=AWM)", the extraction script tags that content with both the primary course AND the linked course. For example:

- `1_Accounting - linked (SOME parts) 7 CIWM` in the CIIA folder
  → Pages tagged as **CIIA/accounting** AND **CIWM/accounting**
- `4_Financial Instruments (=AWM)` in the FMT folder
  → Pages tagged as **FMT/financial_instruments** AND **AWM/financial_instruments**

This means the model learns that this knowledge is relevant to both courses.

### Adding a new course or year

Just add the folder under `pdfs/` following the same pattern. The script handles it automatically. If you add a new course abbreviation, add it to `COURSE_MAP` and `KNOWN_COURSES` in `extract_text.py`.

### Iterate on a subset first

```bash
# Process only CIIA for a quick test
find /srv/llama/training/pdfs -path "*/CIIA*" -name "*.pdf" | head -5
```

Or temporarily move other course folders out of `pdfs/` and run the full pipeline on just one course to validate quality before scaling up.

---

*Generated for Christian's local LLM setup on Arch Linux.
RTX 4090 · AMD 7950X · 64GB DDR5 · /srv/llama/
Courses: CIIA, CIWM, AWM, FMT, FMO*
