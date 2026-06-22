# Fine-Tuning Local LLMs — Complete Guide (v4)

**Target setup:** RTX 4090 (24GB) · AMD 7950X · 64GB DDR5 · Arch Linux
**Base model:** Qwen3-8B (for fine-tuning) · Qwen3.5-27B (for Q&A generation)
**Goal:** Fine-tune on finance training manuals across CIIA, AWM, CIWM, FMT, FMO
**Languages:** English, German, French (auto-detected from AWM filenames)

---

## Table of Contents

1. Folder structure (use as-is)
2. Environment setup
3. PDF text extraction (course/module/language-aware)
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
│       │   ├── AWM_FI_1d.pdf          ← German
│       │   ├── AWM_FI_1e_FMT_FI_1.pdf ← English (also FMT)
│       │   ├── AWM_FI_1f.pdf          ← French
│       │   └── ...
│       ├── 2_Wealth_Management/
│       ├── 3_Tax/
│       └── 4_Law/
├── 3- CIWM/
│   └── 2025/
│       └── .../
├── 4- FMT/
│   └── 2026/
│       └── .../
└── 5- FMO/
    └── 2026/
        └── .../
```

The extraction script automatically:
- Parses **course** from the top-level folder
- Parses **module** from the subfolder
- Detects **linked/shared** modules from folder names ("linked ... 7 CIWM", "(=AWM)")
- Detects **language** from AWM filename suffixes (`d` = German, `e` = English, `f` = French)
- Skips year folders (2025, 2026)

---

## 2. Environment Setup

```bash
source /srv/llama/llm-tools/bin/activate
pip install pymupdf sentencepiece protobuf
mkdir -p /srv/llama/training/{extracted,cleaned,qa-pairs,dataset,output}
```

Verify GPU:

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```

---

## 3. PDF Text Extraction

### extract_text.py

```python
#!/usr/bin/env python3
"""Extract text from PDFs, auto-parsing course/module/language/links from folder and file names.

Expected layout:
    pdfs/<course_folder>/<year>/<module_folder>/<files>.pdf

Course parsing:
    "1- CIIA_ILPIP"  →  CIIA
    "2- AWM"         →  AWM
    "3- CIWM"        →  CIWM
    "4- FMT"         →  FMT
    "5- FMO"         →  FMO

Module parsing:
    "1_Accounting - linked (SOME parts) 7 CIWM"  →  module: accounting, also_for: [CIWM]
    "4_Financial Instruments (=AWM)"              →  module: financial_instruments, also_for: [AWM]

Language detection (from AWM-style filenames):
    AWM_FI_1d.pdf          →  de (German)
    AWM_FI_1e_FMT_FI_1.pdf →  en (English)
    AWM_FI_1f.pdf          →  fr (French)
"""

import json
import re
import fitz  # pymupdf
from pathlib import Path

PDF_DIR = Path("/srv/llama/training/pdfs")
OUTPUT_FILE = Path("/srv/llama/training/extracted/raw_pages.jsonl")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# ── Course name mapping ─────────────────────────────────────────────
COURSE_MAP = {
    "CIIA":  "CIIA",
    "ILPIP": "CIIA",
    "AWM":   "AWM",
    "CIWM":  "CIWM",
    "FMT":   "FMT",
    "FMO":   "FMO",
}

KNOWN_COURSES = {"CIIA", "AWM", "CIWM", "FMT", "FMO"}

# ── Language suffix map ──────────────────────────────────────────────
LANG_SUFFIXES = {"d": "de", "e": "en", "f": "fr"}


def parse_course(folder_name: str) -> str:
    """Extract canonical course ID from top-level folder name."""
    upper = folder_name.upper()
    for key, course in COURSE_MAP.items():
        if key in upper:
            return course
    return folder_name.strip()


def parse_module(folder_name: str) -> tuple[str, list[str]]:
    """Extract module name and linked courses from module folder name.

    Returns: (module_name, linked_courses)
    """
    name = folder_name
    linked = []

    # Detect "(=XXX)" pattern
    eq_match = re.search(r'\(=\s*(\w+)\)', name)
    if eq_match:
        candidate = eq_match.group(1).upper()
        if candidate in KNOWN_COURSES:
            linked.append(candidate)
        name = name[:eq_match.start()]

    # Detect "linked ... 7 CIWM" pattern
    link_match = re.search(r'[-–]\s*linked.*?(\d\s+)?([A-Z]{2,})', name, re.IGNORECASE)
    if link_match:
        candidate = link_match.group(2).upper()
        if candidate in KNOWN_COURSES:
            linked.append(candidate)
        name = re.sub(r'\s*[-–]\s*linked.*$', '', name, flags=re.IGNORECASE)

    # Remove leading number + separator
    name = re.sub(r'^\d+[_\s]+', '', name)

    # Normalize to snake_case
    name = name.strip().lower()
    name = re.sub(r'[^a-z0-9äöüéèà]+', '_', name)
    name = name.strip('_')

    return name, list(set(linked))


def detect_language(filename: str) -> str:
    """Detect language from filename suffix.

    AWM_FI_1d.pdf          → de
    AWM_FI_1e_FMT_FI_1.pdf → en
    AWM_FI_1f.pdf          → fr

    Non-matching files default to English.
    """
    stem = Path(filename).stem.lower()
    # Match: digit followed by d/e/f, then either underscore or end-of-string
    match = re.search(r'_(\d+)([def])(?:_|$)', stem)
    if match:
        lang_char = match.group(2)
        return LANG_SUFFIXES.get(lang_char, "en")
    return "en"


def is_year_folder(name: str) -> bool:
    """Check if a folder name is just a year."""
    return bool(re.match(r'^\d{4}$', name.strip()))


def find_pdf_tags(pdf_path: Path) -> list[dict]:
    """Determine course, module, language, and cross-references for a PDF."""
    rel = pdf_path.relative_to(PDF_DIR)
    parts = rel.parts

    if len(parts) < 3:
        return [{"course": "UNKNOWN", "module": "general"}]

    # Course from first directory
    primary_course = parse_course(parts[0])

    # Module from first non-year subdirectory
    module_folder = None
    for part in parts[1:-1]:
        if not is_year_folder(part):
            module_folder = part
            break

    if module_folder is None:
        return [{"course": primary_course, "module": "general"}]

    module_name, linked_courses = parse_module(module_folder)

    # Language from filename
    lang = detect_language(pdf_path.name)

    # Build tag list
    tags = [{"course": primary_course, "module": module_name}]
    for linked in linked_courses:
        if linked != primary_course:
            tags.append({"course": linked, "module": module_name})

    return tags, lang


# ── Main extraction ──────────────────────────────────────────────────
total_pages = 0
total_docs = 0
all_tags_seen = set()
lang_counts = {"en": 0, "de": 0, "fr": 0}

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for pdf_path in sorted(PDF_DIR.rglob("*.pdf")):
        result = find_pdf_tags(pdf_path)

        # Handle both old (list) and new (tuple) return
        if isinstance(result, tuple):
            tags, lang = result
        else:
            tags = result
            lang = "en"

        rel_path = pdf_path.relative_to(PDF_DIR)
        tag_str = ", ".join(f"{t['course']}/{t['module']}" for t in tags)
        print(f"{rel_path}")
        print(f"  → {tag_str} [{lang}]")

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
                            "lang": lang,
                            "source": pdf_path.name,
                            "page": page.number + 1,
                            "text": text,
                        }
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        total_pages += 1
                        lang_counts[lang] = lang_counts.get(lang, 0) + 1
            doc.close()
            total_docs += 1
        except Exception as e:
            print(f"  ERROR: {e}")

print(f"\n{'='*60}")
print(f"Extracted {total_pages} tagged pages from {total_docs} PDFs")
print(f"Output: {OUTPUT_FILE}")
print(f"\nLanguage distribution:")
for lang, count in sorted(lang_counts.items()):
    lang_name = {"en": "English", "de": "German", "fr": "French"}.get(lang, lang)
    print(f"  {lang_name}: {count}")
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
2- AWM/2026/1_Financial_Instruments/AWM_FI_1d.pdf
  → AWM/financial_instruments [de]
2- AWM/2026/1_Financial_Instruments/AWM_FI_1e_FMT_FI_1.pdf
  → AWM/financial_instruments [en]
2- AWM/2026/1_Financial_Instruments/AWM_FI_1f.pdf
  → AWM/financial_instruments [fr]
1- CIIA_ILPIP/2026/1_Accounting - linked (SOME parts) 7 CIWM/CIIA_ACC_1.pdf
  → CIIA/accounting, CIWM/accounting [en]
...

Language distribution:
  English: 5200
  German: 1800
  French: 1800

All course/module combinations found:
  AWM/financial_instruments
  AWM/law
  ...
```

### Verify tag + language distribution

```bash
python3 -c "
import json
from collections import Counter

courses = Counter()
modules = Counter()
langs = Counter()
combo = Counter()

with open('/srv/llama/training/extracted/raw_pages.jsonl') as f:
    for line in f:
        r = json.loads(line)
        courses[r['course']] += 1
        modules[f\"{r['course']}/{r['module']}\"] += 1
        langs[r['lang']] += 1
        combo[f\"{r['course']}/{r['lang']}\"] += 1

print('Pages per course:')
for k, v in courses.most_common():
    print(f'  {k}: {v}')
print()
print('Pages per language:')
for k, v in langs.most_common():
    print(f'  {k}: {v}')
print()
print('Course × Language:')
for k, v in sorted(combo.items()):
    print(f'  {k}: {v}')
"
```

---

## 4. Data Cleaning

### clean_and_chunk.py

```python
#!/usr/bin/env python3
"""Clean extracted text and split into chunks. Preserves course/module/lang tags."""

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
            meta = {k: record[k] for k in ("course", "module", "lang", "source", "page")}
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
"""Generate Q&A pairs from chunks using local LLM.
Course/module/language-aware prompts."""

import json
import time
from pathlib import Path
from openai import OpenAI

INPUT_FILE = Path("/srv/llama/training/cleaned/chunks.jsonl")
OUTPUT_FILE = Path("/srv/llama/training/qa-pairs/qa_raw.jsonl")

client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")

# ── Display names ────────────────────────────────────────────────────
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

LANG_NAMES = {
    "en": "English",
    "de": "German",
    "fr": "French",
}


def get_system_prompt(course: str, module: str, lang: str) -> str:
    course_full = COURSE_FULLNAMES.get(course, course)
    module_full = MODULE_DISPLAY.get(module, module.replace("_", " ").title())
    lang_name = LANG_NAMES.get(lang, "English")

    return f"""You are an expert in finance education for the {course_full} program, specifically the module: {module_full}.

Given a text passage in {lang_name} from the {course} training manual on {module_full}, generate 3-5 high-quality question-answer pairs.

Rules:
- Generate questions and answers in {lang_name} (same language as the source text)
- Vary question types: definitional, conceptual, applied/scenario-based
- Answers must be self-contained, accurate, and detailed
- Output valid JSON only, no markdown, no preamble
- Format: [{{"question": "...", "answer": "..."}}]"""


def generate_qa(chunk: dict) -> list:
    course = chunk["course"]
    module = chunk["module"]
    lang = chunk.get("lang", "en")

    try:
        response = client.chat.completions.create(
            model="local",
            messages=[
                {"role": "system", "content": get_system_prompt(course, module, lang)},
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
            pair["lang"] = lang
            pair["source"] = chunk["source"]
            pair["page"] = chunk["page"]
        return pairs

    except json.JSONDecodeError:
        return []
    except Exception as e:
        print(f"  Error [{course}/{module}/{lang}] {chunk['source']} p{chunk['page']}: {e}")
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
"""Convert Q&A pairs to chat-format training data.
Course/module/language-specific system prompts."""

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

LANG_INSTRUCTION = {
    "en": "",
    "de": " Antworte auf Deutsch.",
    "fr": " Répondez en français.",
}


def make_system_prompt(course: str, module: str, lang: str) -> str:
    mod = MODULE_DISPLAY.get(module, module.replace("_", " ").title())
    lang_suffix = LANG_INSTRUCTION.get(lang, "")
    return (
        f"You are a knowledgeable tutor for the {course} certification program. "
        f"You are answering questions about {mod}. "
        f"Provide accurate, detailed answers based on the official {course} curriculum."
        f"{lang_suffix}"
    )


# ── Load and format ──
pairs = []
stats = {"courses": {}, "modules": {}, "langs": {}}

with open(INPUT_FILE) as f:
    for line in f:
        raw = json.loads(line)
        if not raw.get("question") or not raw.get("answer"):
            continue

        course = raw.get("course", "UNKNOWN")
        module = raw.get("module", "general")
        lang = raw.get("lang", "en")

        pairs.append({
            "messages": [
                {"role": "system", "content": make_system_prompt(course, module, lang)},
                {"role": "user", "content": raw["question"]},
                {"role": "assistant", "content": raw["answer"]},
            ]
        })

        stats["courses"][course] = stats["courses"].get(course, 0) + 1
        stats["modules"][f"{course}/{module}"] = stats["modules"].get(f"{course}/{module}", 0) + 1
        stats["langs"][lang] = stats["langs"].get(lang, 0) + 1

# ── Shuffle and split ──
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
print(f"\nPer language:")
for l, n in sorted(stats["langs"].items()):
    lang_name = {"en": "English", "de": "German", "fr": "French"}.get(l, l)
    print(f"  {lang_name}: {n}")
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

### Querying in different languages

```bash
# English — CIIA Fixed Income
curl -s http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "messages": [
    {"role": "system", "content": "You are a tutor for the CIIA program. Answer questions about Fixed Income."},
    {"role": "user", "content": "Explain modified duration and its relationship to bond price sensitivity."}
  ]
}'

# German — AWM Financial Instruments
curl -s http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "messages": [
    {"role": "system", "content": "You are a tutor for the AWM program. Answer questions about Financial Instruments. Antworte auf Deutsch."},
    {"role": "user", "content": "Was ist der Unterschied zwischen einem ETF und einem Indexfonds?"}
  ]
}'

# French — AWM Financial Instruments
curl -s http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "messages": [
    {"role": "system", "content": "You are a tutor for the AWM program. Answer questions about Financial Instruments. Répondez en français."},
    {"role": "user", "content": "Quelle est la différence entre une obligation et une action?"}
  ]
}'
```

---

## 10. Evaluation

### eval_compare.py

```python
#!/usr/bin/env python3
"""Compare base vs fine-tuned model across courses and languages."""

from openai import OpenAI

base = OpenAI(base_url="http://localhost:8080/v1", api_key="x")
ft = OpenAI(base_url="http://localhost:8081/v1", api_key="x")

TESTS = [
    # (course, module, lang, system_suffix, question)
    ("CIIA", "fixed_income", "en", "",
     "Explain the concept of duration in bond pricing."),
    ("CIIA", "derivatives", "en", "",
     "What is the put-call parity relationship?"),
    ("AWM", "financial_instruments", "de", " Antworte auf Deutsch.",
     "Was ist der Unterschied zwischen systematischem und unsystematischem Risiko?"),
    ("AWM", "financial_instruments", "fr", " Répondez en français.",
     "Quelle est la différence entre un ETF et un fonds indiciel?"),
    ("CIWM", "wealth_management", "en", "",
     "What are the key elements of a client investment profile?"),
    ("FMT", "trade_and_post_trade_functions", "en", "",
     "What is the role of a central counterparty (CCP)?"),
    ("FMO", "investment_funds", "en", "",
     "What are the key operational risks in fund administration?"),
]

for course, module, lang, suffix, question in TESTS:
    mod_display = module.replace("_", " ").title()
    sys_msg = f"You are a tutor for the {course} program. Answer about {mod_display}.{suffix}"
    msgs = [{"role": "system", "content": sys_msg}, {"role": "user", "content": question}]

    b = base.chat.completions.create(model="x", messages=msgs, temperature=0.3, max_tokens=500)
    f_resp = ft.chat.completions.create(model="x", messages=msgs, temperature=0.3, max_tokens=500)

    print(f"\n{'='*70}")
    print(f"  {course}/{module} [{lang}]")
    print(f"  Q: {question}")
    print(f"\n--- BASE ---\n{b.choices[0].message.content[:400]}")
    print(f"\n--- FINE-TUNED ---\n{f_resp.choices[0].message.content[:400]}")
```

---

## 11. Tips

### Language balance

After running `format_dataset.py`, check `stats.json`. If German/French are heavily underrepresented compared to English, consider oversampling them (duplicate those training examples 2-3x) to prevent the model from being biased toward English answers.

### How linked modules work

- `1_Accounting - linked (SOME parts) 7 CIWM` in CIIA → tagged as **CIIA/accounting** + **CIWM/accounting**
- `4_Financial Instruments (=AWM)` in FMT → tagged as **FMT/financial_instruments** + **AWM/financial_instruments**
- `AWM_FI_1e_FMT_FI_1.pdf` → detected as English, course tag comes from the folder (AWM), FMT link comes from the folder name `(=AWM)` or vice versa

### Adding new courses or languages

- New course: add a folder, add its abbreviation to `COURSE_MAP` and `KNOWN_COURSES` in `extract_text.py`
- New language suffix: add the letter to `LANG_SUFFIXES` in `extract_text.py` (e.g. `"i": "it"` for Italian)

### Complete file overview

```
/srv/llama/training/
├── extract_text.py
├── clean_and_chunk.py
├── generate_qa.py
├── format_dataset.py
├── finetune.py
├── export_gguf.py
├── eval_compare.py
├── pdfs/                      # your existing structure, copied as-is
│   ├── 1- CIIA_ILPIP/
│   ├── 2- AWM/
│   ├── 3- CIWM/
│   ├── 4- FMT/
│   └── 5- FMO/
├── extracted/
│   └── raw_pages.jsonl        # {course, module, lang, source, page, text}
├── cleaned/
│   └── chunks.jsonl           # tagged + chunked
├── qa-pairs/
│   └── qa_raw.jsonl           # tagged Q&A pairs in en/de/fr
├── dataset/
│   ├── train.jsonl            # 95% — course/module/lang-aware system prompts
│   ├── val.jsonl              # 5%
│   └── stats.json             # distribution per course/module/lang
└── output/
    └── lora-adapter/

/srv/llama/models/
└── qwen3-8b-finance/
    └── unsloth.Q4_K_M.gguf
```

---

*Generated for Christian's local LLM setup on Arch Linux.
RTX 4090 · AMD 7950X · 64GB DDR5 · /srv/llama/
Courses: CIIA, CIWM, AWM, FMT, FMO
Languages: English, German, French*
