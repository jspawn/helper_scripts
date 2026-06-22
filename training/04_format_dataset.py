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


# -- Load and format --
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

# -- Shuffle and split --
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