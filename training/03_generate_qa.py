#!/usr/bin/env python3
"""Generate Q&A pairs from chunks using local LLM.
Course/module/language-aware prompts with formula support."""

import json
import time
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from openai import OpenAI

INPUT_FILE = Path("/srv/llama/training/cleaned/chunks.jsonl")
OUTPUT_FILE = Path("/srv/llama/training/qa-pairs/qa_raw.jsonl")
PROGRESS_FILE = Path("/srv/llama/training/qa-pairs/progress.json")
LOG_FILE = Path("/srv/llama/training/qa-pairs/generation.log")

# How often to save (every N chunks)
SAVE_EVERY = 5

client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")

# -- Display names ----------------------------------------------------------
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
- For mathematical formulas, write them in plain text notation:
  - Use * for multiplication, / for division, ^ for exponents
  - Example: "PV = C / (1 + r)^n" not LaTeX or markdown math
  - Example: "sigma_p = sqrt(w1^2 * sigma1^2 + w2^2 * sigma2^2 + 2*w1*w2*cov12)"
- Do NOT use markdown formatting (no **, no ```, no $)
- Do NOT use curly braces in your answers
- Output ONLY a valid JSON array, nothing else before or after
- Format: [{{"question": "...", "answer": "..."}}]"""


def log(msg: str):
    """Write to both console and log file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def extract_json(text: str) -> list:
    """Robustly extract a JSON array from model output."""
    text = text.strip()

    # Remove markdown code fences if present
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("["):
                text = part
                break

    # Find the JSON array boundaries
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise json.JSONDecodeError("No JSON array found", text, 0)

    json_str = text[start:end + 1]

    # Fix common issues with formula-heavy output
    # Replace literal backslashes that aren't JSON escapes
    # (models sometimes output \frac, \sigma, etc.)
    import re
    json_str = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', json_str)

    return json.loads(json_str)


def validate_pair(pair: dict) -> bool:
    """Check that a Q&A pair is usable."""
    q = pair.get("question", "").strip()
    a = pair.get("answer", "").strip()
    if not q or not a:
        return False
    if len(q) < 10 or len(a) < 20:
        return False
    # Skip pairs that are just the model refusing or being meta
    skip_phrases = ["I cannot", "I'm sorry", "As an AI", "I don't have"]
    for phrase in skip_phrases:
        if phrase.lower() in a.lower():
            return False
    return True


def generate_qa(chunk: dict) -> list:
    """Generate Q&A pairs for a single chunk."""
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
            max_tokens=3000,  # more room for formula-heavy answers
        )

        content = response.choices[0].message.content.strip()
        pairs = extract_json(content)

        valid_pairs = []
        for pair in pairs:
            if validate_pair(pair):
                pair["course"] = course
                pair["module"] = module
                pair["lang"] = lang
                pair["source"] = chunk["source"]
                pair["page"] = chunk["page"]
                valid_pairs.append(pair)

        return valid_pairs

    except json.JSONDecodeError as e:
        log(f"  JSON ERROR [{course}/{module}] {chunk['source']} p{chunk['page']}: {e}")
        # Save the failed output for debugging
        try:
            failed_dir = Path("/srv/llama/training/qa-pairs/failed")
            failed_dir.mkdir(exist_ok=True)
            failed_file = failed_dir / f"{chunk['source']}_p{chunk['page']}.txt"
            failed_file.write_text(content, encoding="utf-8")
        except:
            pass
        return []
    except Exception as e:
        log(f"  ERROR [{course}/{module}] {chunk['source']} p{chunk['page']}: {e}")
        return []


def save_progress(processed: int, total_pairs: int, errors: int):
    """Save progress so we can resume after interruption."""
    with open(PROGRESS_FILE, "w") as f:
        json.dump({
            "processed": processed,
            "total_pairs": total_pairs,
            "errors": errors,
            "timestamp": datetime.now().isoformat(),
        }, f)


def load_progress() -> dict:
    """Load progress from previous run."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"processed": 0, "total_pairs": 0, "errors": 0}


# -- Main -------------------------------------------------------------------
def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    Path("/srv/llama/training/qa-pairs/failed").mkdir(exist_ok=True)

    # Load all chunks
    chunks = []
    with open(INPUT_FILE) as f:
        chunks = [json.loads(line) for line in f]

    total_chunks = len(chunks)
    log(f"Loaded {total_chunks} chunks from {INPUT_FILE}")

    # Check for previous progress
    progress = load_progress()
    start_from = progress["processed"]
    total_pairs = progress["total_pairs"]
    errors = progress["errors"]

    if start_from > 0:
        log(f"Resuming from chunk {start_from} ({total_pairs} pairs, {errors} errors so far)")
        response = input(f"  Resume from chunk {start_from}? [Y/n/restart]: ").strip().lower()
        if response == "n":
            log("Aborted by user")
            return
        elif response == "restart":
            start_from = 0
            total_pairs = 0
            errors = 0
            log("Restarting from scratch")

    # Open output file in append mode (or write mode if restarting)
    mode = "a" if start_from > 0 else "w"
    start_time = time.time()

    with open(OUTPUT_FILE, mode, encoding="utf-8") as fout:
        for i in range(start_from, total_chunks):
            chunk = chunks[i]
            chunk_id = f"[{i+1}/{total_chunks}]"
            course_mod = f"{chunk['course']}/{chunk['module']}"
            lang = chunk.get('lang', 'en')

            pairs = generate_qa(chunk)

            if pairs:
                for p in pairs:
                    fout.write(json.dumps(p, ensure_ascii=False) + "\n")
                total_pairs += len(pairs)
                log(f"  {chunk_id} {course_mod} [{lang}] {chunk['source']} p{chunk['page']} -> {len(pairs)} pairs (total: {total_pairs})")
            else:
                errors += 1
                log(f"  {chunk_id} {course_mod} [{lang}] {chunk['source']} p{chunk['page']} -> FAILED (errors: {errors})")

            # Flush and save progress periodically
            if (i + 1) % SAVE_EVERY == 0:
                fout.flush()
                save_progress(i + 1, total_pairs, errors)

                # ETA calculation
                elapsed = time.time() - start_time
                chunks_done = (i + 1) - start_from
                if chunks_done > 0:
                    avg_time = elapsed / chunks_done
                    remaining = (total_chunks - i - 1) * avg_time
                    eta = timedelta(seconds=int(remaining))
                    rate = chunks_done / elapsed * 60  # chunks per minute
                    log(f"  --- Progress: {i+1}/{total_chunks} | {total_pairs} pairs | {errors} errors | {rate:.1f} chunks/min | ETA: {eta} ---")

            time.sleep(0.1)

    # Final save
    save_progress(total_chunks, total_pairs, errors)

    elapsed = time.time() - start_time
    elapsed_fmt = str(timedelta(seconds=int(elapsed)))

    log(f"\n{'='*60}")
    log(f"DONE")
    log(f"  Chunks processed: {total_chunks}")
    log(f"  Q&A pairs generated: {total_pairs}")
    log(f"  Errors: {errors} ({100*errors/max(total_chunks,1):.1f}%)")
    log(f"  Time: {elapsed_fmt}")
    log(f"  Output: {OUTPUT_FILE}")
    log(f"  Log: {LOG_FILE}")
    if errors > 0:
        log(f"  Failed outputs saved in: /srv/llama/training/qa-pairs/failed/")
    log(f"{'='*60}")


if __name__ == "__main__":
    main()