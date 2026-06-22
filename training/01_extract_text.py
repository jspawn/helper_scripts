#!/usr/bin/env python3
"""Extract text from PDFs, auto-parsing course/module/language/links from folder and file names.

Expected layout:
    pdfs/<course_folder>/<year>/<module_folder>/<files>.pdf

Course parsing:
    "1- CIIA_ILPIP"  ->  CIIA
    "2- AWM"         ->  AWM
    "3- CIWM"        ->  CIWM
    "4- FMT"         ->  FMT
    "5- FMO"         ->  FMO

Module parsing:
    "1_Accounting - linked (SOME parts) 7 CIWM"  ->  module: accounting, also_for: [CIWM]
    "4_Financial Instruments (=AWM)"              ->  module: financial_instruments, also_for: [AWM]

Language detection (from AWM-style filenames):
    AWM_FI_1d.pdf          ->  de (German)
    AWM_FI_1e_FMT_FI_1.pdf ->  en (English)
    AWM_FI_1f.pdf          ->  fr (French)
"""

import json
import re
import fitz  # pymupdf
from pathlib import Path

PDF_DIR = Path("/srv/llama/training/pdfs")
OUTPUT_FILE = Path("/srv/llama/training/extracted/raw_pages.jsonl")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# -- Course name mapping ---------------------------------------------------
COURSE_MAP = {
    "CIIA":  "CIIA",
    "ILPIP": "CIIA",
    "AWM":   "AWM",
    "CIWM":  "CIWM",
    "FMT":   "FMT",
    "FMO":   "FMO",
}

KNOWN_COURSES = {"CIIA", "AWM", "CIWM", "FMT", "FMO"}

# -- Language suffix map ----------------------------------------------------
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
    # \u2013 = en-dash, matching both hyphen and en-dash
    link_match = re.search(r'[-\u2013]\s*linked.*?(\d\s+)?([A-Z]{2,})', name, re.IGNORECASE)
    if link_match:
        candidate = link_match.group(2).upper()
        if candidate in KNOWN_COURSES:
            linked.append(candidate)
        name = re.sub(r'\s*[-\u2013]\s*linked.*$', '', name, flags=re.IGNORECASE)

    # Remove leading number + separator
    name = re.sub(r'^\d+[_\s]+', '', name)

    # Normalize to snake_case
    # Keep German/French accented chars so they survive in module names
    name = name.strip().lower()
    name = re.sub(r'[^a-z0-9\u00e4\u00f6\u00fc\u00e9\u00e8\u00e0]+', '_', name)
    name = name.strip('_')

    return name, list(set(linked))


def detect_language(filename: str) -> str:
    """Detect language from filename suffix.

    AWM_FI_1d.pdf          -> de
    AWM_FI_1e_FMT_FI_1.pdf -> en
    AWM_FI_1f.pdf          -> fr

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


def find_pdf_tags(pdf_path: Path) -> tuple[list[dict], str]:
    """Determine course, module, language, and cross-references for a PDF.

    Returns: (tags_list, language)
    """
    rel = pdf_path.relative_to(PDF_DIR)
    parts = rel.parts

    lang = detect_language(pdf_path.name)

    if len(parts) < 3:
        return [{"course": "UNKNOWN", "module": "general"}], lang

    # Course from first directory
    primary_course = parse_course(parts[0])

    # Module from first non-year subdirectory
    module_folder = None
    for part in parts[1:-1]:
        if not is_year_folder(part):
            module_folder = part
            break

    if module_folder is None:
        return [{"course": primary_course, "module": "general"}], lang

    module_name, linked_courses = parse_module(module_folder)

    # Build tag list
    tags = [{"course": primary_course, "module": module_name}]
    for linked in linked_courses:
        if linked != primary_course:
            tags.append({"course": linked, "module": module_name})

    return tags, lang


# -- Main extraction --------------------------------------------------------
total_pages = 0
total_docs = 0
all_tags_seen = set()
lang_counts = {"en": 0, "de": 0, "fr": 0}

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for pdf_path in sorted(PDF_DIR.rglob("*.pdf")):
        tags, lang = find_pdf_tags(pdf_path)

        rel_path = pdf_path.relative_to(PDF_DIR)
        tag_str = ", ".join(f"{t['course']}/{t['module']}" for t in tags)
        print(f"{rel_path}")
        print(f"  -> {tag_str} [{lang}]")

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