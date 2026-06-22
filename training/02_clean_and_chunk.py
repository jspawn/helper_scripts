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
    text = re.sub(r"([a-z\u00e4\u00f6\u00fc\u00e9\u00e8\u00e0,])\n([a-z\u00e4\u00f6\u00fc\u00e9\u00e8\u00e0])", r"\1 \2", text)
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

print(f"Done: {total} chunks -> {OUTPUT_FILE}")