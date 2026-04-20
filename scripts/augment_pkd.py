import json
import time
from pathlib import Path

import ollama

PKD_FILE = Path(__file__).parent.parent / "pkd.json"
OUTPUT_FILE = Path(__file__).parent.parent / "pkd_augmented.json"
MODEL = "llama3.1:8b"

PROMPT_TEMPLATE = """Jesteś ekspertem ds. działalności gospodarczej w Polsce.

Kod PKD: {code}
Oficjalny opis: {description}

Wygeneruj dla tego kodu PKD:
1. "keywords" - lista 8-12 słów kluczowych, którymi przedsiębiorca opisałby tę działalność naturalnym językiem (synonimy, potoczne nazwy, branżowe określenia)
2. "examples" - lista 3-5 przykładowych zdań opisujących konkretną działalność tego typu, tak jak powiedziałby to przyszły właściciel firmy

Odpowiedz TYLKO w formacie JSON:
{{
  "keywords": ["słowo1", "słowo2", ...],
  "examples": ["Prowadzę sklep z...", "Zajmuję się...", ...]
}}"""


def augment(pkd: dict) -> dict:
    prompt = PROMPT_TEMPLATE.format(
        code=pkd["code"],
        description=pkd["description"],
    )
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        format="json",
    )
    extra = json.loads(response["message"]["content"])
    return {**pkd, **extra}


def main():
    with open(PKD_FILE, encoding="utf-8") as f:
        pkd_data = json.load(f)

    print(f"Augmentuję {len(pkd_data)} kodów PKD przez Ollama ({MODEL})...")

    results = []
    for i, pkd in enumerate(pkd_data):
        try:
            augmented = augment(pkd)
            results.append(augmented)
            print(f"[{i+1}/{len(pkd_data)}] {pkd['code']} ✓")
        except Exception as e:
            print(f"[{i+1}/{len(pkd_data)}] {pkd['code']} BŁĄD: {e}")
            results.append(pkd)
        time.sleep(0.1)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nZapisano do {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
