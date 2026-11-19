import json
import sys
import time
from pathlib import Path

import ollama

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from config import MODEL_AUGMENT, OLLAMA_HOST  # noqa: E402

PKD_FILE = ROOT / "pkd.json"
OUTPUT_FILE = ROOT / "pkd_augmented.json"
MAX_ATTEMPTS = 3

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

client = ollama.Client(host=OLLAMA_HOST)


def validate_extra(extra) -> dict:
    """Sprawdza, że keywords/examples to niepuste listy stringów. Zwraca tylko te dwa pola."""
    if not isinstance(extra, dict):
        raise ValueError(f"odpowiedź nie jest obiektem JSON ({type(extra).__name__})")
    cleaned = {}
    for field in ("keywords", "examples"):
        value = extra.get(field)
        if not isinstance(value, list):
            raise ValueError(f"'{field}' nie jest listą ({type(value).__name__})")
        items = [s.strip() for s in value if isinstance(s, str) and s.strip()]
        if not items:
            raise ValueError(f"'{field}' nie zawiera żadnego tekstu")
        cleaned[field] = items
    return cleaned


def augment(pkd: dict) -> dict:
    prompt = PROMPT_TEMPLATE.format(code=pkd["code"], description=pkd["description"])
    response = client.chat(
        model=MODEL_AUGMENT,
        messages=[{"role": "user", "content": prompt}],
        format="json",
    )
    extra = validate_extra(json.loads(response.message.content))
    return {**pkd, **extra}


def main():
    with open(PKD_FILE, encoding="utf-8") as f:
        pkd_data = json.load(f)

    print(f"Augmentuję {len(pkd_data)} kodów PKD przez Ollama ({MODEL_AUGMENT})...")

    results, failed = [], []
    for i, pkd in enumerate(pkd_data):
        prefix = f"[{i+1}/{len(pkd_data)}] {pkd['code']}"
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                results.append(augment(pkd))
                print(f"{prefix} ✓")
                break
            except Exception as e:
                print(f"{prefix} próba {attempt}/{MAX_ATTEMPTS} — BŁĄD: {e}")
                time.sleep(0.5)
        else:
            failed.append(pkd["code"])
            results.append(pkd)  # zostaje sam oficjalny opis, bez keywords/examples

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nZapisano do {OUTPUT_FILE}")

    if failed:
        print(f"\nUWAGA: {len(failed)} kodów bez augmentacji (tylko oficjalny opis): {', '.join(failed)}")
        print("Uruchom skrypt ponownie albo uzupełnij je ręcznie przed indeksowaniem.")
        sys.exit(1)


if __name__ == "__main__":
    main()
