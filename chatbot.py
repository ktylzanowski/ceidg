"""
Prosty chatbot PKD — CLI.

Wymagania:
    - Qdrant uruchomiony (docker compose up -d)
    - Ollama uruchomiony z modelem: ollama pull llama3.1:8b
    - Zaindeksowane PKD: python scripts/ingest_pkd.py
"""

import json
import ollama
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# --- Konfiguracja ---
COLLECTION_NAME = "pkd_codes"
MODEL_EMBED = "BAAI/bge-m3"
MODEL_LLM = "llama3.1:8b"

QUESTIONS = [
    ("main_description", "Opisz w kilku słowach, czym chcesz się zajmować w firmie?"),
    ("type",             "Czy sprzedajesz produkty czy świadczysz usługi? (produkty/usługi/oba)"),
    ("client",           "Kto jest Twoim głównym klientem? (b2b/b2c/oba)"),
    ("channel",          "Gdzie odbywa się działalność? (online/stacjonarnie/hybryda)"),
]

# --- Inicjalizacja ---
print("Ładuję model embeddingowy...")
embedder = SentenceTransformer(MODEL_EMBED)
qdrant = QdrantClient(host="localhost", port=6333)
print("Gotowy.\n")


def build_query(answers: dict) -> str:
    parts = [f"Działalność: {answers['main_description']}"]
    if answers.get("type"):
        parts.append(answers["type"])
    if answers.get("client"):
        parts.append(answers["client"])
    if answers.get("channel"):
        parts.append(answers["channel"])
    return " | ".join(parts)


def search_pkd(query: str, top_k: int = 20) -> list[dict]:
    vector = embedder.encode(query, normalize_embeddings=True).tolist()
    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=top_k,
    ).points
    return [
        {"code": r.payload["code"], "description": r.payload["description"], "score": round(r.score, 3)}
        for r in results
    ]


def ask_llm(answers: dict, candidates: list[dict]) -> dict:
    candidates_text = "\n".join(
        f"- {c['code']}: {c['description']} (score: {c['score']})"
        for c in candidates
    )

    prompt = f"""Jesteś ekspertem PKD. Na podstawie opisu działalności wybierz najlepsze kody PKD.

Działalność użytkownika:
- Opis: {answers['main_description']}
- Typ: {answers.get('type', 'nie podano')}
- Klient: {answers.get('client', 'nie podano')}
- Kanał: {answers.get('channel', 'nie podano')}

Kandydaci z bazy PKD:
{candidates_text}

Wybierz 1 kod główny i maksymalnie 5 dodatkowych. Odpowiedz tylko w formacie JSON:
{{
  "main": {{"code": "XX.XX.X", "description": "...", "justification": "..."}},
  "additional": [
    {{"code": "XX.XX.X", "description": "...", "justification": "..."}}
  ]
}}"""

    response = ollama.chat(
        model=MODEL_LLM,
        messages=[{"role": "user", "content": prompt}],
        format="json",
    )

    return json.loads(response["message"]["content"])


def print_result(result: dict):
    print("\n" + "=" * 50)
    print("WYNIKI PKD")
    print("=" * 50)

    main = result.get("main", {})
    print(f"\nKOD GŁÓWNY:")
    print(f"  {main.get('code')} — {main.get('description')}")
    print(f"  Uzasadnienie: {main.get('justification')}")

    additional = result.get("additional", [])
    if additional:
        print(f"\nKODY DODATKOWE:")
        for pkd in additional:
            print(f"  {pkd.get('code')} — {pkd.get('description')}")
            print(f"  Uzasadnienie: {pkd.get('justification')}")

    print("=" * 50 + "\n")


def main():
    print("=" * 50)
    print("Chatbot PKD/CEIDG")
    print("=" * 50 + "\n")

    answers = {}

    for key, question in QUESTIONS:
        print(f">>> {question}")
        answer = input("Twoja odpowiedź: ").strip()
        answers[key] = answer
        print()

    print("Szukam dopasowań w bazie PKD...")
    query = build_query(answers)
    candidates = search_pkd(query)
    print(f"Znaleziono {len(candidates)} kandydatów. Analizuję przez LLM...")

    result = ask_llm(answers, candidates)
    print_result(result)


if __name__ == "__main__":
    main()
