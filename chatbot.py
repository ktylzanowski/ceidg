import json
import os
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

load_dotenv()

COLLECTION_NAME = "pkd_codes"
MODEL_EMBED = "BAAI/bge-m3"
MODEL_LLM = "llama-3.3-70b-versatile"

groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

QUESTIONS = [
    ("main_description", "Opisz w kilku słowach, czym chcesz się zajmować w firmie?"),
]

print("Ładuję model embeddingowy...")
embedder = SentenceTransformer(MODEL_EMBED)
qdrant = QdrantClient(host="localhost", port=6333)
print("Gotowy.\n")


def build_query(answers: dict) -> str:
    return f"Działalność: {answers['main_description']}"


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


SYSTEM_PROMPT = """Jesteś ekspertem ds. klasyfikacji PKD (Polska Klasyfikacja Działalności). Pomagasz przedsiębiorcom dobrać właściwe kody PKD do ich działalności.

Zasady uzasadnień:
- Wyjaśniaj KONKRETNIE, którą część opisanej działalności użytkownika dany kod obejmuje.
- Nie parafrazuj nazwy kodu ani nie pisz że jest "podobny do głównego kodu".
- Pisz po polsku, zwięźle (1-2 zdania).

Przykład poprawnego uzasadnienia:
Działalność: "projektuję strony internetowe i aplikacje mobilne dla firm"
Kod główny 62.01.Z: "Tworzenie oprogramowania — obejmuje bezpośrednio projektowanie i kodowanie aplikacji mobilnych oraz webowych na zlecenie klientów."
Kod dodatkowy 62.02.Z (confidence 0.85): "Doradztwo w zakresie informatyki — obejmuje doradztwo technologiczne przy wyborze technologii i architektury rozwiązań dla klientów firmowych."

Odpowiadaj wyłącznie w formacie JSON."""

def ask_llm(answers: dict, candidates: list[dict]) -> dict:
    filtered = [c for c in candidates if c["score"] >= 0.6]

    candidates_text = "\n".join(
        f"- {c['code']}: {c['description']} (score: {c['score']})"
        for c in filtered
    )

    user_prompt = f"""Działalność użytkownika: {answers['main_description']}
Uwaga: jeśli działalność to wyłącznie sprzedaż detaliczna przez internet, rozważ kod 47.91.Z.

Kandydaci z bazy PKD:
{candidates_text}

Wybierz 1 kod główny i tyle kodów dodatkowych ile faktycznie pasuje do działalności. Dla każdego kodu dodatkowego podaj confidence (0.0–1.0).

Odpowiedz w formacie JSON:
{{
  "main": {{"code": "XX.XX.X", "description": "...", "justification": "..."}},
  "additional": [
    {{"code": "XX.XX.X", "description": "...", "justification": "...", "confidence": 0.0}}
  ]
}}"""

    response = groq_client.chat.completions.create(
        model=MODEL_LLM,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )

    result = json.loads(response.choices[0].message.content)
    result["additional"] = [
        c for c in result.get("additional", [])
        if c.get("confidence", 0) >= 0.7
    ]
    return result


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
