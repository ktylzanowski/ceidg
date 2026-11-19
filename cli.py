import asyncio

from recommender import Recommender, build_query

QUESTIONS = [
    ("main_description", "Opisz w kilku słowach, czym chcesz się zajmować w firmie?"),
]


def print_result(result: dict):
    print("\n" + "=" * 50)
    print("WYNIKI PKD")
    print("=" * 50)

    if "error" in result:
        print(f"\n{result['error']}")
        print("=" * 50 + "\n")
        return

    if result.get("low_confidence"):
        print("\nUWAGA: niepewne dopasowanie — żaden kod nie pasuje wyraźnie do opisu.")

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


async def main():
    print("=" * 50)
    print("Chatbot PKD/CEIDG")
    print("=" * 50 + "\n")

    print("Ładuję model embeddingowy...")
    recommender = Recommender()
    print("Gotowy.\n")

    try:
        answers = {}
        for key, question in QUESTIONS:
            print(f">>> {question}")
            answers[key] = input("Twoja odpowiedź: ").strip()
            print()

        print("Szukam dopasowań w bazie PKD...")
        candidates = await recommender.search_pkd(build_query(answers))
        print(f"Znaleziono {len(candidates)} kandydatów. Analizuję przez LLM...")

        try:
            result = await recommender.ask_llm(answers, candidates)
        except ValueError as e:
            print(f"\nModel zwrócił niepoprawną odpowiedź: {e}")
            return
        print_result(result)
    finally:
        await recommender.close()


if __name__ == "__main__":
    asyncio.run(main())
