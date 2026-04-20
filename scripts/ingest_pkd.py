import json
import time
from pathlib import Path

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

PKD_FILE = Path(__file__).parent.parent / "pkd.json"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "pkd_codes"
MODEL_NAME = "BAAI/bge-m3"
BATCH_SIZE = 64


def load_pkd(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_embed_text(pkd: dict) -> str:
    return f"{pkd['code']} {pkd['description']}"


def main():
    print(f"Wczytuję {PKD_FILE} ...")
    pkd_data = load_pkd(PKD_FILE)
    print(f"  Znaleziono {len(pkd_data)} kodów PKD.")
    print(f"\nŁaduję model: {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)
    print("  Model gotowy.")

    print(f"\nGeneruję embeddingi (batch_size={BATCH_SIZE}) ...")
    texts = [build_embed_text(pkd) for pkd in pkd_data]

    t0 = time.time()
    vectors = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    print(f"  Embeddingi gotowe ({time.time() - t0:.1f}s). Wymiar: {vectors.shape[1]}")

    print(f"\nŁączę z Qdrant ({QDRANT_HOST}:{QDRANT_PORT}) ...")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        print(f"  Kolekcja '{COLLECTION_NAME}' już istnieje — usuwam i tworzę od nowa.")
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=vectors.shape[1],
            distance=Distance.COSINE,
        ),
    )
    print(f"  Kolekcja '{COLLECTION_NAME}' utworzona.")

    print(f"\nIndeksuję {len(pkd_data)} punktów w Qdrant ...")
    points = [
        PointStruct(
            id=i,
            vector=vectors[i].tolist(),
            payload={
                "code": pkd["code"],
                "description": pkd["description"],
            },
        )
        for i, pkd in enumerate(pkd_data)
    ]

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"  Zaindeksowano {len(points)} punktów.")

    print("\nTest wyszukiwania: 'sklep internetowy sprzedaż online'")
    test_vector = model.encode(
        "sklep internetowy sprzedaż online",
        normalize_embeddings=True,
    ).tolist()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=test_vector,
        limit=5,
    ).points
    for r in results:
        print(f"  [{r.score:.3f}] {r.payload['code']} — {r.payload['description']}")

    print("\nGotowe!")


if __name__ == "__main__":
    main()
