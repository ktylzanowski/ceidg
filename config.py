import os

from dotenv import load_dotenv

load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "pkd_codes")

MODEL_EMBED = os.getenv("MODEL_EMBED", "BAAI/bge-m3")

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "300"))
MODEL_LLM = os.getenv("MODEL_LLM", "llama3.1:8b")
# Model do jednorazowej augmentacji (scripts/augment_pkd.py) — może być większy niż MODEL_LLM
MODEL_AUGMENT = os.getenv("MODEL_AUGMENT", MODEL_LLM)
