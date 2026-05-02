# PKD Recommender

Aplikacja webowa pomagająca przedsiębiorcom dobrać odpowiednie kody PKD (Polska Klasyfikacja Działalności) na podstawie opisu działalności firmy.

## Jak działa

Użytkownik wpisuje opis swojej działalności w języku naturalnym, a system:

1. Zamienia opis na wektor embeddingow (model `BAAI/bge-m3`)
2. Wyszukuje semantycznie podobne kody PKD w bazie Qdrant
3. Wysyła kandydatów do LLM, który wybiera najlepiej pasujące kody i generuje uzasadnienia

## Stack

- **Backend:** Python, FastAPI
- **Baza wektorowa:** Qdrant (Docker)
- **Embeddingi:** `sentence-transformers` / `BAAI/bge-m3`
- **LLM:** Groq API (`llama-3.3-70b-versatile`)
- **Frontend:** Vue 3 + Vite
- **Serwer:** VPS z Nginx jako reverse proxy

## Historia projektu

Projekt zaczynałem lokalnie z **Ollama** i modelem `llama3.1:8b` uruchomionym na własnym sprzęcie. Po przeniesieniu na **VPS bez GPU** inference lokalnego modelu trwał 1–3 minuty na zapytanie, co było nieakceptowalne.

Rozwiązaniem była zamiana lokalnego LLM na **Groq API** — bezpłatny tier daje ~14 400 zapytań dziennie, a czas odpowiedzi spadł do 1–3 sekund. Model zmieniony na `llama-3.3-70b-versatile`, który przy porównywalnym czasie odpowiedzi jest znacznie dokładniejszy niż lokalny `llama3.1:8b`.

## Uruchomienie

### Wymagania

- Python 3.11+
- Node.js 18+
- Docker

### Instalacja

```bash
# Qdrant
docker compose up -d

# Backend
pip install -r requirements.txt

# Uzupełnij klucz API
cp .env.example .env
# Wpisz swój klucz z console.groq.com

# Frontend
cd frontend
npm install
npm run build
```

### Przygotowanie bazy danych PKD

Przed pierwszym uruchomieniem trzeba zaindeksować kody PKD w Qdrant. Dane źródłowe to dwa pliki JSON:

- **`pkd.json`** — surowa lista wszystkich kodów PKD z oficjalnymi nazwami w formacie `{"code": "XX.XX.X", "description": "..."}`. Punkt wyjścia do całego pipeline'u.

- **`pkd_augmented.json`** — wzbogacona wersja `pkd.json`. Każdy kod ma dodatkowo `keywords` (słowa kluczowe językiem naturalnym, synonimy, potoczne nazwy) i `examples` (przykładowe zdania opisujące działalność). Wzbogacenie zostało wygenerowane przez lokalny LLM (Ollama `llama3.1:8b`) i znacząco poprawia jakość wyszukiwania semantycznego — model embeddingowy lepiej dopasowuje opisy użytkowników do kodów PKD.

```bash
# Krok 1: wygeneruj pkd_augmented.json (wymaga lokalnego Ollama, jednorazowo)
python scripts/augment_pkd.py

# Krok 2: zaindeksuj kody w Qdrant (wymaga działającego Dockera)
python scripts/ingest_pkd.py
```

> `pkd_augmented.json` jest już dołączony do repozytorium — krok 1 można pominąć i przejść od razu do indeksowania.

### Uruchomienie API

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

## Zmienne środowiskowe

| Zmienna | Opis |
|---|---|
| `GROQ_API_KEY` | Klucz API z [console.groq.com](https://console.groq.com) |
