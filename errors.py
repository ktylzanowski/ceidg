"""Mapowanie wyjątków z Qdranta/Ollamy/LLM na komunikaty dla użytkownika — wspólne dla API i CLI."""

from typing import NamedTuple, Optional

import httpx
from ollama import ResponseError
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

INVALID_DESCRIPTION_MESSAGE = (
    "Podany opis nie opisuje działalności gospodarczej. Opisz konkretnie czym chcesz się zajmować."
)


class AppError(NamedTuple):
    status: int
    code: str
    message: str


def classify_error(e: BaseException) -> Optional[AppError]:
    """Zwraca opis błędu dla znanych wyjątków, None dla nieznanych (te mają polecieć dalej)."""
    if isinstance(e, (ResponseHandlingException, UnexpectedResponse)):
        return AppError(503, "search_unavailable", "Baza kodów PKD jest chwilowo niedostępna.")
    if isinstance(e, ConnectionError):
        return AppError(503, "llm_unavailable", "Model językowy jest chwilowo niedostępny.")
    if isinstance(e, httpx.TimeoutException):
        return AppError(504, "llm_timeout", "Model językowy nie odpowiedział w wymaganym czasie.")
    if isinstance(e, ResponseError):
        if e.status_code == 404:
            return AppError(503, "llm_unavailable", "Model językowy nie jest zainstalowany na serwerze.")
        return AppError(502, "llm_error", "Model językowy zwrócił błąd.")
    if isinstance(e, ValueError):  # również json.JSONDecodeError i błędy walidacji odpowiedzi LLM
        return AppError(502, "llm_bad_response", "Model językowy zwrócił niepoprawną odpowiedź.")
    return None
