import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request
from ollama import ResponseError
from pydantic import BaseModel, Field
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from recommender import Recommender

logger = logging.getLogger("pkd")


def _error(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"code": code, "message": message})


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.recommender = Recommender()
    yield
    await app.state.recommender.close()


limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="PKD Recommender", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


class RecommendRequest(BaseModel):
    main_description: str = Field(max_length=1000)


@app.post("/api/recommend")
@limiter.limit("10/minute")
async def recommend(request: Request, req: RecommendRequest):
    if not req.main_description.strip():
        raise HTTPException(status_code=400, detail="main_description nie może być pusty")

    recommender: Recommender = request.app.state.recommender
    try:
        result = await recommender.recommend({"main_description": req.main_description})
    except (ResponseHandlingException, UnexpectedResponse):
        logger.exception("Qdrant niedostępny")
        raise _error(503, "search_unavailable", "Baza kodów PKD jest chwilowo niedostępna.")
    except ConnectionError:
        logger.exception("Brak połączenia z Ollamą")
        raise _error(503, "llm_unavailable", "Model językowy jest chwilowo niedostępny.")
    except httpx.TimeoutException:
        logger.exception("Timeout Ollamy")
        raise _error(504, "llm_timeout", "Model językowy nie odpowiedział w wymaganym czasie.")
    except ResponseError as e:
        logger.exception("Błąd Ollamy (status %s)", e.status_code)
        if e.status_code == 404:
            raise _error(503, "llm_unavailable", "Model językowy nie jest zainstalowany na serwerze.")
        raise _error(502, "llm_error", "Model językowy zwrócił błąd.")
    except ValueError:  # również json.JSONDecodeError
        logger.exception("Niepoprawna odpowiedź LLM")
        raise _error(502, "llm_bad_response", "Model językowy zwrócił niepoprawną odpowiedź.")

    if "error" in result:
        raise _error(422, "invalid_description", result["error"])
    return result
