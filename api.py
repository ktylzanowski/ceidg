import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from errors import INVALID_DESCRIPTION_MESSAGE, classify_error
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
        raise _error(400, "empty_description", "Opis działalności nie może być pusty.")

    recommender: Recommender = request.app.state.recommender
    try:
        result = await recommender.recommend({"main_description": req.main_description})
    except Exception as e:
        err = classify_error(e)
        if err is None:
            raise
        logger.exception("Błąd przetwarzania zapytania (%s)", err.code)
        raise _error(*err)

    if "error" in result:
        raise _error(422, "invalid_description", INVALID_DESCRIPTION_MESSAGE)
    return result
