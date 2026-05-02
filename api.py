from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from chatbot import ask_llm, build_query, search_pkd

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="PKD Recommender")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)


class RecommendRequest(BaseModel):
    main_description: str


@app.post("/api/recommend")
@limiter.limit("10/minute")
async def recommend(request: Request, req: RecommendRequest):
    if not req.main_description.strip():
        raise HTTPException(status_code=400, detail="main_description nie może być pusty")

    answers = {"main_description": req.main_description}
    candidates = await search_pkd(build_query(answers))
    result = await ask_llm(answers, candidates)
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])
    return result
