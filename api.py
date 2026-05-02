from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from chatbot import ask_llm, build_query, search_pkd

app = FastAPI(title="PKD Recommender")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)


class RecommendRequest(BaseModel):
    main_description: str


@app.post("/recommend")
@app.post("/api/recommend")
async def recommend(req: RecommendRequest):
    if not req.main_description.strip():
        raise HTTPException(status_code=400, detail="main_description nie może być pusty")

    answers = {"main_description": req.main_description}
    candidates = await search_pkd(build_query(answers))
    result = await ask_llm(answers, candidates)
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])
    return result
