from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import json, os
from typing import Optional
from sentence_transformers import SentenceTransformer
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CORPUS_PATH = os.environ.get("CORPUS_PATH", "../tagged.json")
corpus = json.load(open(CORPUS_PATH))
model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_text(r):
    techs = "; ".join(r.get("techniques", []))
    return f"{r['statement']} || techniques: {techs}; {techs}"

print("Computing embeddings...")
embeddings = model.encode([embed_text(r) for r in corpus], normalize_embeddings=True)
print(f"Ready — {len(corpus)} problems loaded.")


@app.get("/practice")
def practice(
    area: Optional[str] = None,
    technique: Optional[str] = None,
    difficulty: Optional[str] = None,
    text: Optional[str] = None,
    like: Optional[str] = None,
    k: int = 5,
):
    pool_idx = list(range(len(corpus)))

    if area:
        pool_idx = [i for i in pool_idx
                    if area.lower() in [a.lower() for a in corpus[i].get("area", [])]]
    if technique:
        pool_idx = [i for i in pool_idx
                    if any(technique.lower() in t.lower()
                           for t in corpus[i].get("techniques", []))]
    if difficulty:
        pool_idx = [i for i in pool_idx
                    if corpus[i].get("difficulty") == difficulty]

    if not pool_idx:
        return {"results": [], "message": "No problems match those filters."}

    if not text and not like:
        return {"results": [corpus[i] for i in pool_idx[:k]]}

    if like:
        match = next((i for i, r in enumerate(corpus) if r["id"] == like), None)
        if match is None:
            return {"error": f"{like} not found"}
        qv = embeddings[match]
        pool_idx = [i for i in pool_idx if corpus[i]["id"] != like]
    else:
        qv = model.encode([text], normalize_embeddings=True)[0]

    pool_emb = np.array([embeddings[i] for i in pool_idx])
    sims = pool_emb @ qv
    order = np.argsort(-sims)

    results = []
    for j in order[:k]:
        r = dict(corpus[pool_idx[j]])
        r["match_pct"] = round(float(max(0, sims[j])) * 100)
        results.append(r)

    return {"results": results}


@app.get("/problem/{problem_id}")
def get_problem(problem_id: str):
    match = next((r for r in corpus if r["id"] == problem_id), None)
    if not match:
        return {"error": "Not found"}
    return match


@app.get("/stats")
def stats():
    from collections import Counter
    areas = Counter(a for r in corpus for a in r.get("area", []))
    contests = Counter(r.get("contest", "?") for r in corpus)
    return {
        "total": len(corpus),
        "by_area": dict(areas),
        "by_contest": dict(contests),
    }
