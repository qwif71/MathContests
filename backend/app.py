from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import json, os
from typing import Optional
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

CORPUS_PATH = os.environ.get("CORPUS_PATH", "tagged.json")
corpus = json.load(open(CORPUS_PATH))
embeddings = np.load("embeddings.npy")
print(f"Ready — {len(corpus)} problems loaded.")


def embed_query(text):
    """Embed a query string using a lightweight approach."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode([text], normalize_embeddings=True)[0]


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
        qv = embed_query(text)

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
