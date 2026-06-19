from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json, os, time
from collections import defaultdict, deque
from typing import Optional
from admin import router as admin_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://qwif71.github.io"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(admin_router)

BASE_DIR = os.environ.get("BASE_DIR", "/opt/render/project/src/backend")
CORPUS_PATH = os.environ.get("CORPUS_PATH", os.path.join(BASE_DIR, "tagged.json"))

corpus = json.load(open(CORPUS_PATH))

# ---------------------------------------------------------------------------
# Search index: TF-IDF over each problem's statement + repeated technique
# names (same text tag_and_compare.embed_text used to feed to the old neural
# embedder). TF-IDF needs no large model download and rebuilds in
# milliseconds, which is what makes admin imports/deletes able to update the
# live index instantly instead of needing a slow sentence-transformers reload.
# `vectorizer` and `tfidf_matrix` are rebuilt by rebuild_search_index(),
# called once at startup and again after every admin import/delete.
# ---------------------------------------------------------------------------
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = None
tfidf_matrix = None


def _index_text(r: dict) -> str:
    """Text fed to the vectorizer for one problem. Techniques are repeated
    to weight them higher than incidental words in the statement — mirrors
    the old embed_text() convention from tag_and_compare.py."""
    techs = "; ".join(r.get("techniques", []))
    parts = [
        r.get("statement", ""),
        techs, techs,
        " ".join(r.get("area", [])),
        r.get("summary", ""),
    ]
    return " || ".join(p for p in parts if p)


def rebuild_search_index():
    """Recompute the TF-IDF matrix for the current in-memory corpus. Called
    at startup and again by admin.py after any import/delete, so the live
    index always matches the live corpus — no redeploy needed."""
    global vectorizer, tfidf_matrix
    if not corpus:
        vectorizer, tfidf_matrix = None, None
        return
    vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
    tfidf_matrix = vectorizer.fit_transform([_index_text(r) for r in corpus])


rebuild_search_index()
print(f"Ready — {len(corpus)} problems loaded.")


def keyword_score(r: dict, query: str) -> float:
    """Plain substring/word-overlap score. This is what makes a query like
    'pentagon' reliably surface every problem containing that exact word,
    even if TF-IDF alone might underweight a rare, specific term relative
    to more common words in a short query."""
    query_words = set(query.lower().split())
    if not query_words:
        return 0.0
    target = " ".join([
        " ".join(r.get("techniques", [])),
        " ".join(r.get("area", [])),
        " ".join(r.get("subtopics", [])),
        r.get("summary", ""),
        r.get("statement", ""),
    ]).lower()
    target_words = set(target.split())
    overlap = len(query_words & target_words)
    # Bonus for exact substring presence (catches multi-word phrases and
    # words TF-IDF's tokenizer might split differently).
    substring_bonus = 0.3 if query.lower() in target else 0.0
    return min(1.0, overlap / len(query_words) + substring_bonus)


# ---------------------------------------------------------------------------
# Rate limiting for public, unauthenticated endpoints. Simple in-process
# sliding window per client IP — no Redis or external service needed at this
# scale. Admin routes (everything under /admin/*, defined in admin.py) are
# NOT covered by this: they require login already, and the admin is trusted
# to import/search as much as needed. This only guards endpoints anyone on
# the internet can hit directly, like /practice, to prevent scraping or
# accidental hammering (e.g. once /practice calls a paid embeddings API down
# the line — TF-IDF itself is free/local, but this limit is cheap insurance
# either way and matters more the moment any paid call gets added per-query).
# ---------------------------------------------------------------------------
RATE_LIMIT_REQUESTS = 30       # requests allowed
RATE_LIMIT_WINDOW = 60         # per this many seconds
_request_log: dict[str, deque] = defaultdict(deque)


def _check_rate_limit(request: Request):
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    log = _request_log[ip]
    while log and now - log[0] > RATE_LIMIT_WINDOW:
        log.popleft()
    if len(log) >= RATE_LIMIT_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: max {RATE_LIMIT_REQUESTS} requests "
                    f"per {RATE_LIMIT_WINDOW}s. Try again shortly.",
        )
    log.append(now)


@app.get("/practice")
def practice(
    request: Request,
    area: Optional[str] = None,
    technique: Optional[str] = None,
    difficulty: Optional[str] = None,
    text: Optional[str] = None,
    like: Optional[str] = None,
    k: int = 5,
):
    _check_rate_limit(request)
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
        query_text = _index_text(corpus[match])
        pool_idx = [i for i in pool_idx if corpus[i]["id"] != like]
    else:
        query_text = text

    if vectorizer is None or tfidf_matrix is None:
        scores = [(i, keyword_score(corpus[i], text or query_text)) for i in pool_idx]
    else:
        qv = vectorizer.transform([query_text])
        pool_matrix = tfidf_matrix[pool_idx]
        tfidf_sims = (pool_matrix @ qv.T).toarray().flatten()
        # Blend TF-IDF similarity with plain keyword overlap, so an exact
        # word like "pentagon" still surfaces strongly even if it's rare
        # enough that TF-IDF alone underweights it relative to the rest of
        # a longer query.
        kw_query = text or query_text
        scores = []
        for j, i in enumerate(pool_idx):
            kw = keyword_score(corpus[i], kw_query)
            blended = 0.7 * float(tfidf_sims[j]) + 0.3 * kw
            scores.append((i, blended))

    scores.sort(key=lambda x: -x[1])
    results = []
    for i, score in scores[:k]:
        r = dict(corpus[i])
        r["match_pct"] = round(max(0.0, score) * 100)
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
