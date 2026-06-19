from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json, os, time
from collections import defaultdict, deque
from typing import Optional
from admin import router as admin_router
import settings as st

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
# AI-assisted query parsing — OFF by default (see settings.py). When enabled
# by the admin, a free-text query like "something with a clever recursive
# counting argument" gets turned into structured area/technique hints via a
# Claude tool call (reusing tag_and_compare's taxonomy), which both narrows
# the pool *and* gives the TF-IDF/keyword ranker an enriched query string
# to work with. If the call fails for any reason, we silently fall back to
# the plain query — a broken AI path should never break search itself.
# ---------------------------------------------------------------------------
QUERY_TOOL = {
    "name": "record_query_intent",
    "description": "Record the structured search intent behind a free-text practice request.",
    "input_schema": {
        "type": "object",
        "properties": {
            "areas": {"type": "array", "items": {"type": "string"},
                       "description": "0-2 areas from: Algebra, Combinatorics, Geometry, Number Theory"},
            "techniques": {"type": "array", "items": {"type": "string"},
                            "description": "0-4 specific techniques/concepts implied by the request"},
            "expanded_query": {"type": "string",
                                 "description": "The request rewritten as a dense, keyword-rich search string"},
        },
        "required": ["areas", "techniques", "expanded_query"],
    },
}


def ai_parse_query(text: str) -> Optional[dict]:
    """Returns {"areas": [...], "techniques": [...], "expanded_query": str} or
    None if AI querying is disabled or the call fails for any reason."""
    if not st.get_ai_query_enabled():
        return None
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        return None
    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=key)
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            temperature=0,
            tools=[QUERY_TOOL],
            tool_choice={"type": "tool", "name": "record_query_intent"},
            messages=[{"role": "user", "content":
                f"A student wants math practice problems. Their request: {text!r}\n\n"
                "Extract the area(s), technique(s), and an expanded keyword-rich "
                "version of the query that would help a keyword search engine."}],
        )
        for b in msg.content:
            if b.type == "tool_use" and b.name == "record_query_intent":
                return dict(b.input)
    except Exception as e:
        print(f"  ! ai_parse_query failed: {e}")
    return None


@app.get("/practice")
def practice(
    request: Request,
    area: Optional[str] = None,
    technique: Optional[str] = None,
    text: Optional[str] = None,
    like: Optional[str] = None,
    use_ai: bool = False,
    k: int = 5,
):
    _check_rate_limit(request)
    pool_idx = list(range(len(corpus)))

    # Optional AI-assisted parsing of the free-text query. Only actually runs
    # if the admin has the feature turned on (see ai_parse_query); otherwise
    # this is a no-op and behavior is identical to before.
    ai_intent = None
    if use_ai and text:
        ai_intent = ai_parse_query(text)
        if ai_intent:
            # AI-suggested area only narrows the pool if the caller didn't
            # already specify one explicitly.
            if not area and ai_intent.get("areas"):
                area = ai_intent["areas"][0]

    if area:
        pool_idx = [i for i in pool_idx
                    if area.lower() in [a.lower() for a in corpus[i].get("area", [])]]
    if technique:
        pool_idx = [i for i in pool_idx
                    if any(technique.lower() in t.lower()
                           for t in corpus[i].get("techniques", []))]

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
        # Use the AI-expanded query string for ranking if available, since it's
        # built to be keyword-dense for exactly this TF-IDF/keyword scorer.
        query_text = (ai_intent or {}).get("expanded_query") or text

    if vectorizer is None or tfidf_matrix is None:
        scores = [(i, keyword_score(corpus[i], query_text)) for i in pool_idx]
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


# ---------------------------------------------------------------------------
# Rate limiting for public, unauthenticated endpoints. Simple in-process
# sliding window per client IP — no Redis or external service needed at this
# scale. Admin routes (everything under /admin/*, defined in admin.py) are
# NOT covered by this: they require login already, and the admin is trusted
# to import/search as much as needed. This only guards endpoints anyone on
# the internet can hit directly, like /practice, to prevent scraping or
# accidental hammering (now more important than ever: the AI query path,
# when enabled, calls a paid API per request).
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
