"""
admin.py — developer login + web-based problem import, bolted onto app.py.

Auth: a single shared password (ADMIN_PASSWORD env var) -> a signed session
cookie. Good enough for a single-developer tool; not meant for multi-user
access control.

Import: reuses tag_and_compare.py's record_from_text/tag/embed_text and
extract.py's PDF parsers, so the web path and the CLI path share the exact
same logic — no duplicated regexes or prompts to keep in sync.

Persistence: Render's disk is ephemeral, so every successful import commits
the updated tagged.json (and embeddings.npy, if present) straight to GitHub
via the contents API. The in-memory corpus in app.py is updated in place too,
so new problems show up in /practice immediately without a redeploy.

Settings: a small admin-only toggle for whether /practice is allowed to call
the Anthropic API to parse free-text queries (see settings.py + app.py's
ai_parse_query). Defaults OFF; only an authenticated admin can flip it.
"""
import base64
import json
import os
import secrets
import sys
import time
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired

from fastapi import APIRouter, Cookie, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse
import requests

import tag_and_compare as tc
import settings as st
import amio_import

router = APIRouter()

ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "").strip()
SESSION_SECRET = os.environ.get("SESSION_SECRET", "").strip() or secrets.token_hex(32)
SESSION_MAX_AGE = 60 * 60 * 12  # 12 hours

# Render always serves over HTTPS, so this is True in production. Set
# COOKIE_SECURE=false in your local environment if you ever run the backend
# itself locally over plain http:// and need the cookie to be accepted.
COOKIE_SECURE = os.environ.get("COOKIE_SECURE", "true").strip().lower() != "false"
# "none" is required for the cookie to be sent on cross-origin fetches (the
# frontend and backend are on different domains). Cross-site cookies require
# Secure, so this only actually works when COOKIE_SECURE is also true —
# i.e. in production over HTTPS. That's fine: it's exactly the case we need
# to support, and local same-origin testing doesn't depend on this anyway.
COOKIE_SAMESITE = "none" if COOKIE_SECURE else "lax"

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "").strip()
GITHUB_REPO = os.environ.get("GITHUB_REPO", "").strip()  # "owner/repo"
GITHUB_BRANCH = os.environ.get("GITHUB_BRANCH", "main").strip()
CORPUS_REPO_PATH = os.environ.get("CORPUS_REPO_PATH", "backend/tagged.json").strip()

# AIMO source CSV lives committed in the repo (it ships with every deploy,
# same as tagged.json) rather than being uploaded through the admin UI each
# session. Default path matches "Option B" from the data-placement discussion:
# backend/data/amio_raw.csv.
AMIO_CSV_PATH = os.environ.get(
    "AMIO_CSV_PATH",
    os.path.join(os.environ.get("BASE_DIR", "/opt/render/project/src/backend"),
                 "data", "amio_raw.csv"),
)

serializer = URLSafeTimedSerializer(SESSION_SECRET, salt="admin-session")
COOKIE_NAME = "admin_session"


# --------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------

def _make_session_token() -> str:
    return serializer.dumps({"ok": True, "iat": time.time()})


def require_admin(admin_session: str | None = Cookie(default=None)) -> None:
    """FastAPI dependency: raises 401 unless a valid session cookie is present."""
    if not ADMIN_PASSWORD:
        raise HTTPException(500, "ADMIN_PASSWORD is not configured on the server.")
    if not admin_session:
        raise HTTPException(401, "Not logged in.")
    try:
        serializer.loads(admin_session, max_age=SESSION_MAX_AGE)
    except SignatureExpired:
        raise HTTPException(401, "Session expired, please log in again.")
    except BadSignature:
        raise HTTPException(401, "Invalid session.")


@router.post("/admin/login")
async def admin_login(request: Request):
    if not ADMIN_PASSWORD:
        raise HTTPException(500, "ADMIN_PASSWORD is not configured on the server.")
    body = await request.json()
    password = (body or {}).get("password", "")
    if not secrets.compare_digest(password, ADMIN_PASSWORD):
        raise HTTPException(401, "Wrong password.")
    token = _make_session_token()
    resp = JSONResponse({"ok": True})
    resp.set_cookie(
        COOKIE_NAME, token,
        max_age=SESSION_MAX_AGE, httponly=True, path="/",
        samesite=COOKIE_SAMESITE, secure=COOKIE_SECURE,
)
    return resp


@router.post("/admin/logout")
async def admin_logout():
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(
        COOKIE_NAME, samesite=COOKIE_SAMESITE, secure=COOKIE_SECURE, path="/",
    )
    return resp

@router.get("/admin/me")
async def admin_me(admin_session: str | None = Cookie(default=None)):
    try:
        require_admin(admin_session)
    except HTTPException:
        return {"logged_in": False}
    return {"logged_in": True}


# --------------------------------------------------------------------------
# Settings — currently just the AI-assisted query toggle.
# --------------------------------------------------------------------------

@router.get("/admin/settings")
async def get_settings(admin_session: str | None = Cookie(default=None)):
    require_admin(admin_session)
    return {"ai_query_enabled": st.get_ai_query_enabled()}


@router.post("/admin/settings")
async def update_settings(
    request: Request,
    admin_session: str | None = Cookie(default=None),
):
    """Body: {ai_query_enabled: bool}. Persists locally and commits to GitHub
    (best-effort) so the flag survives a redeploy on Render's ephemeral disk."""
    require_admin(admin_session)
    body = await request.json()
    if "ai_query_enabled" not in body:
        raise HTTPException(400, "ai_query_enabled is required.")
    github_result = st.set_ai_query_enabled(bool(body["ai_query_enabled"]))
    return {
        "ai_query_enabled": st.get_ai_query_enabled(),
        "github": github_result,
    }


# --------------------------------------------------------------------------
# Browse / delete / edit — admin-only views of the full corpus.
# --------------------------------------------------------------------------

@router.get("/admin/problems")
async def list_problems(admin_session: str | None = Cookie(default=None)):
    """Name-labels-only listing for scrolling through the whole catalogue in
    the admin UI — just enough to identify each row (id, contest, number,
    area) without shipping every statement/solution over the wire."""
    require_admin(admin_session)
    import app as appmod
    return {
        "problems": [
            {
                "id": r["id"],
                "contest": r.get("contest", ""),
                "number": r.get("number"),
                "area": r.get("area", []),
            }
            for r in appmod.corpus
        ]
    }


@router.get("/admin/problems/{problem_id}")
async def get_problem_full(problem_id: str, admin_session: str | None = Cookie(default=None)):
    """Full record for the edit form — every field, since faulty LaTeX or
    tags can show up anywhere (statement, solution, answer, techniques...)."""
    require_admin(admin_session)
    import app as appmod
    match = next((r for r in appmod.corpus if r["id"] == problem_id), None)
    if not match:
        raise HTTPException(404, "Problem not found.")
    return match


@router.put("/admin/problems/{problem_id}")
async def update_problem(
    problem_id: str,
    request: Request,
    admin_session: str | None = Cookie(default=None),
):
    """Body: a full or partial record — any field may be edited, including
    statement/solution/answer (to fix bad LaTeX from extraction) and the
    tag fields (area/subtopics/techniques/summary). Fields omitted from the
    body are left untouched. Recomputes the search index and commits."""
    require_admin(admin_session)
    import app as appmod

    idx = next((i for i, r in enumerate(appmod.corpus) if r["id"] == problem_id), None)
    if idx is None:
        raise HTTPException(404, "Problem not found.")

    body = await request.json()
    if not isinstance(body, dict) or not body:
        raise HTTPException(400, "Request body must be a non-empty object of fields to update.")

    # id is the corpus key everywhere else (search results, /problem/{id},
    # GitHub diffs); changing it here would silently orphan the old id from
    # any external links, so it's edited via a separate explicit step if
    # ever needed, not through this general-purpose update.
    body.pop("id", None)
    appmod.corpus[idx].update(body)

    appmod.rebuild_search_index()
    try:
        json.dump(appmod.corpus, open(appmod.CORPUS_PATH, "w"), indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"  ! local corpus write failed: {e}", file=sys.stderr)

    github_result = commit_corpus_to_github(
        appmod.corpus, message=f"Edit {problem_id}")
    return {"ok": True, "record": appmod.corpus[idx], "github": github_result}


@router.delete("/admin/problems/{problem_id}")
async def delete_problem(problem_id: str, admin_session: str | None = Cookie(default=None)):
    require_admin(admin_session)
    import app as appmod

    idx = next((i for i, r in enumerate(appmod.corpus) if r["id"] == problem_id), None)
    if idx is None:
        raise HTTPException(404, "Problem not found.")

    appmod.corpus.pop(idx)
    appmod.rebuild_search_index()
    try:
        json.dump(appmod.corpus, open(appmod.CORPUS_PATH, "w"), indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"  ! local corpus write failed: {e}", file=sys.stderr)

    github_result = commit_corpus_to_github(
        appmod.corpus, message=f"Delete {problem_id}")
    return {"ok": True, "total_corpus_size": len(appmod.corpus), "github": github_result}


# --------------------------------------------------------------------------
# One-time migration: old AMIO-prefixed ids -> the systematic scheme
# amio_import.py now generates for everything.
#
#   Old:  AMIO-AMC10A-2023-P1     (AMIO-{variant}-{year}-P{number})
#   New:  AMC2023-10A-INDIVIDUAL-1  (matches amio_import._contest_slug)
#
# This is a dry-run-by-default endpoint: GET previews the rename mapping
# without touching anything; POST actually applies it (and refuses to run
# if it would create a collision with an existing id).
# --------------------------------------------------------------------------

import re as _re

_OLD_AMIO_ID = _re.compile(
    r"^AMIO-AMC(?P<variant>\d+[AB]?)-(?P<year>\d{4})-P(?P<number>\d+)$", _re.IGNORECASE
)


def _migrated_id(old_id: str) -> str | None:
    """Returns the new-scheme id for an old AMIO-... id, or None if old_id
    doesn't match the expected old pattern (left untouched in that case)."""
    m = _OLD_AMIO_ID.match(old_id)
    if not m:
        return None
    return f"AMC{m.group('year')}-{m.group('variant').upper()}-INDIVIDUAL-{int(m.group('number'))}"


def _build_migration_plan(corpus: list) -> tuple[list[dict], list[str]]:
    """Returns (renames, collisions) where renames is a list of
    {old_id, new_id} and collisions is old_ids that would collide with an
    existing id (and are therefore excluded from the plan)."""
    existing_ids = {r["id"] for r in corpus}
    renames, collisions = [], []
    for r in corpus:
        new_id = _migrated_id(r["id"])
        if new_id is None:
            continue
        if new_id == r["id"]:
            continue
        if new_id in existing_ids:
            collisions.append(r["id"])
            continue
        renames.append({"old_id": r["id"], "new_id": new_id})
    return renames, collisions


@router.get("/admin/migrate-amio-ids")
async def preview_amio_id_migration(admin_session: str | None = Cookie(default=None)):
    """Dry run: shows exactly what would be renamed, with no side effects."""
    require_admin(admin_session)
    import app as appmod
    renames, collisions = _build_migration_plan(appmod.corpus)
    return {
        "would_rename": len(renames),
        "collisions_skipped": collisions,
        "renames": renames,
    }


@router.post("/admin/migrate-amio-ids")
async def apply_amio_id_migration(admin_session: str | None = Cookie(default=None)):
    """Applies the migration: rewrites every old AMIO-... id to the new
    systematic scheme, recomputes the search index, persists, and commits.
    Safe to call more than once — already-migrated ids simply produce no
    matches the second time (the regex only matches the old pattern)."""
    require_admin(admin_session)
    import app as appmod

    renames, collisions = _build_migration_plan(appmod.corpus)
    if not renames:
        return {"renamed": 0, "collisions_skipped": collisions,
                "message": "Nothing to migrate."}

    rename_map = {r["old_id"]: r["new_id"] for r in renames}
    for rec in appmod.corpus:
        if rec["id"] in rename_map:
            rec["id"] = rename_map[rec["id"]]

    appmod.rebuild_search_index()
    try:
        json.dump(appmod.corpus, open(appmod.CORPUS_PATH, "w"), indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"  ! local corpus write failed: {e}", file=sys.stderr)

    github_result = commit_corpus_to_github(
        appmod.corpus,
        message=f"Migrate {len(renames)} AMIO-prefixed id(s) to the systematic scheme",
    )
    return {
        "renamed": len(renames),
        "collisions_skipped": collisions,
        "renames": renames,
        "github": github_result,
    }


# --------------------------------------------------------------------------
# GitHub auto-commit
# --------------------------------------------------------------------------

def _github_headers():
    return {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }


def commit_corpus_to_github(corpus: list, message: str) -> dict:
    """PUT the updated tagged.json to GitHub via the contents API.
    Requires GITHUB_TOKEN (a repo-scoped PAT) and GITHUB_REPO ('owner/repo')."""
    if not GITHUB_TOKEN or not GITHUB_REPO:
        return {"committed": False, "reason": "GITHUB_TOKEN/GITHUB_REPO not configured"}

    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{CORPUS_REPO_PATH}"
    # Need the current file's sha to update it.
    get_resp = requests.get(url, headers=_github_headers(),
                             params={"ref": GITHUB_BRANCH}, timeout=20)
    sha = get_resp.json().get("sha") if get_resp.status_code == 200 else None

    content_str = json.dumps(corpus, indent=2, ensure_ascii=False)
    payload = {
        "message": message,
        "content": base64.b64encode(content_str.encode("utf-8")).decode("ascii"),
        "branch": GITHUB_BRANCH,
    }
    if sha:
        payload["sha"] = sha

    put_resp = requests.put(url, headers=_github_headers(), json=payload, timeout=30)
    if put_resp.status_code not in (200, 201):
        return {"committed": False, "reason": put_resp.text[:500]}
    return {"committed": True, "commit_sha": put_resp.json().get("commit", {}).get("sha")}


# --------------------------------------------------------------------------
# Import: paste text
# --------------------------------------------------------------------------

@router.post("/admin/import/text")
async def import_text(
    request: Request,
    admin_session: str | None = Cookie(default=None),
):
    """Body: {text, id, contest, round, model?}
    text follows the same STATEMENT:/SOLUTION: convention as tag_and_compare.py's
    record_from_text. Tags via the Anthropic API, appends to the in-memory
    corpus, recomputes that one embedding, and commits to GitHub."""
    require_admin(admin_session)
    body = await request.json()
    text = (body.get("text") or "").strip()
    pid = (body.get("id") or "").strip()
    contest = (body.get("contest") or "").strip()
    round_name = (body.get("round") or "").strip()
    model = (body.get("model") or tc.MODEL).strip()

    if not text:
        raise HTTPException(400, "No problem text provided.")
    if not pid:
        raise HTTPException(400, "An id is required, e.g. CONTEST2026-INDIVIDUAL-1")

    record = tc.record_from_text(text, pid)
    if contest:
        record["contest"] = contest
    if round_name:
        record["round"] = round_name
    record.setdefault("sources", [contest or "manual import"])

    tagged = tc.tag([record], model)[0]
    return _finish_import([tagged])


# --------------------------------------------------------------------------
# Import: PDF upload (uses extract.py's parsers)
# --------------------------------------------------------------------------

@router.post("/admin/import/pdf")
async def import_pdf(
    file: UploadFile = File(...),
    contest: str = Form(...),
    fmt: str = Form(...),          # "arml" | "mmaths" | "amc" | "llm"
    round_name: str = Form("Individual Round"),
    model: str = Form(tc.MODEL),
    admin_session: str | None = Cookie(default=None),
):
    require_admin(admin_session)
    import extract as ex
    import tempfile

    if fmt not in ("arml", "mmaths", "amc", "llm"):
        raise HTTPException(400, "fmt must be one of arml, mmaths, amc, llm")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        if fmt == "mmaths":
            records = ex.extract_mmaths(tmp_path, contest, round_name)
        elif fmt == "amc":
            records = ex.extract_amc(tmp_path, contest, round_name)
        elif fmt == "llm":
            records = ex.extract_llm(tmp_path, contest, round_name, model)
        elif round_name.lower() == "all":
            records = ex.extract_all(tmp_path, contest)
        else:
            records = ex.extract_round(tmp_path, round_name, contest)
    finally:
        os.unlink(tmp_path)

    if not records:
        raise HTTPException(422, "No problems were extracted from this PDF.")

    for r in records:
        r.setdefault("sources", [contest])

    tc.add_difficulty(records)
    tagged = tc.tag(records, model)
    return _finish_import(tagged)


# --------------------------------------------------------------------------
# AIMO CSV import — the CSV lives committed in the repo at AMIO_CSV_PATH
# (backend/data/amio_raw.csv by default), so it ships with every deploy and
# doesn't need re-uploading each admin session. "Reload" re-reads it from
# disk (useful after you commit an updated CSV); the parsed result is cached
# in memory so repeated contest imports don't re-parse the whole file.
#
# Import remains contest-by-contest (one button per contest in the admin UI)
# so batches can be reviewed as they land instead of importing everything
# from the CSV at once.
# --------------------------------------------------------------------------

_amio_cache: list = []
_amio_cache_path: str | None = None


def _load_amio_csv(force: bool = False) -> list:
    """Parses AMIO_CSV_PATH and caches the result. Re-parses only if forced
    or if the cache is empty (first call)."""
    global _amio_cache, _amio_cache_path
    if _amio_cache and not force and _amio_cache_path == AMIO_CSV_PATH:
        return _amio_cache
    if not os.path.exists(AMIO_CSV_PATH):
        raise HTTPException(
            404,
            f"No AMIO CSV found at {AMIO_CSV_PATH}. Commit it to the repo at "
            "backend/data/amio_raw.csv (or set AMIO_CSV_PATH) and redeploy.",
        )
    _amio_cache = amio_import.parse_csv(AMIO_CSV_PATH)
    _amio_cache_path = AMIO_CSV_PATH
    return _amio_cache


@router.get("/admin/aimo/contests")
async def aimo_contests(
    reload: bool = False,
    admin_session: str | None = Cookie(default=None),
):
    """Lists contests found in the repo's AMIO CSV. Pass ?reload=true to
    re-read the file from disk (e.g. after committing an updated CSV and
    redeploying) instead of using the in-memory cache."""
    require_admin(admin_session)
    records = _load_amio_csv(force=reload)
    unparsed = sum(1 for r in records if not r["contest"])
    return {
        "csv_path": AMIO_CSV_PATH,
        "total_problems_in_csv": len(records),
        "unparsed_links": unparsed,
        "contests": amio_import.list_contests(records),  # {"2024 AMC 8": 25, ...}
    }


@router.get("/admin/aimo/unparsed-sample")
async def aimo_unparsed_sample(
    limit: int = 20,
    admin_session: str | None = Cookie(default=None),
):
    """Diagnostic: shows actual link values that failed to parse, so the
    AMC_LINK_PATTERN/AJHSME_LINK_PATTERN regexes in amio_import.py can be
    widened to cover whatever contest formats the CSV actually contains
    (AIME, ARML, etc.) instead of guessing blind. Temporary tool — fine to
    remove once the patterns are confirmed to cover everything in your
    dataset."""
    require_admin(admin_session)
    records = _load_amio_csv()
    unparsed = [r["link"] for r in records if not r["contest"]]
    return {
        "total_unparsed": len(unparsed),
        "sample": unparsed[:limit],
    }


@router.post("/admin/aimo/import")
async def aimo_import_contest(
    request: Request,
    admin_session: str | None = Cookie(default=None),
):
    """Body: {contest: "2024 AMC 8", model?}. Tags and imports just that one
    contest's problems from the repo's AIMO CSV — the one-click-per-contest
    path, so you can watch each batch land instead of importing everything
    at once."""
    require_admin(admin_session)
    records = _load_amio_csv()

    body = await request.json()
    contest = (body.get("contest") or "").strip()
    model = (body.get("model") or tc.MODEL).strip()
    if not contest:
        raise HTTPException(400, "contest is required, e.g. '2024 AMC 8'.")

    subset = amio_import.records_for_contest(records, contest)
    if not subset:
        raise HTTPException(404, f"No problems found for contest '{contest}' in {AMIO_CSV_PATH}.")

    tagged = amio_import.tag_aimo_records(subset, model)
    return _finish_import(tagged)


# --------------------------------------------------------------------------
# Shared finish step: merge into in-memory corpus, recompute embeddings for
# the new rows, persist, commit.
# --------------------------------------------------------------------------

def _finish_import(new_records: list) -> dict:
    import numpy as np
    import app as appmod  # the running FastAPI app module — holds `corpus`/`embeddings`

    existing_ids = {r["id"] for r in appmod.corpus}
    added, replaced = [], []
    for rec in new_records:
        if rec["id"] in existing_ids:
            idx = next(i for i, r in enumerate(appmod.corpus) if r["id"] == rec["id"])
            appmod.corpus[idx] = rec
            replaced.append(rec["id"])
        else:
            appmod.corpus.append(rec)
            added.append(rec["id"])

    # Recompute embeddings for the whole corpus. sentence-transformers is heavy
    # to keep loaded permanently, so it's imported lazily, here, only on import.
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        vectors = model.encode([tc.embed_text(r) for r in appmod.corpus],
                                normalize_embeddings=True)
        appmod.embeddings = np.array(vectors)
        np.save(appmod.EMBEDDINGS_PATH, appmod.embeddings)
        embeddings_updated = True
    except Exception as e:
        print(f"  ! embedding recompute failed: {e}", file=sys.stderr)
        embeddings_updated = False

    # Persist tagged.json locally (best-effort — Render's disk may reset later,
    # which is exactly why the GitHub commit below is the real persistence).
    try:
        json.dump(appmod.corpus, open(appmod.CORPUS_PATH, "w"),
                   indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"  ! local corpus write failed: {e}", file=sys.stderr)

    ids = added + replaced
    commit_result = commit_corpus_to_github(
        appmod.corpus,
        message=f"Import {len(ids)} problem(s): {', '.join(ids)}",
    )

    return {
        "added": added,
        "replaced": replaced,
        "total_corpus_size": len(appmod.corpus),
        "embeddings_updated": embeddings_updated,
        "github": commit_result,
        "records": new_records,
    }
