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
        max_age=SESSION_MAX_AGE, httponly=True,
        samesite=COOKIE_SAMESITE, secure=COOKIE_SECURE,
    )
    return resp


@router.post("/admin/logout")
async def admin_logout():
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(COOKIE_NAME)
    return resp


@router.get("/admin/me")
async def admin_me(admin_session: str | None = Cookie(default=None)):
    try:
        require_admin(admin_session)
    except HTTPException:
        return {"logged_in": False}
    return {"logged_in": True}


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
