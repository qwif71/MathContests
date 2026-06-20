"""
techniques.py — the canonical technique-tag vocabulary as data, not code.

Used to be a Python list baked into tag_and_compare.py (TECHNIQUES), which
meant adding a new tag required a code change + redeploy. Now it's a
file-backed list (same pattern as settings.py): kept in memory for fast
reads, written to a local JSON file, and best-effort committed to GitHub so
it survives Render's ephemeral disk and redeploys.

tag_and_compare.py's static TECHNIQUES list is still the *seed* — see
DEFAULTS below — but from here on, new techniques only enter the canonical
list through admin approval (see admin.py's tag-review endpoints), not by
editing code.

Fuzzy matching: exposes match_technique(), the single shared utility used by
both (a) the import-time diff against new vs. known techniques, and (b) the
frontend's manual tag-entry box, so "near enough" typed input (casing,
whitespace, common abbreviations) resolves to the existing canonical tag
instead of silently fragmenting into a near-duplicate.
"""
import base64
import difflib
import json
import os
import re
import requests

BASE_DIR = os.environ.get("BASE_DIR", "/opt/render/project/src/backend")
TECHNIQUES_PATH = os.environ.get("TECHNIQUES_PATH", os.path.join(BASE_DIR, "techniques.json"))
TECHNIQUES_REPO_PATH = os.environ.get("TECHNIQUES_REPO_PATH", "backend/techniques.json").strip()

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "").strip()
GITHUB_REPO = os.environ.get("GITHUB_REPO", "").strip()
GITHUB_BRANCH = os.environ.get("GITHUB_BRANCH", "main").strip()

# Below this similarity score, a typed/new technique is treated as genuinely
# new (not a near-duplicate of anything canonical). Tune by trying real near-
# dupes ("power of a point" vs "Power of a Point" vs "POP") against real
# distinct techniques and picking a threshold that separates them.
FUZZY_THRESHOLD = 0.84

_state = {"techniques": [], "pending": []}  # pending = awaiting admin approval


def _seed_defaults() -> list:
    """Seed from tag_and_compare.py's existing TECHNIQUES list, so the very
    first load isn't empty relative to what's already in tagged.json."""
    try:
        import tag_and_compare as tc
        return list(dict.fromkeys(tc.TECHNIQUES))  # dedup, preserve order
    except Exception:
        return []


def _load():
    global _state
    try:
        with open(TECHNIQUES_PATH) as f:
            loaded = json.load(f)
        _state = {
            "techniques": loaded.get("techniques", []),
            "pending": loaded.get("pending", []),
        }
    except (FileNotFoundError, json.JSONDecodeError):
        _state = {"techniques": _seed_defaults(), "pending": []}


_load()


# ---------------------------------------------------------------------------
# Normalization + fuzzy matching — the shared utility
# ---------------------------------------------------------------------------

def _normalize(s: str) -> str:
    """Case/whitespace/punctuation-insensitive form used for comparison."""
    s = s.strip().lower()
    s = re.sub(r"[''`]", "", s)          # drop apostrophes: "Vieta's" == "Vietas"
    s = re.sub(r"[^a-z0-9]+", " ", s)    # collapse punctuation/hyphens to space
    return re.sub(r"\s+", " ", s).strip()


def match_technique(typed: str, candidates: list[str] | None = None) -> dict:
    """Resolve a typed/candidate technique string against the canonical list.

    Returns:
        {
          "input": typed,
          "matched": "<canonical name>" | None,   # best match if confident
          "score": float,                          # 0..1 similarity to `matched`
          "exact": bool,                            # normalized-exact match
          "suggestions": [<canonical name>, ...]    # top alternatives, for UI
        }

    `candidates` lets the caller pass the AREAS/import-time technique list
    too (so a brand-new import batch can fuzzy-match against itself, not
    just against the persisted canonical list) — defaults to the canonical
    list if omitted.
    """
    pool = candidates if candidates is not None else _state["techniques"]
    norm_typed = _normalize(typed)
    if not norm_typed:
        return {"input": typed, "matched": None, "score": 0.0, "exact": False, "suggestions": []}

    # 1. Exact match after normalization (handles casing/whitespace drift,
    #    e.g. "power of a point" == "Power of a Point").
    for c in pool:
        if _normalize(c) == norm_typed:
            return {"input": typed, "matched": c, "score": 1.0, "exact": True, "suggestions": []}

    # 2. Fuzzy match for everything else (typos, near-duplicate phrasings,
    #    short abbreviations like "SFFT" already in the canonical name).
    scored = []
    for c in pool:
        score = difflib.SequenceMatcher(None, norm_typed, _normalize(c)).ratio()
        if score > 0.5:
            scored.append((score, c))
    scored.sort(key=lambda x: -x[0])

    suggestions = [c for _, c in scored[:5]]
    if scored and scored[0][0] >= FUZZY_THRESHOLD:
        return {"input": typed, "matched": scored[0][1], "score": scored[0][0],
                "exact": False, "suggestions": suggestions}

    return {"input": typed, "matched": None,
            "score": scored[0][0] if scored else 0.0,
            "exact": False, "suggestions": suggestions}


# ---------------------------------------------------------------------------
# Canonical list access
# ---------------------------------------------------------------------------

def get_canonical_techniques() -> list:
    return list(_state["techniques"])


def get_pending() -> list:
    """Techniques tagged by the model in an import batch but not yet
    approved by the admin. Kept here (not just in the API response) so a
    pending review survives a page refresh before approval."""
    return list(_state["pending"])


def diff_techniques(batch_techniques: list[str]) -> dict:
    """Given the raw `techniques` lists from a freshly-tagged import batch
    (not yet merged into the corpus), classify each distinct technique
    string as known (fuzzy-matches an existing canonical tag) or new.

    Returns:
        {
          "known": [{"input": ..., "matched": ...}, ...],   # auto-resolved
          "new":   [{"input": ..., "suggestions": [...]}, ...]  # needs admin review
        }
    """
    seen = list(dict.fromkeys(t.strip() for t in batch_techniques if t and t.strip()))
    known, new = [], []
    for t in seen:
        m = match_technique(t)
        if m["matched"]:
            known.append(m)
        else:
            new.append(m)
    return {"known": known, "new": new}


def remap_batch(tagged_records: list, resolutions: dict[str, str]) -> list:
    """Apply admin-approved resolutions to a batch's technique tags before
    merge. `resolutions` maps the original typed/model string -> the final
    canonical string to use (could be a brand-new approved tag, an existing
    tag it was merged into, or unchanged). Records are mutated in place
    (copies returned) — does not touch the persisted canonical list."""
    out = []
    for rec in tagged_records:
        rec = dict(rec)
        techs = rec.get("techniques", [])
        rec["techniques"] = list(dict.fromkeys(
            resolutions.get(t, t) for t in techs if resolutions.get(t, t)
        ))
        out.append(rec)
    return out


def set_pending(new_techniques: list[str]) -> None:
    """Stash the batch's new-technique candidates so they're visible via
    get_pending() until approve/reject runs. Overwrites any prior pending
    set — there's one review queue at a time, matching the one-import-at-a-
    time admin workflow."""
    global _state
    _state["pending"] = list(dict.fromkeys(new_techniques))
    _save_local()


def approve_techniques(names: list[str]) -> list:
    """Add `names` to the canonical list (deduped, normalized-exact-safe),
    clear them from pending, persist locally, and commit to GitHub.
    Returns the updated canonical list."""
    global _state
    existing_norm = {_normalize(c) for c in _state["techniques"]}
    for n in names:
        n = n.strip()
        if n and _normalize(n) not in existing_norm:
            _state["techniques"].append(n)
            existing_norm.add(_normalize(n))
    _state["pending"] = [p for p in _state["pending"] if p not in names]
    _save_local()
    _commit_to_github(f"Approve {len(names)} new technique tag(s): {', '.join(names)}")
    return list(_state["techniques"])


def reject_techniques(names: list[str]) -> list:
    """Drop `names` from pending without adding them to the canonical list.
    Does NOT touch already-imported records — callers decide separately
    whether rejection blocks the import or just leaves the tag unmerged
    (see admin.py's REJECTED_TECHNIQUES_BLOCK_IMPORT)."""
    global _state
    _state["pending"] = [p for p in _state["pending"] if p not in names]
    _save_local()
    return list(_state["pending"])


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _save_local() -> None:
    try:
        os.makedirs(os.path.dirname(TECHNIQUES_PATH), exist_ok=True)
        with open(TECHNIQUES_PATH, "w") as f:
            json.dump(_state, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"  ! techniques.json local write failed: {e}")


def _github_headers():
    return {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }


def _commit_to_github(message: str) -> dict:
    if not GITHUB_TOKEN or not GITHUB_REPO:
        return {"committed": False, "reason": "GITHUB_TOKEN/GITHUB_REPO not configured"}
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{TECHNIQUES_REPO_PATH}"
    get_resp = requests.get(url, headers=_github_headers(),
                             params={"ref": GITHUB_BRANCH}, timeout=20)
    sha = get_resp.json().get("sha") if get_resp.status_code == 200 else None
    content_str = json.dumps(_state, indent=2, ensure_ascii=False)
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
