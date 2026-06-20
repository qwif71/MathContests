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

_state = {"techniques": [], "pending": [], "log": []}  # pending = awaiting admin approval
MAX_LOG_ENTRIES = 200  # keep the log bounded; oldest entries drop off


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
            "log": loaded.get("log", []),
        }
    except (FileNotFoundError, json.JSONDecodeError):
        _state = {"techniques": _seed_defaults(), "pending": [], "log": []}


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


# Words that, when they're the part of two technique names that actually
# differs, mean the names refer to genuinely different techniques even if
# they're otherwise character-for-character similar — e.g. "difference of
# squares" vs "difference of cubes" scores 0.85 on raw SequenceMatcher
# similarity (well above FUZZY_THRESHOLD) but these are different
# identities, not a phrasing variant of the same one. This page exists
# specifically to preserve that kind of distinction, so a fuzzy match is
# rejected outright if the two normalized strings differ ONLY by swapping
# words from this set (in any position), regardless of overall similarity.
_DISTINGUISHING_WORDS = {
    "square", "squares", "cube", "cubes", "fourth", "fifth", "nth",
    "sum", "difference", "product", "quotient",
    "linear", "quadratic", "cubic", "quartic", "polynomial",
    "interior", "exterior", "inscribed", "circumscribed",
    "left", "right", "upper", "lower",
    "min", "max", "minimum", "maximum",
    "row", "column", "diagonal",
    "horizontal", "vertical",
    "odd", "even",
    "necessary", "sufficient",
}


def _distinguishing_word_conflict(a: str, b: str) -> bool:
    """True if normalized strings `a` and `b` are identical except for a
    swap of one or more words in _DISTINGUISHING_WORDS — meaning high
    surface similarity is coincidental, not synonymy."""
    wa, wb = a.split(), b.split()
    if len(wa) != len(wb):
        return False  # different word counts -> not a simple word-swap case
    diffs = [(x, y) for x, y in zip(wa, wb) if x != y]
    if not diffs:
        return False  # identical, not a conflict (caller handles exact-match separately)
    # Every differing word-pair must involve at least one distinguishing word.
    return all(x in _DISTINGUISHING_WORDS or y in _DISTINGUISHING_WORDS for x, y in diffs)


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
    #    Excludes candidates that only differ by a distinguishing word
    #    (squares vs cubes, interior vs exterior, etc.) — those are kept
    #    as separate suggestions, never auto-matched, regardless of score.
    scored = []
    for c in pool:
        norm_c = _normalize(c)
        if _distinguishing_word_conflict(norm_typed, norm_c):
            continue
        score = difflib.SequenceMatcher(None, norm_typed, norm_c).ratio()
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


def diff_techniques(batch_techniques: list[str], batch_records: list | None = None) -> dict:
    """Given the raw `techniques` lists from a freshly-tagged import batch
    (not yet merged into the corpus), classify each distinct technique
    string as known (fuzzy-matches an existing canonical tag) or new.

    If `batch_records` is given (the full tagged records, not just the
    flat technique list), each "new" entry also reports which problem(s)
    in the batch it came from — `example_ids` / `example_summaries` — so
    the admin can see what the tag was applied to without needing a
    separate explanation from the model (which isn't available anyway:
    the tagging call uses forced tool-use, so the model can't emit prose
    reasoning alongside the structured tags).

    Also flags likely batch-internal synonyms: pairs of NEW techniques
    (neither yet canonical) that are suspiciously similar to each other,
    via the same distinguishing-word-safe fuzzy match used against the
    canonical list — e.g. would catch "Power of a Point" and "power of
    point" both being introduced in one batch, but would NOT flag
    "difference of squares" against "difference of cubes".

    Returns:
        {
          "known": [{"input": ..., "matched": ...}, ...],
          "new":   [{"input": ..., "suggestions": [...], "example_ids": [...],
                     "example_summaries": [...], "possible_duplicate_of": str|None}, ...]
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

    # Attach example problems for each new technique, if records were given.
    if batch_records:
        for entry in new:
            examples = [r for r in batch_records if entry["input"] in r.get("techniques", [])]
            entry["example_ids"] = [r.get("id") for r in examples][:5]
            entry["example_summaries"] = [r.get("summary", "") for r in examples][:5]

    # Batch-internal synonym check: compare each new tag against every
    # OTHER new tag (not yet against canonical, since match_technique
    # already did that) using the same distinguishing-word-safe fuzzy
    # logic, so two new tags introduced in the same batch that are likely
    # the same concept get flagged for the admin to merge manually.
    new_names = [n["input"] for n in new]
    for entry in new:
        other_pool = [n for n in new_names if n != entry["input"]]
        if not other_pool:
            entry["possible_duplicate_of"] = None
            continue
        probe = match_technique(entry["input"], candidates=other_pool)
        entry["possible_duplicate_of"] = probe["matched"]  # None if no confident match

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


def _log_event(action: str, names: list[str], extra: dict | None = None) -> None:
    """Append entries to the bounded recent-activity log. Called for
    approvals (and could be extended to merges/rejections later)."""
    import time
    global _state
    for n in names:
        entry = {"action": action, "name": n, "at": time.time()}
        if extra:
            entry.update(extra)
        _state["log"].append(entry)
    # Keep only the most recent MAX_LOG_ENTRIES, oldest first dropped.
    if len(_state["log"]) > MAX_LOG_ENTRIES:
        _state["log"] = _state["log"][-MAX_LOG_ENTRIES:]


def get_log(limit: int = 50) -> list:
    """Most recent technique-log entries, newest first."""
    return list(reversed(_state["log"]))[:max(1, limit)]


def log_merge(resolutions: dict[str, str]) -> None:
    """Log merge decisions: typed/new string -> existing canonical tag it
    was folded into. Doesn't touch the canonical list itself (the existing
    tag is already there)."""
    global _state
    import time
    for typed, target in resolutions.items():
        _state["log"].append({
            "action": "merged", "name": typed, "merged_into": target, "at": time.time(),
        })
    if len(_state["log"]) > MAX_LOG_ENTRIES:
        _state["log"] = _state["log"][-MAX_LOG_ENTRIES:]
    _save_local()


def log_reject(names: list[str]) -> None:
    """Log rejected technique names (dropped, not added to canonical)."""
    _log_event("rejected", names)
    _save_local()


def approve_techniques(names: list[str], renames: dict[str, str] | None = None) -> list:
    """Add `names` to the canonical list (deduped, normalized-exact-safe),
    clear them from pending, persist locally, and commit to GitHub.

    `renames` optionally maps a name being approved -> the actual canonical
    name to use instead (e.g. approving "Sum of Roots (Vieta's)" but
    storing it as "Vieta's Formulas"). This is the name that actually lands
    in the canonical list; the original `names` entry is what gets cleared
    from pending and what the caller (admin.py) needs in order to remap
    the batch's records to the renamed value.

    Returns the updated canonical list."""
    global _state
    renames = renames or {}
    existing_norm = {_normalize(c) for c in _state["techniques"]}
    final_names = []
    for n in names:
        n = n.strip()
        final = renames.get(n, n).strip()
        final_names.append(final)
        if final and _normalize(final) not in existing_norm:
            _state["techniques"].append(final)
            existing_norm.add(_normalize(final))
    _state["pending"] = [p for p in _state["pending"] if p not in names]
    _log_event("approved", final_names)
    _save_local()
    label = ", ".join(final_names)
    _commit_to_github(f"Approve {len(final_names)} new technique tag(s): {label}")
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


def rename_technique(old_name: str, new_name: str) -> dict:
    """Rename a canonical technique. Only touches the canonical list itself
    — admin.py's /admin/techniques/rename endpoint is responsible for also
    rewriting every corpus record that uses old_name, since this module has
    no corpus access. Returns {ok, reason?} so the caller can decide whether
    to proceed with the corpus-wide rewrite."""
    global _state
    old_name, new_name = old_name.strip(), new_name.strip()
    if not old_name or not new_name:
        return {"ok": False, "reason": "Both old and new names are required."}
    if old_name not in _state["techniques"]:
        return {"ok": False, "reason": f"'{old_name}' is not in the canonical list."}
    existing_norm = {_normalize(c) for c in _state["techniques"] if c != old_name}
    if _normalize(new_name) in existing_norm:
        return {"ok": False, "reason": f"'{new_name}' already exists as a canonical tag — merge instead of rename."}
    idx = _state["techniques"].index(old_name)
    _state["techniques"][idx] = new_name
    _log_event("renamed", [old_name], extra={"renamed_to": new_name})
    _save_local()
    _commit_to_github(f"Rename technique tag: {old_name} -> {new_name}")
    return {"ok": True}


def delete_technique(name: str) -> dict:
    """Remove a technique from the canonical list entirely (e.g. it was
    approved by mistake, or turned out to be redundant after the fact).
    Does NOT touch corpus records using it — admin.py's
    /admin/techniques/delete endpoint decides whether to also strip it
    from records or leave it (orphaned tags are harmless, just won't be
    suggested/autocompleted going forward)."""
    global _state
    name = name.strip()
    if name not in _state["techniques"]:
        return {"ok": False, "reason": f"'{name}' is not in the canonical list."}
    _state["techniques"].remove(name)
    _log_event("deleted", [name])
    _save_local()
    _commit_to_github(f"Delete technique tag: {name}")
    return {"ok": True}


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
