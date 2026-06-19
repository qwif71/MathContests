"""
settings.py — small, file-backed feature flags for the backend.

Currently holds exactly one flag: AI_QUERY_ENABLED, which gates whether
/practice is allowed to call the Anthropic API to parse free-text queries
(see app.py's ai_parse_query). Everything else about the app (tagging on
import, etc.) is unaffected by this flag — it only covers query-time calls.

Persistence: same pattern as the corpus — kept in memory for fast reads,
written to a local JSON file, and (best-effort) committed to GitHub so it
survives Render's ephemeral disk and redeploys. If GitHub isn't configured,
the flag still works for the life of the process; it just resets to the
default (False) on the next deploy, which is the safe direction to fail in.
"""
import base64
import json
import os
import requests

BASE_DIR = os.environ.get("BASE_DIR", "/opt/render/project/src/backend")
SETTINGS_PATH = os.environ.get("SETTINGS_PATH", os.path.join(BASE_DIR, "settings.json"))
SETTINGS_REPO_PATH = os.environ.get("SETTINGS_REPO_PATH", "backend/settings.json").strip()

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "").strip()
GITHUB_REPO = os.environ.get("GITHUB_REPO", "").strip()
GITHUB_BRANCH = os.environ.get("GITHUB_BRANCH", "main").strip()

DEFAULTS = {"ai_query_enabled": False}

_state = dict(DEFAULTS)


def _load():
    global _state
    try:
        with open(SETTINGS_PATH) as f:
            loaded = json.load(f)
        _state = {**DEFAULTS, **loaded}
    except (FileNotFoundError, json.JSONDecodeError):
        _state = dict(DEFAULTS)


_load()


def get_ai_query_enabled() -> bool:
    return bool(_state.get("ai_query_enabled", False))


def _github_headers():
    return {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }


def _commit_to_github(message: str) -> dict:
    if not GITHUB_TOKEN or not GITHUB_REPO:
        return {"committed": False, "reason": "GITHUB_TOKEN/GITHUB_REPO not configured"}
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{SETTINGS_REPO_PATH}"
    get_resp = requests.get(url, headers=_github_headers(),
                             params={"ref": GITHUB_BRANCH}, timeout=20)
    sha = get_resp.json().get("sha") if get_resp.status_code == 200 else None
    content_str = json.dumps(_state, indent=2)
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


def set_ai_query_enabled(value: bool) -> dict:
    """Update the flag, persist locally, and best-effort commit to GitHub.
    Returns the github commit result so the admin UI can show whether it
    actually persisted across deploys."""
    global _state
    _state["ai_query_enabled"] = bool(value)
    try:
        os.makedirs(os.path.dirname(SETTINGS_PATH), exist_ok=True)
        with open(SETTINGS_PATH, "w") as f:
            json.dump(_state, f, indent=2)
    except Exception as e:
        print(f"  ! settings local write failed: {e}")
    return _commit_to_github(
        f"{'Enable' if value else 'Disable'} AI-assisted query parsing")
