"""
validate_corpus.py — sanity checks for tagged math-contest corpora.

Run from backend:
    python validate_corpus.py tagged.json

This is intentionally API-free. Use it before/after imports to catch issues that
would waste tagging tokens later: unknown technique drift, duplicate-ish tags,
solution bleed-through, missing fields, and suspicious technique/summary mismatch.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    import techniques as technique_store
except Exception:  # pragma: no cover - lets the validator run even during setup
    technique_store = None

AREAS = {"Algebra", "Combinatorics", "Geometry", "Number Theory"}
REQUIRED = ("id", "statement")
TAG_FIELDS = ("area", "subtopics", "techniques", "difficulty", "answer", "summary")


def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _load_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit(f"{path} must contain a JSON list of records.")
    return data


def _canonical_techniques() -> list[str]:
    if technique_store is None:
        return []
    try:
        return technique_store.get_canonical_techniques()
    except Exception:
        return []


def _match_technique(name: str) -> dict[str, Any]:
    if technique_store is not None:
        try:
            return technique_store.match_technique(name)
        except Exception:
            pass
    return {"input": name, "matched": None, "score": 0.0, "suggestions": []}


def validate(records: list[dict[str, Any]]) -> int:
    errors: list[str] = []
    warnings: list[str] = []

    ids = [r.get("id") for r in records]
    for pid, n in Counter(ids).items():
        if pid and n > 1:
            errors.append(f"duplicate id: {pid} appears {n} times")

    by_statement: defaultdict[str, list[str]] = defaultdict(list)
    unknown_techs: defaultdict[str, list[str]] = defaultdict(list)
    canonical = _canonical_techniques()
    canonical_norms = {_norm(t) for t in canonical}

    for i, r in enumerate(records, start=1):
        pid = r.get("id") or f"<record {i}>"

        for field in REQUIRED:
            if not r.get(field):
                errors.append(f"{pid}: missing required field {field!r}")

        # Fully untagged records may exist in raw files; warn, don't fail.
        missing_tags = [field for field in TAG_FIELDS if field not in r]
        if missing_tags:
            warnings.append(f"{pid}: missing tag fields {missing_tags}")

        areas = r.get("area", []) or []
        bad_areas = [a for a in areas if a not in AREAS]
        if bad_areas:
            errors.append(f"{pid}: invalid area(s): {bad_areas}")

        statement_key = _norm(r.get("statement", ""))
        if statement_key:
            by_statement[statement_key].append(pid)

        solution = r.get("solution", "") or ""
        # This catches extraction chunks that accidentally include the next problem.
        if re.search(r"\bProblem\s+\d+\s*[.:]", solution, flags=re.I):
            warnings.append(f"{pid}: solution may contain another problem statement")
        if re.search(r"\bSolution\s+\d+\s*[.:]", solution, flags=re.I):
            warnings.append(f"{pid}: solution may contain another solution section")

        techs = r.get("techniques", []) or []
        seen_local = set()
        for t in techs:
            if not isinstance(t, str) or not t.strip():
                errors.append(f"{pid}: blank/non-string technique tag {t!r}")
                continue
            nt = _norm(t)
            if nt in seen_local:
                warnings.append(f"{pid}: duplicate technique tag {t!r}")
            seen_local.add(nt)

            if canonical and nt not in canonical_norms:
                m = _match_technique(t)
                if not m.get("matched"):
                    unknown_techs[t].append(pid)
                elif m.get("matched") != t:
                    warnings.append(
                        f"{pid}: technique {t!r} should probably be canonicalized to {m['matched']!r}"
                    )

        summary = (r.get("summary") or "").strip()
        if summary and len(summary.split()) > 25:
            warnings.append(f"{pid}: summary is long ({len(summary.split())} words)")

    for key, problem_ids in by_statement.items():
        if len(problem_ids) > 1:
            warnings.append(f"duplicate/same statement? {problem_ids[:8]}")

    for tech, problem_ids in sorted(unknown_techs.items(), key=lambda x: (-len(x[1]), x[0].lower())):
        m = _match_technique(tech)
        suggestion = ""
        if m.get("suggestions"):
            suggestion = f"; suggestions: {m['suggestions'][:3]}"
        warnings.append(f"unknown technique {tech!r} used by {len(problem_ids)} record(s): {problem_ids[:6]}{suggestion}")

    print(f"Checked {len(records)} records.")
    if errors:
        print("\nERRORS")
        for e in errors:
            print(f"  - {e}")
    if warnings:
        print("\nWARNINGS")
        for w in warnings:
            print(f"  - {w}")
    if not errors and not warnings:
        print("No problems found.")

    return 1 if errors else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", help="Path to tagged/raw corpus JSON")
    args = ap.parse_args()
    raise SystemExit(validate(_load_records(Path(args.corpus))))


if __name__ == "__main__":
    main()
