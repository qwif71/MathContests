#!/usr/bin/env python3
"""
import_aimo.py — one-time import of AMC/AIME problems from the AI-MO/aops
Hugging Face dataset into this repo's corpus format.

Reuses tag_and_compare.py's tag()/add_difficulty() so the resulting records
are identical in shape to anything imported via the CLI or the admin page —
no duplicated tagging logic, no separate prompt to keep in sync.

WHY A SEPARATE SCRIPT INSTEAD OF AN ADMIN-PAGE UPLOAD:
This is a single bulk pull from a public dataset, not an ongoing import
workflow. It doesn't need a password-gated UI, a PDF parser, or a GitHub
auto-commit step — those exist in admin.py for repeated day-to-day use.
This script writes a local JSON file; you fold it into tagged.json with the
existing `combine` command (same as add_contest.sh does for PDFs), so you
get to look at a diff before anything touches the live corpus.

USAGE:
    pip install datasets --upgrade
    export ANTHROPIC_API_KEY=sk-ant-...

    # 1. Dry run first — see what would be pulled in, no tagging, no API calls.
    python import_aimo.py --dry-run

    # 2. Pull + tag for real. Writes aimo_amc_aime.json next to tagged.json.
    python import_aimo.py --out aimo_amc_aime.json

    # 3. Fold into your real corpus (uses tag_and_compare.py's own dedup):
    python tag_and_compare.py combine tagged.json aimo_amc_aime.json --out tagged.json

    # 4. Review the diff, then let the admin page's normal GitHub-commit path
    #    (or a manual `git commit`) ship it — this script never touches GitHub.

TAG FORMAT IN THE SOURCE DATASET (confirmed against the live dataset viewer):
    tags = ["origin:aops", "<YEAR> Contests", "<contest/round name>"]
e.g. ["origin:aops", "2022 Contests", "2022 AIME Problems"]
     ["origin:aops", "2024 Contests", "2024 AMC 10A Problems"]
The third tag is what we filter on and parse contest/year/round/number from.
Some rows (the MAA-copyright boilerplate row at the top of each contest
folder) have empty problem/solution text — those are skipped automatically
since they carry no usable statement.
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tag_and_compare as tc  # noqa: E402  (reuse tag()/add_difficulty(), no copy-paste)

# Matches e.g. "2024 AMC 10A Problems" -> year=2024 contest="AMC 10A"
#              "2022 AIME Problems"    -> year=2022 contest="AIME"  (AIME I/II if present)
CONTEST_RE = re.compile(
    r"^(?P<year>\d{4})\s+(?P<contest>AMC\s*(?:10|12)\s*[AB]?|AIME\s*(?:I{1,2})?)\b",
    re.IGNORECASE,
)


def parse_contest_tag(tag_str):
    """Pull (year, contest_label) out of a tag like '2024 AMC 10A Problems'.
    Returns None if the tag doesn't match the expected AMC/AIME pattern."""
    m = CONTEST_RE.match(tag_str.strip())
    if not m:
        return None
    year = m.group("year")
    contest = re.sub(r"\s+", " ", m.group("contest").strip()).upper()
    return year, contest


def is_amc_or_aime_row(tags):
    """A row counts if any tag matches the AMC/AIME contest pattern. Plain
    substring matching on 'AMC'/'AIME' would also catch unrelated mentions
    inside problem-prose tags (there aren't any here, since tags are
    structural, not content-derived) — the regex is the stricter, correct
    check and costs nothing extra."""
    for t in tags or []:
        if parse_contest_tag(t):
            return True
    return False


def problem_number_from_path(path):
    """metadata.path looks like '.../2024 AMC 10A Problems/2998765.json' —
    the AoPS thread id, not the problem number. AI-MO's source data doesn't
    carry an explicit problem-within-contest index, so we don't fabricate
    one: 'number' is left unset and add_difficulty() falls back to its
    default bucket for that (contest, round) group instead of a fabricated
    position-based score. This is honest about what we don't know, rather
    than guessing a number that could be wrong."""
    return None


def make_record(row, idx, contest_label, year):
    problem = (row.get("problem") or "").strip()
    solution = (row.get("solution") or "").strip()
    if not problem:
        return None  # the MAA-copyright boilerplate rows etc.

    path = (row.get("metadata") or {}).get("path", "")
    thread_id = re.search(r"(\d+)\.json$", path)
    thread_id = thread_id.group(1) if thread_id else str(idx)

    pid = f"AIMO-{contest_label.replace(' ', '')}-{year}-{thread_id}"

    return {
        "id": pid,
        "statement": " ".join(problem.split()),
        "solution": " ".join(solution.split()),
        "contest": f"{contest_label} {year}",
        "round": "Individual Round",
        "number": problem_number_from_path(path),
        "sources": [f"AI-MO/aops ({contest_label} {year})"],
    }


def load_rows():
    from datasets import load_dataset
    ds = load_dataset("AI-MO/aops", split="train")
    return ds


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="aimo_amc_aime.json",
                     help="Output path for the new (untagged-then-tagged) records.")
    ap.add_argument("--dry-run", action="store_true",
                     help="Just report what would be pulled in; no tagging, no API calls.")
    ap.add_argument("--model", default=tc.MODEL,
                     help="Anthropic model to use for tagging (default: %(default)s).")
    ap.add_argument("--limit", type=int, default=None,
                     help="Optional cap on number of records, for a quick test run.")
    args = ap.parse_args()

    print("Loading AI-MO/aops from Hugging Face (this downloads ~141MB once, "
          "then caches)...", file=sys.stderr)
    ds = load_rows()
    print(f"Loaded {len(ds)} total rows.", file=sys.stderr)

    records, skipped_empty, by_contest = [], 0, {}
    for idx, row in enumerate(ds):
        tags = row.get("tags") or []
        match = next((parse_contest_tag(t) for t in tags if parse_contest_tag(t)), None)
        if not match:
            continue
        year, contest_label = match
        rec = make_record(row, idx, contest_label, year)
        if rec is None:
            skipped_empty += 1
            continue
        records.append(rec)
        by_contest[rec["contest"]] = by_contest.get(rec["contest"], 0) + 1

    if args.limit:
        records = records[:args.limit]

    print(f"\nMatched {len(records) + skipped_empty} AMC/AIME-tagged rows "
          f"({skipped_empty} skipped for empty problem text).", file=sys.stderr)
    print(f"Usable records: {len(records)}\n", file=sys.stderr)
    for contest, n in sorted(by_contest.items()):
        print(f"  {n:4d}  {contest}", file=sys.stderr)

    if args.dry_run:
        print("\n--dry-run: stopping before tagging/API calls. "
              "Sample record:", file=sys.stderr)
        if records:
            print(json.dumps(records[0], indent=2, ensure_ascii=False), file=sys.stderr)
        return

    if not records:
        sys.exit("No records matched — check the tag pattern against the live dataset.")

    print(f"\nTagging {len(records)} records via {args.model} "
          f"(this calls the Anthropic API once per record)...", file=sys.stderr)
    tc.add_difficulty(records)
    tagged = tc.tag(records, args.model)

    with open(args.out, "w") as f:
        json.dump(tagged, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {len(tagged)} tagged records -> {args.out}", file=sys.stderr)
    print(f"\nNext step:\n"
          f"  python tag_and_compare.py combine tagged.json {args.out} --out tagged.json",
          file=sys.stderr)


if __name__ == "__main__":
    main()
