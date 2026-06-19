#!/usr/bin/env python3
"""
import_kaggle_amio.py — import AMC/AIME/AMC8/AHSME/USAMO/USAJMO problems from
the Kaggle "AMIO parsed Art Of Problem Solving website" dataset
(alexryzhkov/amio-parsed-art-of-problem-solving-website) into this repo's
corpus format.

WHY THIS DATASET INSTEAD OF (OR ALONGSIDE) AI-MO/aops:
The `link` column contains the literal AoPS URL, e.g.
    .../wiki/index.php/2024_AMC_8_Problems/Problem_1
The trailing "Problem_N" is parsed straight out of that URL — this is the
*actual* source number, not an inference, so the fuzzy-matching step
(fetch_problem_numbers.py) used for AI-MO records is unnecessary here.

CSV SHAPE (confirmed from a real sample):
    problem_id,link,problem,solution,letter,answer
Multiple rows share the same problem_id — one row per community-contributed
SOLUTION, not one row per problem. This script groups by problem_id and:
  - uses the first row's `problem` text as the canonical statement
  - keeps the first solution as `solution`
  - keeps any additional solutions for that problem as `candidates`
  - keeps `letter`/`answer` as `answer` (prefers `letter` when both present,
    matching the multiple-choice format used by AMC; falls back to `answer`
    for free-response contests like AIME/USAMO/USAJMO where there's no letter)

USAGE:
    1. Download the CSV from Kaggle (kaggle datasets download -d
       alexryzhkov/amio-parsed-art-of-problem-solving-website --unzip)
    2. python import_kaggle_amio.py path/to/file.csv --dry-run
    3. python import_kaggle_amio.py path/to/file.csv --out kaggle_amio.json
    4. python tag_and_compare.py combine tagged.json kaggle_amio.json --out tagged.json

This script does NOT call the Anthropic API for tagging/difficulty by
default — it reuses tag_and_compare.py's tag()/add_difficulty() exactly like
import_aimo.py does, so output records are schema-identical and go through
the same combine/dedup path.
"""
import argparse
import csv
import json
import os
import re
import sys
from collections import OrderedDict, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tag_and_compare as tc  # noqa: E402  (reuse tag()/add_difficulty())

# Parses e.g. ".../wiki/index.php/2024_AMC_8_Problems/Problem_1"
#  -> year=2024 contest="AMC 8" number=1
# and ".../wiki/index.php/2023_AIME_I_Problems/Problem_7"
#  -> year=2023 contest="AIME I" number=7
# and ".../wiki/index.php/1995_AHSME_Problems/Problem_12"
#  -> year=1995 contest="AHSME" number=12
LINK_RE = re.compile(
    r"/(?P<year>\d{4})_(?P<season>Fall_|Spring_)?(?P<contest>AMC_(?:8|10|12)[ABP]?|AJHSME|AHSME|"
    r"AIME(?:_I{1,2})?|USAMO|USAJMO)_Problems/Problem_(?P<number>\d+)",
    re.IGNORECASE,
)


def parse_link(link):
    """Extract (year, contest_label, number) from an AoPS problem URL.
    Returns None if the URL doesn't match a recognized contest pattern
    (e.g. a 'Problems_and_Solutions' index page link, or something
    malformed) rather than guessing."""
    m = LINK_RE.search(link or "")
    if not m:
        return None
    year = m.group("year")
    contest = m.group("contest").replace("_", " ").upper()
    season = m.group("season")
    if season:
        contest = f"{contest.split()[0]} {contest.split()[1]} {season.strip('_').upper()}" \
            if len(contest.split()) > 1 else f"{contest} {season.strip('_').upper()}"
    number = int(m.group("number"))
    return year, contest, number


def clean(text):
    return " ".join((text or "").split())


def load_csv_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def group_by_problem(rows):
    """Group CSV rows (one per solution) into one entry per problem_id,
    preserving first-seen order and collecting all solutions."""
    groups = OrderedDict()
    for row in rows:
        pid = row.get("problem_id")
        if not pid:
            continue
        if pid not in groups:
            groups[pid] = {
                "problem_id": pid,
                "link": row.get("link", ""),
                "problem": row.get("problem", ""),
                "solutions": [],
                "letter": row.get("letter") or None,
                "answer": row.get("answer") or None,
            }
        sol = clean(row.get("solution"))
        if sol:
            groups[pid]["solutions"].append(sol)
        # Prefer a non-empty letter/answer if an earlier row lacked one.
        if not groups[pid]["letter"] and row.get("letter"):
            groups[pid]["letter"] = row.get("letter")
        if not groups[pid]["answer"] and row.get("answer"):
            groups[pid]["answer"] = row.get("answer")
    return groups


def make_record(group):
    parsed = parse_link(group["link"])
    if parsed is None:
        return None, "unparseable_link"

    year, contest, number = parsed
    statement = clean(group["problem"])
    if not statement:
        return None, "empty_statement"

    solutions = group["solutions"]
    solution = solutions[0] if solutions else ""
    candidates = solutions[1:] if len(solutions) > 1 else []

    answer = group["letter"] or group["answer"] or None

    contest_label = contest_display_name(contest)
    pid = f"AMIO-{contest.replace(' ', '')}-{year}-P{number}"

    rec = {
        "id": pid,
        "statement": statement,
        "solution": solution,
        "candidates": candidates,
        "contest": f"{contest_label} {year}",
        "round": "Individual Round",
        "number": number,
        "answer": answer,
        "sources": [group["link"]],
    }
    return rec, None


def contest_display_name(raw):
    """AHSME/AJHSME stay as-is; AMC/AIME get a space before the number/suffix
    for readability ('AMC 10A' not 'AMC10A'). raw is already space-joined
    from parse_link, e.g. 'AMC 10A', 'AIME I', 'AHSME'."""
    return raw


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv_path", help="Path to the Kaggle AMIO CSV file")
    ap.add_argument("--out", default="kaggle_amio.json",
                     help="Output path for the tagged records (default: %(default)s)")
    ap.add_argument("--dry-run", action="store_true",
                     help="Report counts and a sample record; no tagging, no API calls")
    ap.add_argument("--model", default=tc.MODEL,
                     help="Anthropic model to use for tagging (default: %(default)s)")
    ap.add_argument("--limit", type=int, default=None,
                     help="Optional cap on number of records, for a quick test run")
    ap.add_argument("--contests", default=None,
                     help="Comma-separated contest filter, e.g. 'AMC 8,AMC 10A,AMC 10B,AMC 12A,"
                          "AMC 12B,AIME I,AIME II'. Default: all contests in the file.")
    ap.add_argument("--years", default=None,
                     help="Comma-separated year filter, e.g. '2023' or '2022,2023'. "
                          "Default: all years in the file.")
    args = ap.parse_args()

    print(f"Reading {args.csv_path}...", file=sys.stderr)
    rows = list(load_csv_rows(args.csv_path))
    print(f"Loaded {len(rows)} CSV rows (one per solution).", file=sys.stderr)

    groups = group_by_problem(rows)
    print(f"Grouped into {len(groups)} unique problems.", file=sys.stderr)

    wanted = None
    if args.contests:
        wanted = {c.strip().upper() for c in args.contests.split(",")}

    wanted_years = None
    if args.years:
        wanted_years = {y.strip() for y in args.years.split(",")}

    records, by_contest = [], defaultdict(int)
    skip_reasons = defaultdict(int)

    for group in groups.values():
        rec, skip_reason = make_record(group)
        if skip_reason:
            skip_reasons[skip_reason] += 1
            continue
        contest_only, _, year_only = rec["contest"].rpartition(" ")
        if wanted is not None and contest_only not in wanted:
            continue
        if wanted_years is not None and year_only not in wanted_years:
            continue
        records.append(rec)
        by_contest[rec["contest"]] += 1

    if args.limit:
        records = records[:args.limit]

    print(f"\nUsable records: {len(records)}", file=sys.stderr)
    if skip_reasons:
        print("Skipped:", dict(skip_reasons), file=sys.stderr)
    print(file=sys.stderr)
    for contest, n in sorted(by_contest.items()):
        print(f"  {n:4d}  {contest}", file=sys.stderr)

    if args.dry_run:
        print("\n--dry-run: stopping before tagging/API calls. Sample record:", file=sys.stderr)
        if records:
            print(json.dumps(records[0], indent=2, ensure_ascii=False), file=sys.stderr)
        return

    if not records:
        sys.exit("No records matched — check --contests filter or CSV path.")

    print(f"\nTagging {len(records)} records via {args.model} "
          f"(calls the Anthropic API once per record)...", file=sys.stderr)
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
