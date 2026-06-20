"""
amio_import.py — Import problems from the AIMO-style CSV (AoPS-scraped).

CSV columns: problem_id, link, problem, solution, letter, answer
Each problem_id appears on MULTIPLE rows — one per community-submitted
solution to the same problem. This module groups those rows, derives the
contest name + problem number from `link` (the only place that info lives),
and tags each problem using a combined view of ALL its solutions, not just
one, so the tagger can pick the cleanest/most representative one.

Workflow exposed:
  parse_csv(path)        -> list of grouped problem dicts (ungraded, untagged)
  list_contests(records)  -> {"2024 AMC 8": 25, "2023 AMC 10A": 25, ...}
  records_for_contest(records, contest) -> just that contest's problems
  tag_aimo_records(records, model)      -> tagged, ready to merge into corpus

The "AMIO" name is intentionally NOT used anywhere in generated ids or
contest names — that's just the dataset's internal codename, not something
that should leak into problem identifiers shown to users.
"""
import csv
import io
import re
from collections import defaultdict

# Matches e.g. ".../2024_AMC_8_Problems/Problem_1" or
# ".../2023_AMC_10A_Problems/Problem_19"
AMC_LINK_PATTERN = re.compile(
    r"(\d{4})_AMC_(\d+[AB]?)_Problems/Problem_(\d+)", re.IGNORECASE
)

# AJHSME = American Junior High School Mathematics Examination, the AMC 8's
# name before it was renamed in 1999. Kept as its own contest label (not
# folded into "AMC 8") since that's the exam's actual historical name —
# e.g. ".../1998_AJHSME_Problems/Problem_2" -> "1998 AJHSME".
AJHSME_LINK_PATTERN = re.compile(
    r"(\d{4})_AJHSME_Problems/Problem_(\d+)", re.IGNORECASE
)

# AHSME (American High School Mathematics Examination) was the AMC 10/12's
# name before the 2000 split — e.g.
# ".../1999_AHSME_Problems/Problem_4" -> "1999 AHSME".
AHSME_LINK_PATTERN = re.compile(
    r"(\d{4})_AHSME_Problems/Problem_(\d+)", re.IGNORECASE
)

# AIME (American Invitational Mathematics Examination) has two sittings per
# year, I and II, e.g. ".../2023_AIME_I_Problems/Problem_7" or
# ".../2023_AIME_II_Problems/Problem_12". Pre-2000ish years only had one
# sitting and may appear as plain ".../1998_AIME_Problems/Problem_5" (no
# I/II) — the "(?:_(I{1,2}))?" group is optional to cover both shapes.
AIME_LINK_PATTERN = re.compile(
    r"(\d{4})_AIME(?:_(I{1,2}))?_Problems/Problem_(\d+)", re.IGNORECASE
)


def _parse_link(link: str):
    """Returns (contest_name, number) or (None, None) if the link doesn't
    match a known AoPS URL shape. Other contest types (ARML, USAMO, etc.)
    would need their own pattern added here later."""
    link = link or ""

    m = AMC_LINK_PATTERN.search(link)
    if m:
        year, variant, num = m.groups()
        return f"{year} AMC {variant.upper()}", int(num)

    m = AJHSME_LINK_PATTERN.search(link)
    if m:
        year, num = m.groups()
        return f"{year} AJHSME", int(num)

    m = AHSME_LINK_PATTERN.search(link)
    if m:
        year, num = m.groups()
        return f"{year} AHSME", int(num)

    m = AIME_LINK_PATTERN.search(link)
    if m:
        year, sitting, num = m.groups()
        contest = f"{year} AIME {sitting.upper()}" if sitting else f"{year} AIME"
        return contest, int(num)

    return None, None


def _contest_slug(contest: str) -> str:
    """Stable id prefix for a contest, e.g. '2024 AMC 8' -> 'AMC2024-8',
    '1998 AJHSME' -> 'AJHSME1998', '2023 AIME I' -> 'AIME2023-I',
    '1998 AIME' (no sitting) -> 'AIME1998'."""
    parts = contest.split()
    year = parts[0]
    if len(parts) == 2:
        # No variant/sitting segment, e.g. "1998 AJHSME" or "1998 AIME"
        # -> ["1998", "AJHSME"] / ["1998", "AIME"]
        return f"{parts[1]}{year}"
    variant = "".join(parts[2:])
    return f"{parts[1]}{year}-{variant}"


def parse_csv(path_or_buffer) -> list:
    """Read the CSV and group rows by problem_id. Returns one dict per
    distinct problem:
        {problem_id, link, contest, number, statement, answer, letter,
         solutions: [str, ...]}
    Rows whose link doesn't parse into a known contest format are skipped
    (reported in the returned list's missing problems via the caller, if
    needed) rather than silently mis-filed."""
    if hasattr(path_or_buffer, "read"):
        f = path_or_buffer
    else:
        f = open(path_or_buffer, newline="", encoding="utf-8")

    grouped = {}
    order = []
    try:
        reader = csv.DictReader(f)
        for row in reader:
            pid = (row.get("problem_id") or "").strip()
            if not pid:
                continue
            if pid not in grouped:
                contest, number = _parse_link(row.get("link", ""))
                grouped[pid] = {
                    "problem_id": pid,
                    "link": (row.get("link") or "").strip(),
                    "contest": contest,
                    "number": number,
                    "statement": (row.get("problem") or "").strip(),
                    "letter": (row.get("letter") or "").strip(),
                    "answer": (row.get("answer") or "").strip(),
                    "solutions": [],
                }
                order.append(pid)
            sol = (row.get("solution") or "").strip()
            if sol:
                grouped[pid]["solutions"].append(sol)
    finally:
        if not hasattr(path_or_buffer, "read"):
            f.close()

    return [grouped[pid] for pid in order]


def list_contests(records: list) -> dict:
    """{"2024 AMC 8": 25, ...} — counts only problems with a recognized
    contest (unparseable links are excluded, not bucketed under None)."""
    counts = defaultdict(int)
    for r in records:
        if r["contest"]:
            counts[r["contest"]] += 1
    return dict(sorted(counts.items()))


def records_for_contest(records: list, contest: str) -> list:
    return [r for r in records if r["contest"] == contest]


def _to_tag_input(r: dict) -> dict:
    """Shape one grouped AIMO record into the {id, statement, solution}
    input tag_and_compare.tag() expects. All solutions are concatenated,
    clearly labeled, so the model can read across alternates rather than
    being shown only the first (possibly weakest) one."""
    pid = f"{_contest_slug(r['contest'])}-INDIVIDUAL-{r['number']}"
    combined_solutions = "\n\n---\n\n".join(
        f"Solution {i+1}:\n{s}" for i, s in enumerate(r["solutions"])
    ) or "(no solution text available)"
    return {
        "id": pid,
        "statement": r["statement"],
        "solution": combined_solutions,
        "_aimo": r,  # stashed for re-attaching metadata after tagging
    }


def tag_aimo_records(records: list, model=None) -> list:
    """Tags a list of grouped AIMO records (from records_for_contest) via
    the Anthropic API, reusing tag_and_compare's taxonomy/prompt/tool. The
    final 'answer' field comes from the tagger's LaTeX-cleaned read of the
    solutions (consistent with how the rest of the corpus stores answers),
    falling back to the CSV's raw answer/letter if tagging didn't return one.
    Each output record also keeps the AoPS link in `sources` and the original
    community solutions in `candidates`, matching records you've already
    confirmed are in the corpus (e.g. AMIO-AMC10A-2023-P1)."""
    import tag_and_compare as tc

    to_tag = [_to_tag_input(r) for r in records if r["contest"]]
    tagged = tc.tag(
        [{"id": t["id"], "statement": t["statement"], "solution": t["solution"]}
         for t in to_tag],
        model or tc.MODEL,
    )

    out = []
    for t, tagged_rec in zip(to_tag, tagged):
        r = t["_aimo"]
        final = dict(tagged_rec)
        if not final.get("answer"):
            final["answer"] = r["answer"] or r["letter"]
        final["contest"] = r["contest"]
        final["round"] = "Individual Round"
        final["number"] = r["number"]
        final["sources"] = [r["link"]] if r["link"] else [r["contest"]]
        # Keep the alternate community solutions around for reference, same
        # field name already used elsewhere in the corpus for this purpose.
        if len(r["solutions"]) > 1:
            final["candidates"] = r["solutions"][1:]
        out.append(final)
    return out
