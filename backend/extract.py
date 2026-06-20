"""
extract.py — Pull competition problems out of an ARML-style contest PDF.

Usage:
    python extract.py ARML_2026_Contest.pdf "Individual Round" --contest "ARML 2026" -o raw.json

Produces a JSON list of records, one per problem:
    {number, statement, answer, solution}

The math comes out lossy (x2 for x squared, a few unmapped glyphs); that's
expected and fine — the tagging step reads through it. This script's job is
just clean segmentation.
"""
import argparse, json, re
import pdfplumber

# Repeated boilerplate to strip from every page.
FOOTER_PATTERNS = [
    re.compile(r"ARML encourages the reproduction.*?educational purposes\.?", re.S),
    re.compile(r"Commercial usage of ARML problems.*?prohibited\.?", re.S),
]
# A line that is just a page number.
PAGE_NUM = re.compile(r"^\s*\d+\s*$")
# Section header on a page's first line, e.g. "6 Individual Round".
SECTION_HEADER = re.compile(r"^\d+\s+(.*?)\s*$")


def clean(text: str) -> str:
    for pat in FOOTER_PATTERNS:
        text = pat.sub("", text)
    lines = [ln for ln in text.splitlines() if not PAGE_NUM.match(ln)]
    return "\n".join(lines).strip()


def page_texts(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        return [(p.extract_text(x_tolerance=1) or "") for p in pdf.pages]


def find_sections(pages):
    """Return ordered list of (title, start_page, end_page_exclusive)."""
    headers = []
    for i, t in enumerate(pages):
        first = t.split("\n")[0] if t else ""
        m = SECTION_HEADER.match(first)
        # Heuristic: real headers start with a capital letter word after the number.
        if m and re.match(r"[A-Z]", m.group(1)):
            headers.append((m.group(1).strip(), i))
    sections = []
    for idx, (title, start) in enumerate(headers):
        end = headers[idx + 1][1] if idx + 1 < len(headers) else len(pages)
        sections.append((title, start, end))
    return sections


def section_text(pages, sections, title):
    for t, start, end in sections:
        if t.lower() == title.lower():
            body = "\n".join(pages[start:end])
            # drop the header line itself
            body = re.sub(r"^\d+\s+" + re.escape(title) + r"\s*", "", body)
            return clean(body)
    return ""


def split_by(marker, text):
    """Split a block into {number: chunk} using markers like 'Problem 3.'."""
    pat = re.compile(rf"{marker}\s+(\d+)\s*\.", re.S)
    out, matches = {}, list(pat.finditer(text))
    for i, m in enumerate(matches):
        num = int(m.group(1))
        chunk_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out[num] = text[m.end():chunk_end].strip()
    return out


def trim_solution_chunk(chunk: str) -> str:
    """Trim accidental bleed-through from ARML solution chunks.

    In some ARML PDFs, a "Solution N." chunk can include the statement for
    the next problem before "Solution N+1." appears. That contaminates tags
    because the tagger reads the full solution. Keep the actual solution and
    cut off any next "Problem K." marker.
    """
    return re.split(r"\bProblem\s+\d+\s*\.", chunk, maxsplit=1)[0].strip()


def extract_answers_positional(pdf_path, sections, round_name):
    """ARML's answers page stacks fractions vertically, which scrambles linear
    text. Grab only words on the same horizontal line as each 'Answer N.' label.
    Clean for simple answers; partial (never cross-contaminated) for stacked
    fractions. The tagging step normalizes the final answer from the solution."""
    title = f"{round_name} Answers"
    rng = next(((s, e) for t, s, e in sections if t.lower() == title.lower()), None)
    if not rng:
        return {}
    answers = {}
    with pdfplumber.open(pdf_path) as pdf:
        for pi in range(rng[0], rng[1]):
            words = pdf.pages[pi].extract_words(x_tolerance=1)
            for i, w in enumerate(words):
                if w["text"] == "Answer" and i + 1 < len(words):
                    num = words[i + 1]["text"].rstrip(".")
                    if not num.isdigit():
                        continue
                    label_top = w["top"]
                    same_line = [
                        x["text"] for x in words[i + 2:]
                        if abs(x["top"] - label_top) < 4 and x["text"] != "Answer"
                    ]
                    # stop at the next label on the same line (rare) — heuristic ok
                    answers[int(num)] = " ".join(same_line).strip()
    return answers


def extract_round(pdf_path, round_name, contest=""):
    pages = page_texts(pdf_path)
    sections = find_sections(pages)

    statements = split_by("Problem", section_text(pages, sections, round_name))
    answers = extract_answers_positional(pdf_path, sections, round_name)
    sol_block = section_text(pages, sections, f"{round_name} Solutions")
    # In the solutions section each entry restates the problem then gives the
    # solution; keep only the text after the matching "Solution N.".
    solutions = {}
    for num, chunk in split_by("Solution", sol_block).items():
        solutions[num] = trim_solution_chunk(chunk)

    records = []
    for num in sorted(statements):
        records.append({
            "id": f"{contest.replace(' ', '')}-{round_name.split()[0].upper()}-{num}",
            "contest": contest,
            "round": round_name,
            "number": num,
            "statement": " ".join(statements[num].split()),
            "answer": " ".join(answers.get(num, "").split()),
            "solution": " ".join(solutions.get(num, "").split()),
        })
    return records


# Rounds that use plain "Problem N." numbering, so the same parser handles them.
# (Relay uses "Problem 1-1.", Super Relay uses bare "1.", and the Power Round is
# one multi-part question — those need their own parsers, added later.)
STANDARD_ROUNDS = ["Team Round", "Individual Round", "Tiebreaker Round"]


def extract_all(pdf_path, contest=""):
    pages = page_texts(pdf_path)
    present = {t for t, _, _ in find_sections(pages)}
    records = []
    for rd in STANDARD_ROUNDS:
        if rd in present:
            records.extend(extract_round(pdf_path, rd, contest))
    return records


# --- MMATHS / Yale format -------------------------------------------------
# A single solutions PDF: problems numbered "1." "2." ... each followed by
# "Proposed by: ...", "Answer: ...", and "Solution: ...". No section headers.
MMATHS_FOOTER = re.compile(r"(?im)^\s*(yale math competitions|november \d{4}|"
                           r"mmaths.*solutions)\s*$")


def extract_mmaths(pdf_path, contest="", round_name="Individual Round"):
    text = "\n".join(page_texts(pdf_path))
    text = MMATHS_FOOTER.sub("", text)
    text = "\n".join(ln for ln in text.splitlines() if not PAGE_NUM.match(ln))

    # Problem boundaries: a line that starts "<n>. " where n increases from 1.
    marks = [(int(m.group(1)), m.start(), m.end())
             for m in re.finditer(r"(?m)^(\d+)\.\s", text)]
    # keep only the strictly increasing 1,2,3,... chain (ignore stray "1." etc.)
    chain, expect = [], 1
    for num, s, e in marks:
        if num == expect:
            chain.append((num, s, e))
            expect += 1

    records = []
    for i, (num, _, body_start) in enumerate(chain):
        end = chain[i + 1][1] if i + 1 < len(chain) else len(text)
        block = text[body_start:end]
        # statement = up to "Proposed by:" or "Answer:"; answer between
        # "Answer:" and "Solution:"; solution = after "Solution:".
        stmt = re.split(r"proposed by:|answer:", block, maxsplit=1,
                        flags=re.I)[0]
        ans = sol = ""
        ma = re.search(r"answer:\s*(.*?)(?=solution:|\Z)", block, re.I | re.S)
        if ma:
            ans = ma.group(1)
        ms = re.search(r"solution:\s*(.*)\Z", block, re.I | re.S)
        if ms:
            sol = ms.group(1)
        records.append({
            "id": f"{contest.replace(' ', '')}-{round_name.split()[0].upper()}-{num}",
            "contest": contest,
            "round": round_name,
            "number": num,
            "statement": " ".join(stmt.split()),
            "answer": " ".join(ans.split()),
            "solution": " ".join(sol.split()),
        })
    return records


# --- AMC / MAA format -----------------------------------------------------
# "Problem N:" markers, answer-choice lines "(A) ... (E)", and "Answer (X): ..."
# blocks whose solutions may contain "OR"-separated alternates. AMC 8/10/12 and
# AIME share this layout.
AMC_HEADER = re.compile(r"(?im)^\s*20\d\d\s+amc\s+\d+\s+[ab]\b.*$")


def parse_amc(text):
    """Segment AMC-style solution text into problem records. Factored out so it
    can be tested on pasted text, not only on a PDF."""
    text = AMC_HEADER.sub("", text)
    text = "\n".join(ln for ln in text.splitlines() if not PAGE_NUM.match(ln))
    marks = [(int(m.group(1)), m.start())
             for m in re.finditer(r"(?m)^Problem\s+(\d+):", text)]
    chain, expect = [], 1
    for num, s in marks:
        if num == expect:
            chain.append((num, s))
            expect += 1
    out = []
    for i, (num, s) in enumerate(chain):
        end = chain[i + 1][1] if i + 1 < len(chain) else len(text)
        block = re.sub(r"^Problem\s+\d+:\s*", "", text[s:end])
        ma = re.search(r"Answer\s*\(([A-E])\)\s*:\s*(.*)\Z", block, re.S)
        if ma:
            letter, solution, stmt_part = ma.group(1), ma.group(2), block[:ma.start()]
        else:
            letter, solution, stmt_part = "", "", block
        # drop the multiple-choice list "(A) ... (E)" from the statement
        cm = re.search(r"\(A\).*?\(E\)", stmt_part, re.S)
        statement = stmt_part[:cm.start()] if cm else stmt_part
        out.append({"number": num,
                    "statement": " ".join(statement.split()),
                    "answer": letter,
                    "solution": " ".join(solution.split())})
    return out


def extract_amc(pdf_path, contest="", round_name="Individual Round"):
    text = "\n".join(page_texts(pdf_path))
    recs = parse_amc(text)
    for r in recs:
        r.update(id=f"{contest.replace(' ', '')}-{round_name.split()[0].upper()}-"
                    f"{r['number']}", contest=contest, round=round_name)
    return recs


# --- LLM fallback: works on ANY layout ------------------------------------
# When no regex format fits (some random contest from the web), hand the raw
# text to the model and let it segment. Costs a few cents per PDF; use it only
# when a free parser doesn't exist for the source.
EXTRACT_TOOL = {
    "name": "record_problems",
    "description": "Record every problem found in the competition text.",
    "input_schema": {
        "type": "object",
        "properties": {
            "problems": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "number": {"type": "string"},
                        "statement": {"type": "string"},
                        "answer": {"type": "string"},
                        "solution": {"type": "string"},
                    },
                    "required": ["number", "statement", "solution"],
                },
            }
        },
        "required": ["problems"],
    },
}

EXTRACT_PROMPT = """Below is text extracted from a math competition PDF. Split it \
into individual problems. For each, capture its number/label, the problem \
statement, the final answer if stated, and the full solution (include all \
alternate solutions). Ignore page headers, footers, page numbers, credits, and \
multiple-choice answer letters unless they are part of the statement. Notation may \
be lossy from PDF extraction — keep it as-is, don't try to fix it. Call the \
record_problems tool with every problem you find.

TEXT:
"""


def extract_llm(pdf_path, contest="", round_name="", model="claude-sonnet-4-6"):
    import os
    import sys
    from anthropic import Anthropic
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        sys.exit("Set ANTHROPIC_API_KEY first:  export ANTHROPIC_API_KEY='sk-ant-...'")
    text = "\n".join(page_texts(pdf_path))
    text = "\n".join(ln for ln in text.splitlines() if not PAGE_NUM.match(ln))

    client = Anthropic(api_key=key)
    msg = client.messages.create(
        model=model,
        max_tokens=16000,
        tools=[EXTRACT_TOOL],
        tool_choice={"type": "tool", "name": "record_problems"},
        messages=[{"role": "user", "content": EXTRACT_PROMPT + text}],
    )
    problems = []
    for b in msg.content:
        if b.type == "tool_use" and b.name == "record_problems":
            problems = b.input.get("problems", [])
    records = []
    for p in problems:
        num = p.get("number", "")
        records.append({
            "id": f"{contest.replace(' ', '')}-"
                  f"{(round_name or 'P').split()[0].upper()}-{num}",
            "contest": contest,
            "round": round_name,
            "number": int(num) if str(num).isdigit() else num,
            "statement": " ".join(p.get("statement", "").split()),
            "answer": " ".join(p.get("answer", "").split()),
            "solution": " ".join(p.get("solution", "").split()),
        })
    return records


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("round", help='e.g. "Individual Round", or "all" for every '
                                  'standard round in the PDF (arml format)')
    ap.add_argument("--contest", default="")
    ap.add_argument("--format", default="arml",
                    choices=["arml", "mmaths", "amc", "llm"],
                    help="PDF layout: arml (Problem N. + sections), mmaths "
                         "(Proposed by/Answer/Solution), amc (Problem N: + "
                         "choices), or llm (universal fallback, uses the API)")
    ap.add_argument("--model", default="claude-sonnet-4-6",
                    help="model for --format llm")
    ap.add_argument("-o", "--out", default="raw.json")
    args = ap.parse_args()

    if args.format == "mmaths":
        recs = extract_mmaths(args.pdf, args.contest, args.round)
    elif args.format == "amc":
        recs = extract_amc(args.pdf, args.contest, args.round)
    elif args.format == "llm":
        recs = extract_llm(args.pdf, args.contest, args.round, args.model)
    elif args.round.lower() == "all":
        recs = extract_all(args.pdf, args.contest)
    else:
        recs = extract_round(args.pdf, args.round, args.contest)
    with open(args.out, "w") as f:
        json.dump(recs, f, indent=2, ensure_ascii=False)
    print(f"Extracted {len(recs)} problems -> {args.out}")
