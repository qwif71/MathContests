"""
tag_and_compare.py — Two layers on top of extract.py's output:

  LAYER 1 (tagging):    each problem -> Anthropic API -> structured tags
                        ("what's tested") + a normalized answer in LaTeX.
  LAYER 2 (comparison): sentence-transformers embeddings -> find similar
                        problems by cosine similarity.

Run layer 1 (needs ANTHROPIC_API_KEY in your environment):
    python tag_and_compare.py tag raw.json -o tagged.json

Run layer 2 (needs the problems already tagged):
    python tag_and_compare.py similar tagged.json --to ARML2026-INDIVIDUAL-7 -k 5

Install:  pip install anthropic sentence-transformers
"""
import argparse, json, os, re, sys

# ---------------------------------------------------------------------------
# Taxonomy — edit freely. AREAS is a closed list; TECHNIQUES is open vocabulary
# (the model may introduce new ones), but listing canonical names keeps the
# tags consistent across contests so the comparison layer stays meaningful.
# ---------------------------------------------------------------------------
# The competition "big four" — coarse label. Probability folds under
# Combinatorics; trigonometry shows up under Geometry/Algebra (as a technique).
AREAS = ["Algebra", "Combinatorics", "Geometry", "Number Theory"]

# Open vocabulary, but these canonical names keep tags consistent across
# contests so the comparison layer stays meaningful. Add to this freely.
TECHNIQUES = [
    # --- Geometry ---
    "Power of a Point", "Menelaus", "Ceva", "trig Ceva", "Law of Cosines",
    "Law of Sines", "extended Law of Sines", "Ptolemy", "Stewart's theorem",
    "angle bisector theorem", "similar triangles", "congruent triangles",
    "isosceles triangles", "Pythagorean theorem", "special right triangles",
    "coordinate geometry", "distance formula", "trigonometric identities",
    "double-angle", "half-angle", "product-to-sum", "sum-to-product",
    "angle chasing", "directed angles", "triangle inequality", "vectors",
    "complex numbers", "DeMoivre's theorem", "Heron's formula",
    "Brahmagupta's formula", "shoelace formula", "Pick's theorem",
    "mass points", "area ratios", "homothety", "spiral similarity",
    "inversion", "radical axis", "cyclic quadrilateral", "tangent-secant",
    "intersecting chords", "arcs and inscribed angles",
    "inradius/circumradius formulas", "Euler line", "nine-point circle",
    "Simson line", "British Flag theorem", "Routh's theorem",
    "rotation", "reflection", "3D geometry", "cross-section", "Descartes' Circle Theorem",
    # --- Algebra ---
    "Vieta's formulas", "Newton's sums", "symmetric functions",
    "polynomial roots", "factor theorem", "remainder theorem",
    "rational root theorem", "Descartes' rule of signs",
    "Lagrange interpolation", "roots of unity", "conjugate roots",
    "completing the square", "discriminant", "quadratic formula",
    "system of linear equations", "Simon's Favorite Factoring Trick (SFFT)",
    "AM-GM", "weighted AM-GM", "Cauchy-Schwarz", "power mean (QM-AM)",
    "AM-HM", "Jensen", "Holder", "Minkowski", "rearrangement inequality",
    "Schur's inequality", "SOS (sum of squares)", "trivial inequality",
    "Abel summation", "geometric series", "arithmetic series", "telescoping",
    "telescoping product", "binomial theorem", "partial fractions",
    "functional equation", "Cauchy functional equation",
    "recurrence relation", "characteristic equation", "logarithm properties",
    "absolute value", "floor/ceiling functions", "exponent rules",
    "distance-rate-time", "work-rate", "weighted average",
    "Intermediate Value Theorem", "substitution", "quadratic equations", "mixture problems",
    "weighted averages", "polynomial factorization", "parametrization",
    # --- Number Theory ---
    "modular arithmetic", "Chinese Remainder Theorem",
    "Fermat's little theorem", "Euler's theorem", "Euler's totient function",
    "Wilson's theorem", "base representation", "Diophantine equation",
    "Pell equation", "Vieta jumping", "Fibonacci/Lucas",
    "divisibility", "gcd/lcm", "Bezout's identity", "Euclidean algorithm",
    "prime factorization", "number of divisors", "sum of divisors",
    "Legendre's formula", "lifting the exponent (LTE)", "order of an element",
    "primitive roots", "quadratic residues", "Legendre symbol",
    "divisor function", "multiplicative functions",
    "Frobenius (Chicken McNugget)", "modular inverse", "p-adic valuation",
    "perfect squares/cubes", "digit problems",
    # --- Combinatorics / Probability ---
    "permutations/combinations", "circular arrangements",
    "distinguishable vs indistinguishable", "paths on a grid",
    "inclusion-exclusion", "complementary counting", "stars and bars",
    "pigeonhole", "linearity of expectation", "expected value",
    "indicator variables", "conditional probability", "Bayes' theorem",
    "geometric distribution", "binomial distribution", "recursion",
    "double counting", "generating functions", "bijection",
    "multinomial counting", "induction", "strong induction",
    "hockey stick identity", "Pascal's identity", "Vandermonde's identity",
    "Catalan numbers", "derangements", "Burnside's lemma", "symmetry",
    "extremal principle", "invariants", "monovariants", "coloring argument",
    "graph theory", "handshake lemma", "probabilistic method",
    "casework", "logical deduction",
    # --- NEW ---
    "block/period identification", "pairing argument",
    "exhaustive case elimination", "multiplication principle",
]
# Generic moves that appear in almost every hard solution. Tag these ONLY when
# they are the central idea, never as incidental filler (see prompt rule).
GENERIC = ["casework", "substitution", "algebraic manipulation", "computation",
           "symmetry"]

# Fold common out-of-taxonomy labels into the big four instead of trusting the
# model to honor the enum (weaker models sometimes don't). Anything not in AREAS
# and not mapped here is dropped, so a stray "Probability" can't leak through.
AREA_SYNONYMS = {"probability": "Combinatorics", "trigonometry": "Geometry",
                 "trig": "Geometry", "analysis": "Algebra"}


def normalize_areas(areas):
    canon = {a.lower(): a for a in AREAS}
    out = []
    for a in areas or []:
        mapped = canon.get(a.strip().lower()) or AREA_SYNONYMS.get(a.strip().lower())
        if mapped and mapped not in out:
            out.append(mapped)
    return out

MODEL = "claude-sonnet-4-6"   # good cost/quality for tagging; swap to a
                              # haiku model for cheap bulk runs.

# The prompt is split so the big invariant part (rules + the 136-technique list)
# can be marked cacheable: it's identical on every call, so after the first
# problem the rest read it at ~10% the input price instead of paying full freight.
PROMPT_STATIC = """You are classifying a competition math problem by the concepts \
it tests. Do NOT solve the problem yourself, and do NOT judge whether the solution \
is correct.

Read the SOLUTION one step at a time. For each step, ask: what theorem, \
identity, definition, or technique does this step rely on?

Rules for the technique tags — follow these strictly:
- LOAD-BEARING ONLY: include a technique only if some specific step would fail \
without it. If you cannot point to the step that uses it, leave it out. It is \
better to return two precise tags than five loose ones.
- MOST SPECIFIC NAME: prefer the narrowest correct name (e.g. "Power of a Point", \
not "circle properties"; "linearity of expectation", not "probability").
- NO GENERIC FILLER: do not tag routine moves such as {generic} unless that move \
is the CENTRAL idea of the solution. For instance, tag "casework" only when \
enumerating cases is the main solving strategy, not when a couple of cases appear \
in passing.
- NO TOPIC GUESSING: do not add a technique just because the subject could relate \
to it; it must actually appear in the solution.

The text was extracted from a PDF, so notation may be lossy (e.g. "x2" means x \
squared); reconstruct the intended math.

Call the record_tags tool with your classification. Field guidance:
- area: 1-2 entries, each from exactly this set: {areas}
- subtopics: 1-3 short noun phrases for the problem's subject.
- techniques: 1-4 load-bearing methods the solution uses, in order. Prefer these \
canonical names where they apply: {techniques}
- difficulty: "easy", "medium", or "hard".
- answer: the final answer as clean LaTeX, read from the solution's result.
- summary: one sentence (<=20 words) on what the problem tests.

If the SOLUTION section below is empty, classify from the statement alone, naming \
the techniques a standard solution would require."""

PROMPT_PROBLEM = """PROBLEM STATEMENT:
{statement}

SOLUTION (classify from this, step by step; may be empty):
{solution}"""

# Schema handed to the model as a tool. Forcing a tool call means the SDK returns
# an already-valid dict — no JSON string to parse, so malformed-JSON failures
# (stray quotes, braces in LaTeX, trailing commas) simply cannot happen.
TAG_TOOL = {
    "name": "record_tags",
    "description": "Record the classification tags for one competition math problem.",
    "input_schema": {
        "type": "object",
        "properties": {
            "area": {"type": "array", "items": {"type": "string", "enum": AREAS}},
            "subtopics": {"type": "array", "items": {"type": "string"}},
            "techniques": {"type": "array", "items": {"type": "string"}},
            "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
            "answer": {"type": "string"},
            "summary": {"type": "string"},
        },
        "required": ["area", "subtopics", "techniques", "difficulty",
                     "answer", "summary"],
    },
}


def tag(records, model=MODEL):
    from anthropic import Anthropic
    # .strip() guards against a stray newline/space on the key (a common
    # copy-paste mishap that makes the API reject the auth header).
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        sys.exit("Set ANTHROPIC_API_KEY first:  export ANTHROPIC_API_KEY='sk-ant-...'")
    client = Anthropic(api_key=key)
    # Format the invariant prefix once and mark it cacheable.
    static_text = PROMPT_STATIC.format(
        areas=AREAS, techniques=TECHNIQUES, generic=GENERIC)

    out = []
    for r in records:
        tags = {}
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=600,
                temperature=0,                       # reproducible run-to-run
                tools=[TAG_TOOL],
                tool_choice={"type": "tool", "name": "record_tags"},
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": static_text,
                     "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": PROMPT_PROBLEM.format(
                        statement=r["statement"], solution=r.get("solution", ""))},
                ]}],
            )
            for b in msg.content:
                if b.type == "tool_use" and b.name == "record_tags":
                    tags = dict(b.input)            # already a validated dict
                    tags["area"] = normalize_areas(tags.get("area"))
                    break
        except Exception as e:
            print(f"  ! {r['id']}: {e}", file=sys.stderr)
        out.append({**r, **tags})
        print(f"  tagged {r['id']}: {tags.get('area')} / "
              f"{tags.get('techniques')}", file=sys.stderr)
    return out


def record_from_text(text, pid="PASTED-1"):
    """Build one problem record from pasted text. Recognizes 'STATEMENT:' and
    'SOLUTION:' headers (also '=== SOLUTION ===' or an inline 'Solution.'). If no
    solution is present, the solution is left empty and tagging falls back to the
    statement."""
    parts = re.split(r"(?im)^\s*=*\s*solution\s*=*\s*:?\s*$", text, maxsplit=1)
    if len(parts) == 2:
        stmt, sol = parts
    else:
        m = re.search(r"(?is)\bsolution\b\s*[:.]", text)
        stmt, sol = (text[:m.start()], text[m.end():]) if m else (text, "")
    stmt = re.sub(r"(?im)^\s*=*\s*statement\s*=*\s*:?\s*", "", stmt, count=1)
    return {"id": pid,
            "statement": " ".join(stmt.split()),
            "solution": " ".join(sol.split())}


def embed_text(r):
    """What we hand to the embedder. Repeating techniques nudges the vectors
    toward method-similarity rather than surface-word similarity."""
    techs = "; ".join(r.get("techniques", []))
    return f"{r['statement']} || techniques: {techs}; {techs}"


# Difficulty from provenance, not from solving. Each round has a base level, and
# a problem's position within its round bumps it up (problem 1 easiest, last
# hardest). These numbers are a rough prior — tune them to your own sense of the
# contests. CONTEST_OFFSET lets you say one contest runs harder than another.
ROUND_BASE = {"team": 3, "individual": 5, "tiebreaker": 6,
              "relay": 4, "super": 4, "power": 7}
CONTEST_OFFSET = {"arml": 1.0, "mmaths": 0.0}      # by contest-name substring
POSITION_SPREAD = 3.0                              # points across a round


def add_difficulty(records):
    """Stamp each record with difficulty_score (1-10) from its round and its
    position within that round. Grouped per (contest, round) so position is
    relative to that specific round's length."""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in records:
        groups[(r.get("contest"), r.get("round"))].append(r)
    for (contest, rnd), group in groups.items():
        ordered = sorted(group, key=lambda r: r["number"]
                         if isinstance(r.get("number"), int) else 9999)
        n = len(ordered)
        base = ROUND_BASE.get((rnd or "").split()[0].lower() if rnd else "", 4)
        offset = next((v for k, v in CONTEST_OFFSET.items()
                       if k in (contest or "").lower()), 0.0)
        for i, r in enumerate(ordered):
            frac = i / (n - 1) if n > 1 else 0.0
            score = base + offset + frac * POSITION_SPREAD
            r["difficulty_score"] = round(min(10.0, max(1.0, score)), 1)
    return records


def _row(r, score=None):
    pct = f"{round(max(0.0, score) * 100)}% match  " if score is not None else ""
    lvl = r.get("difficulty_score")
    lvl = f"L{lvl}" if lvl is not None else r.get("difficulty", "?")
    src = " / ".join(r.get("sources", [r.get("contest", "?")]))
    techs = ", ".join(r.get("techniques", []))
    print(f"  {pct}{r['id']}  [{', '.join(r.get('area', []))} | {lvl}]  ({src})")
    print(f"        {r.get('summary', '')}")
    if techs:
        print(f"        techniques: {techs}")


def practice(records, like=None, like_file=None, text=None,
             area=None, technique=None, difficulty=None, k=5):
    """Recommend practice problems. Tag filters (area/technique/difficulty) cut
    the pool; a query (an existing id via `like`, a pasted problem via
    `like_file`, or free text via `text`) ranks what remains by closeness."""
    pool = list(records)
    if area:
        pool = [r for r in pool
                if area.lower() in [a.lower() for a in r.get("area", [])]]
    if technique:
        pool = [r for r in pool if any(technique.lower() in t.lower()
                                       for t in r.get("techniques", []))]
    if difficulty:
        pool = [r for r in pool if r.get("difficulty") == difficulty]
    if not pool:
        sys.exit("No problems match those filters.")

    # Build the query string, if any.
    query, exclude_id = None, None
    if like:
        match = next((r for r in records if r["id"] == like), None)
        if not match:
            sys.exit(f"{like} not found in corpus.")
        query, exclude_id = embed_text(match), like
    elif like_file:
        query = record_from_text(open(like_file).read())["statement"]
    elif text:
        query = text

    if query is None:                       # no query → just list the filtered set
        print(f"\n{len(pool)} problems match:\n")
        for r in pool[:k]:
            _row(r)
        return

    from sentence_transformers import SentenceTransformer
    import numpy as np
    model = SentenceTransformer("all-MiniLM-L6-v2")
    pool = [r for r in pool if r["id"] != exclude_id]
    pv = model.encode([embed_text(r) for r in pool], normalize_embeddings=True)
    qv = model.encode([query], normalize_embeddings=True)[0]
    sims = pv @ qv
    order = np.argsort(-sims)
    print(f"\nBest matches:\n")
    for j in order[:k]:
        _row(pool[j], float(sims[j]))


def find_similar(records, target_id, k=5):
    """Thin wrapper kept for the `similar` command: rank by closeness to one id."""
    practice(records, like=target_id, k=k)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("tag")
    pt.add_argument("infile")
    pt.add_argument("-o", "--out", default="tagged.json")
    pt.add_argument("--model", default=MODEL)
    pt.add_argument("--retag", action="store_true",
                    help="re-tag everything, even problems already in the out file")

    ps = sub.add_parser("similar")
    ps.add_argument("infile")
    ps.add_argument("--to", required=True, help="problem id to compare against")
    ps.add_argument("-k", type=int, default=5)

    po = sub.add_parser("tagone",
                        help="tag a single pasted problem from a text file")
    po.add_argument("infile", help="text file with STATEMENT:/SOLUTION: sections")
    po.add_argument("--id", default="PASTED-1")
    po.add_argument("--model", default=MODEL)
    po.add_argument("--append", help="corpus JSON to add the tagged problem to")

    pp = sub.add_parser("practice",
                        help="recommend practice problems from a tagged corpus")
    pp.add_argument("infile", help="tagged corpus JSON")
    pp.add_argument("--area", help="filter: Algebra / Combinatorics / Geometry / "
                                   "Number Theory")
    pp.add_argument("--technique", help="filter: substring of a technique tag, "
                                        "e.g. 'Power of a Point'")
    pp.add_argument("--difficulty", choices=["easy", "medium", "hard"])
    pp.add_argument("--like", help="rank by similarity to this problem id")
    pp.add_argument("--like-file", dest="like_file",
                    help="rank by similarity to a pasted problem (text file)")
    pp.add_argument("--text", help="rank by similarity to a free-text description")
    pp.add_argument("-k", type=int, default=5)

    pc = sub.add_parser("combine",
                        help="merge several extracted JSON files into one corpus")
    pc.add_argument("infiles", nargs="+", help="raw/tagged JSON files to merge")
    pc.add_argument("-o", "--out", default="corpus_raw.json")
    pc.add_argument("--keep-duplicates", action="store_true",
                    help="don't merge problems that share the same statement")

    args = ap.parse_args()

    if args.cmd == "tag":
        records = json.load(open(args.infile))
        # Skip problems already tagged in the out file (unless --retag). This is
        # the main cost saver: re-running only pays for genuinely new work.
        existing = {}
        if args.out and os.path.exists(args.out) and not args.retag:
            for t in json.load(open(args.out)):
                if t.get("area"):            # has real tags
                    existing[t["id"]] = t
        todo = [r for r in records if r["id"] not in existing]
        if not todo:
            print("Everything is already tagged — nothing to do "
                  "(use --retag to redo).")
        newly = tag(todo, args.model) if todo else []
        by_id = {**existing, **{t["id"]: t for t in newly}}
        merged = [by_id.get(r["id"], r) for r in records]   # input order
        json.dump(merged, open(args.out, "w"), indent=2, ensure_ascii=False)
        print(f"Wrote {len(merged)} problems "
              f"({len(newly)} newly tagged) -> {args.out}")

    elif args.cmd == "similar":
        records = json.load(open(args.infile))
        find_similar(records, args.to, args.k)

    elif args.cmd == "tagone":
        rec = record_from_text(open(args.infile).read(), args.id)
        tagged = tag([rec], args.model)[0]
        print(json.dumps(tagged, indent=2, ensure_ascii=False))
        if args.append:
            corpus = json.load(open(args.append)) if os.path.exists(args.append) else []
            corpus = [c for c in corpus if c.get("id") != tagged["id"]]  # replace dupes
            corpus.append(tagged)
            json.dump(corpus, open(args.append, "w"), indent=2, ensure_ascii=False)
            print(f"\nAppended {tagged['id']} -> {args.append} "
                  f"({len(corpus)} total)")

    elif args.cmd == "practice":
        records = json.load(open(args.infile))
        practice(records, like=args.like, like_file=args.like_file,
                 text=args.text, area=args.area, technique=args.technique,
                 difficulty=args.difficulty, k=args.k)

    elif args.cmd == "combine":
        def norm(s):                      # for matching the SAME problem across contests
            return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()
        merged, by_id, by_text = [], set(), {}
        for path in args.infiles:
            for r in json.load(open(path)):
                r.setdefault("sources", [r.get("contest", "?")])
                if r["id"] in by_id:
                    print(f"  ! duplicate id {r['id']} skipped", file=sys.stderr)
                    continue
                key = norm(r.get("statement"))
                if not args.keep_duplicates and key and key in by_text:
                    kept = merged[by_text[key]]
                    for src in r["sources"]:
                        if src not in kept["sources"]:
                            kept["sources"].append(src)
                    print(f"  merged {r['id']} into {kept['id']} "
                          f"(same problem) -> sources {kept['sources']}",
                          file=sys.stderr)
                    continue
                by_id.add(r["id"])
                if key:
                    by_text[key] = len(merged)
                merged.append(r)
        add_difficulty(merged)            # stamp difficulty_score from provenance
        json.dump(merged, open(args.out, "w"), indent=2, ensure_ascii=False)
        from collections import Counter
        by_contest = Counter(s for r in merged for s in r["sources"])
        print(f"Combined {len(merged)} unique problems -> {args.out}")
        for c, n in sorted(by_contest.items()):
            print(f"  {n:3d}  {c}")
