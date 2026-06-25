"""
tag_and_compare.py — tagging, combining, and local similarity search for contest problems.

Layer 1 (tagging): each problem -> Anthropic API -> structured tags.
Layer 2 (comparison/practice): local TF-IDF similarity, no torch/sentence-transformers.

Run layer 1:
    python tag_and_compare.py tag raw.json -o tagged.json

Find similar problems:
    python tag_and_compare.py similar tagged.json --to ARML2026-INDIVIDUAL-7 -k 5

Recommend practice:
    python tag_and_compare.py practice tagged.json --area Algebra --technique "Vieta" -k 10
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict, Counter
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------
# AREAS is closed. TECHNIQUES is retained as a seed/fallback for techniques.py,
# but tagging prompts use the live canonical list from techniques.json whenever
# it is available.
AREAS = ["Algebra", "Combinatorics", "Geometry", "Number Theory"]

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
    # --- newer seed tags ---
    "block/period identification", "pairing argument",
    "exhaustive case elimination", "multiplication principle",
]

BRANCHES = {
    'Algebra': {
        'Polynomials': [
            "Vieta's formulas",
            "Newton's sums",
            'symmetric functions',
            'polynomial roots',
            'factor theorem',
            'remainder theorem',
            'rational root theorem',
            "Descartes' rule of signs",
            'Lagrange interpolation',
            'roots of unity',
            'conjugate roots',
            'partial fractions',
            'polynomial factorization',
            'difference of squares',
            'difference of cubes',
            "Simon's Favorite Factoring Trick (SFFT)",
            'characteristic equation',
            'quadratic equations',
            'quadratic formula',
            'discriminant',
            'completing the square',
            'system of linear equations',
            'linear independence',
        ],
        'Inequalities': [
            'AM-GM',
            'weighted AM-GM',
            'Cauchy-Schwarz',
            'triangle inequality (algebra)',
            'power mean inequality',
            'power mean (QM-AM)',
            'AM-HM',
            'rearrangement inequality',
            'SOS (sum of squares)',
            'Jensen',
            'Holder',
            'Minkowski',
            "Schur's inequality",
            'trivial inequality',
            'inequalities',
            'optimization',
            'Lagrange multipliers',
        ],
        'Algebraic Manipulation': [
            'telescoping',
            'substitution',
            'functional equation',
            'Cauchy functional equation',
            'absolute value',
            'floor/ceiling',
            'floor/ceiling functions',
            'logarithms',
            'logarithm properties',
            'exponentials',
            'exponent rules',
            'change of base formula',
            'Abel summation',
            'parametrization',
            'order of operations',
        ],
        'Sequences and Series': [
            'arithmetic sequences',
            'geometric sequences',
            'arithmetic series',
            'geometric series',
            'summation formulas',
            'product formulas',
            'telescoping product',
            'recursion',
            'recurrence relation',
            'linear recurrence',
            'generating functions',
            'matrix exponentiation',
            'Fibonacci/Lucas',
        ],
        'Complex Numbers': [
            'complex numbers',
            "DeMoivre's theorem",
            'roots of unity',
        ],
        'Applied Algebra': [
            'distance-rate-time',
            'work-rate',
            'weighted average',
            'weighted averages',
            'mixture problems',
            'unit conversion',
            'Intermediate Value Theorem',
        ],
    },
    'Combinatorics': {
        'Counting Techniques': [
            'bijection',
            'pigeonhole principle',
            'pigeonhole',
            'stars and bars',
            'inclusion-exclusion',
            'complementary counting',
            'permutations',
            'combinations',
            'permutations/combinations',
            'multinomial theorem',
            'multinomial counting',
            'binomial theorem',
            'double counting',
            'circular arrangements',
            'distinguishable vs indistinguishable',
            'paths on a grid',
            'multiplication principle',
            'casework',
            'symmetry',
        ],
        'Combinatorial Identities': [
            "Vandermonde's identity",
            'Hockey Stick identity',
            "Pascal's triangle",
            "Pascal's identity",
            'Catalan numbers',
            'Stirling numbers',
            'derangements',
        ],
        'Symmetry and Enumeration': [
            "Burnside's lemma",
            'Polya enumeration',
            'symmetry',
        ],
        'Graph Theory': [
            'graph theory',
            'coloring',
            'coloring argument',
            'handshake lemma',
            "Dijkstra's algorithm",
        ],
        'Probability and Expected Value': [
            'linearity of expectation',
            'conditional probability',
            "Bayes' theorem",
            'geometric probability',
            'expected value',
            'variance',
            'random walks',
            'Markov chains',
            'indicator variables',
            'geometric distribution',
            'binomial distribution',
            'probabilistic method',
            'geometric series',
        ],
        'Combinatorial Arguments': [
            'extremal principle',
            'invariants',
            'monovariant',
            'monovariants',
            'principle of reflection',
            'game theory',
            'strategy stealing',
            'induction',
            'strong induction',
            'logical deduction',
            'block/period identification',
            'pairing argument',
            'exhaustive case elimination',
            'greedy algorithm',
            'recursion',
            'generating functions',
        ],
    },
    'Geometry': {
        'Triangle Geometry': [
            'Law of Cosines',
            'Law of Sines',
            'extended Law of Sines',
            "Stewart's theorem",
            'angle bisector theorem',
            'similar triangles',
            'congruent triangles',
            'isosceles triangles',
            'Pythagorean theorem',
            'special right triangles',
            'triangle inequality',
            'mass points',
            'area ratios',
            "Heron's formula",
            'inradius/circumradius formulas',
            "Routh's theorem",
            'British Flag theorem',
        ],
        'Circle Geometry': [
            'Power of a Point',
            'Ptolemy',
            'radical axis',
            'cyclic quadrilateral',
            'tangent-secant',
            'intersecting chords',
            'arcs and inscribed angles',
            'Euler line',
            'nine-point circle',
            'Simson line',
            "Descartes' Circle Theorem",
            'perpendicular bisector',
        ],
        'Angle and Line Theorems': [
            'Menelaus',
            'Ceva',
            'trig Ceva',
            'angle chasing',
            'directed angles',
        ],
        'Transformations': [
            'homothety',
            'spiral similarity',
            'inversion',
            'rotation',
            'reflection',
            'symmetry',
            'complex numbers',
            'vectors',
        ],
        'Coordinate and Vector Methods': [
            'coordinate geometry',
            'distance formula',
            'vectors',
            'shoelace formula',
            "Pick's theorem",
            "Brahmagupta's formula",
            'tangent addition formula',
            'parametrization',
        ],
        'Trigonometry': [
            'trigonometric identities',
            'double-angle',
            'half-angle',
            'product-to-sum',
            'sum-to-product',
            'Law of Sines',
            'Law of Cosines',
            'tangent addition formula',
        ],
        'Solid Geometry': [
            '3D geometry',
            'cross-section',
        ],
    },
    'Number Theory': {
        'Divisibility and Primes': [
            'divisibility',
            'GCD/LCM',
            'prime factorization',
            'Euclidean algorithm',
            "Bezout's identity",
            'Chicken McNugget theorem',
            'Frobenius (Chicken McNugget)',
            'number of divisors',
            'sum of divisors',
            'divisor function',
            'multiplicative functions',
            'perfect squares/cubes',
            'pairing argument',
        ],
        'Modular Arithmetic': [
            'modular arithmetic',
            'Chinese Remainder Theorem',
            "Fermat's little theorem",
            "Euler's theorem",
            "Wilson's theorem",
            'quadratic residues',
            'Legendre symbol',
            'order of an element',
            'primitive roots',
            'modular inverse',
            "Euler's totient function",
        ],
        'Valuations and Lifting': [
            'lifting the exponent (LTE)',
            'p-adic valuation',
            "Legendre's formula",
        ],
        'Diophantine Equations': [
            'Diophantine equations',
            'Diophantine equation',
            'Pell equation',
            'Vieta jumping',
            'infinite descent',
            'well-ordering principle',
            'induction',
            'strong induction',
            "Simon's Favorite Factoring Trick (SFFT)",
        ],
        'Number Representations': [
            'floor sums',
            'digit problems',
            'base conversion',
            'base representation',
            'Fibonacci/Lucas',
        ],
        'Number Theory Arguments': [
            'monovariant',
            'monovariants',
            'invariants',
            'pigeonhole principle',
            'pigeonhole',
            'extremal principle',
        ],
    },
}

GENERIC = ["casework", "substitution", "algebraic manipulation", "computation", "symmetry"]
AREA_SYNONYMS = {"probability": "Combinatorics", "trigonometry": "Geometry", "trig": "Geometry", "analysis": "Algebra"}
MODEL = "claude-sonnet-4-6"


def canonical_techniques() -> list[str]:
    """Return the live canonical technique list from techniques.json.

    Falls back to TECHNIQUES if techniques.py/techniques.json is unavailable, so
    command-line tagging still works in a fresh checkout.
    """
    try:
        import techniques as technique_store
        loaded = technique_store.get_canonical_techniques()
        return loaded or TECHNIQUES
    except Exception:
        return TECHNIQUES


def normalize_areas(areas: list[str] | None) -> list[str]:
    canon = {a.lower(): a for a in AREAS}
    out: list[str] = []
    for a in areas or []:
        key = (a or "").strip().lower()
        mapped = canon.get(key) or AREA_SYNONYMS.get(key)
        if mapped and mapped not in out:
            out.append(mapped)
    return out


def normalize_techniques(techniques: list[str] | None) -> list[str]:
    """Map model-returned technique strings onto the canonical list when possible."""
    out: list[str] = []
    try:
        import techniques as technique_store
    except Exception:
        technique_store = None

    for t in techniques or []:
        if not isinstance(t, str) or not t.strip():
            continue
        name = t.strip()
        if technique_store is not None:
            try:
                match = technique_store.match_technique(name)
                name = match.get("matched") or name
            except Exception:
                pass
        if name not in out:
            out.append(name)
    return out


PROMPT_STATIC = """You are classifying a competition math problem by the concepts it tests.
Do NOT solve the problem yourself, and do NOT judge whether the solution is correct.

Read the SOLUTION one step at a time. For each step, ask: what theorem,
identity, definition, or technique does this step rely on?

Rules for the technique tags — follow these strictly:
- LOAD-BEARING ONLY: include a technique only if some specific step would fail
  without it. If you cannot point to the step that uses it, leave it out. It is
  better to return two precise tags than five loose ones.
- MOST SPECIFIC NAME: prefer the narrowest correct name (e.g. "Power of a Point",
  not "circle properties"; "linearity of expectation", not "probability").
- NO GENERIC FILLER: do not tag routine moves such as {generic} unless that move
  is the CENTRAL idea of the solution. For instance, tag "casework" only when
  enumerating cases is the main solving strategy, not when a couple of cases appear
  in passing.
- NO TOPIC GUESSING: do not add a technique just because the subject could relate
  to it; it must actually appear in the solution.

The text was extracted from a PDF, so notation may be lossy (e.g. "x2" means x
squared); reconstruct the intended math.

Call the record_tags tool with your classification. Field guidance:
- area: 1-2 entries, each from exactly this set: {areas}
- subtopics: 1-3 short noun phrases for the problem's subject.
- techniques: 1-4 load-bearing methods the solution uses, in order. Prefer these
  canonical names where they apply: {techniques}
- difficulty: "easy", "medium", or "hard".
- answer: the final answer as clean LaTeX, read from the solution's result.
- summary: one sentence (<=20 words) on what the problem tests.

If the SOLUTION section below is empty, classify from the statement alone, naming
the techniques a standard solution would require."""

PROMPT_PROBLEM = """PROBLEM STATEMENT:
{statement}

SOLUTION (classify from this, step by step; may be empty):
{solution}"""

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
        "required": ["area", "subtopics", "techniques", "difficulty", "answer", "summary"],
    },
}


def tag(records: list[dict[str, Any]], model: str = MODEL) -> list[dict[str, Any]]:
    from anthropic import Anthropic

    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        sys.exit("Set ANTHROPIC_API_KEY first: export ANTHROPIC_API_KEY='sk-ant-...'")

    client = Anthropic(api_key=key)
    prompt_techniques = canonical_techniques()
    static_text = PROMPT_STATIC.format(areas=AREAS, techniques=prompt_techniques, generic=GENERIC)

    out = []
    for r in records:
        tags: dict[str, Any] = {}
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=600,
                temperature=0,
                tools=[TAG_TOOL],
                tool_choice={"type": "tool", "name": "record_tags"},
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": static_text, "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": PROMPT_PROBLEM.format(
                        statement=r.get("statement", ""), solution=r.get("solution", ""))},
                ]}],
            )
            for block in msg.content:
                if block.type == "tool_use" and block.name == "record_tags":
                    tags = dict(block.input)
                    tags["area"] = normalize_areas(tags.get("area"))
                    tags["techniques"] = normalize_techniques(tags.get("techniques"))
                    break
        except Exception as e:
            print(f"  ! {r.get('id', '<unknown>')}: {e}", file=sys.stderr)
        out.append({**r, **tags})
        print(f"  tagged {r.get('id', '<unknown>')}: {tags.get('area')} / {tags.get('techniques')}", file=sys.stderr)
    return out


def record_from_text(text: str, pid: str = "PASTED-1") -> dict[str, str]:
    parts = re.split(r"(?im)^\s*=*\s*solution\s*=*\s*:?\s*$", text, maxsplit=1)
    if len(parts) == 2:
        stmt, sol = parts
    else:
        m = re.search(r"(?is)\bsolution\b\s*[:.]", text)
        stmt, sol = (text[:m.start()], text[m.end():]) if m else (text, "")
    stmt = re.sub(r"(?im)^\s*=*\s*statement\s*=*\s*:?\s*", "", stmt, count=1)
    return {"id": pid, "statement": " ".join(stmt.split()), "solution": " ".join(sol.split())}


def embed_text(r: dict[str, Any]) -> str:
    techs = "; ".join(r.get("techniques", []) or [])
    areas = "; ".join(r.get("area", []) or [])
    summary = r.get("summary", "") or ""
    statement = r.get("statement", "") or ""
    # Repeating techniques deliberately gives method overlap more weight while
    # staying in a cheap local TF-IDF model.
    return f"{statement}\nsummary: {summary}\nareas: {areas}\ntechniques: {techs}; {techs}; {techs}"


ROUND_BASE = {"team": 3, "individual": 5, "tiebreaker": 6, "relay": 4, "super": 4, "power": 7}
CONTEST_OFFSET = {"arml": 1.0, "mmaths": 0.0}
POSITION_SPREAD = 3.0


def add_difficulty(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: defaultdict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        groups[(r.get("contest"), r.get("round"))].append(r)
    for (contest, rnd), group in groups.items():
        ordered = sorted(group, key=lambda r: r["number"] if isinstance(r.get("number"), int) else 9999)
        n = len(ordered)
        base = ROUND_BASE.get((rnd or "").split()[0].lower() if rnd else "", 4)
        offset = next((v for k, v in CONTEST_OFFSET.items() if k in (contest or "").lower()), 0.0)
        for i, r in enumerate(ordered):
            frac = i / (n - 1) if n > 1 else 0.0
            r["difficulty_score"] = round(min(10.0, max(1.0, base + offset + frac * POSITION_SPREAD)), 1)
    return records


def _row(r: dict[str, Any], score: float | None = None) -> None:
    pct = f"{round(max(0.0, score) * 100)}% match  " if score is not None else ""
    lvl = r.get("difficulty_score")
    lvl = f"L{lvl}" if lvl is not None else r.get("difficulty", "?")
    src = " / ".join(r.get("sources", [r.get("contest", "?")]))
    techs = ", ".join(r.get("techniques", []) or [])
    print(f"  {pct}{r['id']}  [{', '.join(r.get('area', []) or [])} | {lvl}]  ({src})")
    print(f"        {r.get('summary', '')}")
    if techs:
        print(f"        techniques: {techs}")


def _tfidf_rank(pool: list[dict[str, Any]], query: str) -> list[tuple[dict[str, Any], float]]:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except Exception as e:
        sys.exit(f"TF-IDF similarity needs scikit-learn installed: {e}")

    texts = [embed_text(r) for r in pool]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")
    matrix = vectorizer.fit_transform(texts + [query])
    sims = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    ranked = sorted(zip(pool, sims), key=lambda pair: float(pair[1]), reverse=True)
    return [(r, float(s)) for r, s in ranked]


def practice(records: list[dict[str, Any]], like: str | None = None, like_file: str | None = None,
             text: str | None = None, area: str | None = None, technique: str | None = None,
             difficulty: str | None = None, k: int = 5) -> None:
    pool = list(records)
    if area:
        pool = [r for r in pool if area.lower() in [a.lower() for a in r.get("area", []) or []]]
    if technique:
        pool = [r for r in pool if any(technique.lower() in t.lower() for t in r.get("techniques", []) or [])]
    if difficulty:
        pool = [r for r in pool if r.get("difficulty") == difficulty]
    if not pool:
        sys.exit("No problems match those filters.")

    query, exclude_id = None, None
    if like:
        match = next((r for r in records if r.get("id") == like), None)
        if not match:
            sys.exit(f"{like} not found in corpus.")
        query, exclude_id = embed_text(match), like
    elif like_file:
        query = record_from_text(Path(like_file).read_text(encoding="utf-8"))["statement"]
    elif text:
        query = text

    if query is None:
        print(f"\n{len(pool)} problems match:\n")
        for r in pool[:k]:
            _row(r)
        return

    pool = [r for r in pool if r.get("id") != exclude_id]
    print("\nBest matches:\n")
    for r, score in _tfidf_rank(pool, query)[:k]:
        _row(r, score)


def find_similar(records: list[dict[str, Any]], target_id: str, k: int = 5) -> None:
    practice(records, like=target_id, k=k)


def _load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _dump_json(data: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("tag")
    pt.add_argument("infile")
    pt.add_argument("-o", "--out", default="tagged.json")
    pt.add_argument("--model", default=MODEL)
    pt.add_argument("--retag", action="store_true", help="re-tag everything, even problems already in the out file")

    ps = sub.add_parser("similar")
    ps.add_argument("infile")
    ps.add_argument("--to", required=True, help="problem id to compare against")
    ps.add_argument("-k", type=int, default=5)

    po = sub.add_parser("tagone", help="tag a single pasted problem from a text file")
    po.add_argument("infile", help="text file with STATEMENT:/SOLUTION: sections")
    po.add_argument("--id", default="PASTED-1")
    po.add_argument("--model", default=MODEL)
    po.add_argument("--append", help="corpus JSON to add the tagged problem to")

    pp = sub.add_parser("practice", help="recommend practice problems from a tagged corpus")
    pp.add_argument("infile", help="tagged corpus JSON")
    pp.add_argument("--area", help="filter: Algebra / Combinatorics / Geometry / Number Theory")
    pp.add_argument("--technique", help="filter: substring of a technique tag, e.g. 'Power of a Point'")
    pp.add_argument("--difficulty", choices=["easy", "medium", "hard"])
    pp.add_argument("--like", help="rank by similarity to this problem id")
    pp.add_argument("--like-file", dest="like_file", help="rank by similarity to a pasted problem text file")
    pp.add_argument("--text", help="rank by similarity to a free-text description")
    pp.add_argument("-k", type=int, default=5)

    pc = sub.add_parser("combine", help="merge several extracted JSON files into one corpus")
    pc.add_argument("infiles", nargs="+", help="raw/tagged JSON files to merge")
    pc.add_argument("-o", "--out", default="corpus_raw.json")
    pc.add_argument("--keep-duplicates", action="store_true", help="don't merge problems that share the same statement")

    args = ap.parse_args()

    if args.cmd == "tag":
        records = _load_json(args.infile)
        existing: dict[str, dict[str, Any]] = {}
        if args.out and os.path.exists(args.out) and not args.retag:
            for t in _load_json(args.out):
                if t.get("area"):
                    existing[t["id"]] = t
        todo = [r for r in records if r.get("id") not in existing]
        if not todo:
            print("Everything is already tagged — nothing to do (use --retag to redo).")
        newly = tag(todo, args.model) if todo else []
        by_id = {**existing, **{t["id"]: t for t in newly}}
        merged = [by_id.get(r.get("id"), r) for r in records]
        _dump_json(merged, args.out)
        print(f"Wrote {len(merged)} problems ({len(newly)} newly tagged) -> {args.out}")

    elif args.cmd == "similar":
        find_similar(_load_json(args.infile), args.to, args.k)

    elif args.cmd == "tagone":
        rec = record_from_text(Path(args.infile).read_text(encoding="utf-8"), args.id)
        tagged = tag([rec], args.model)[0]
        print(json.dumps(tagged, indent=2, ensure_ascii=False))
        if args.append:
            corpus = _load_json(args.append) if os.path.exists(args.append) else []
            corpus = [c for c in corpus if c.get("id") != tagged["id"]]
            corpus.append(tagged)
            _dump_json(corpus, args.append)
            print(f"\nAppended {tagged['id']} -> {args.append} ({len(corpus)} total)")

    elif args.cmd == "practice":
        practice(_load_json(args.infile), like=args.like, like_file=args.like_file,
                 text=args.text, area=args.area, technique=args.technique,
                 difficulty=args.difficulty, k=args.k)

    elif args.cmd == "combine":
        def norm(s: str | None) -> str:
            return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()

        merged: list[dict[str, Any]] = []
        by_id: set[str] = set()
        by_text: dict[str, int] = {}
        for path in args.infiles:
            for r in _load_json(path):
                r.setdefault("sources", [r.get("contest", "?")])
                if r.get("id") in by_id:
                    print(f"  ! duplicate id {r.get('id')} skipped", file=sys.stderr)
                    continue
                key = norm(r.get("statement"))
                if not args.keep_duplicates and key and key in by_text:
                    kept = merged[by_text[key]]
                    for src in r.get("sources", []):
                        if src not in kept["sources"]:
                            kept["sources"].append(src)
                    print(f"  merged {r.get('id')} into {kept.get('id')} (same problem) -> sources {kept['sources']}", file=sys.stderr)
                    continue
                by_id.add(r.get("id"))
                if key:
                    by_text[key] = len(merged)
                merged.append(r)
        add_difficulty(merged)
        _dump_json(merged, args.out)
        by_contest = Counter(s for r in merged for s in r.get("sources", []))
        print(f"Combined {len(merged)} unique problems -> {args.out}")
        for c, n in sorted(by_contest.items()):
            print(f"  {n:3d}  {c}")


if __name__ == "__main__":
    main()
