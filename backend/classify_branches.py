"""
classify_branches.py — One-shot script that asks Claude to propose branch
groupings for all existing technique tags, then writes the result to
branches.json for your review before anything touches the live corpus.

Run:
    python classify_branches.py

Output: branches.json — a dict mapping each area to its branches, each
branch listing its techniques. Review and edit this file, then run:
    python apply_branches.py

to actually update tag_and_compare.py and stamp the live corpus with branch tags.

No corpus is touched by this script.
"""
import json, os, sys

try:
    from anthropic import Anthropic
except ImportError:
    sys.exit("pip install anthropic first")

AREAS = ["Algebra", "Combinatorics", "Geometry", "Number Theory"]

TECHNIQUES = [
    # Geometry
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
    "rotation", "reflection", "3D geometry", "cross-section",
    "Descartes' Circle Theorem",
    # Algebra
    "Vieta's formulas", "Newton's sums", "symmetric functions",
    "polynomial roots", "factor theorem", "remainder theorem",
    "rational root theorem", "Descartes' rule of signs",
    "Lagrange interpolation", "roots of unity", "conjugate roots",
    "partial fractions", "telescoping", "AM-GM", "Cauchy-Schwarz",
    "triangle inequality (algebra)", "power mean inequality",
    "AM-HM", "rearrangement inequality", "SOS (sum of squares)",
    "functional equation", "logarithms", "exponentials",
    "absolute value", "floor/ceiling", "completing the square",
    "discriminant", "quadratic formula", "substitution", "recursion",
    "generating functions", "linear recurrence", "matrix exponentiation",
    "arithmetic sequences", "geometric sequences",
    "summation formulas", "product formulas", "inequalities",
    "optimization", "Lagrange multipliers",
    # Combinatorics
    "bijection", "pigeonhole principle", "stars and bars",
    "inclusion-exclusion", "Burnside's lemma", "Polya enumeration",
    "graph theory", "coloring", "complementary counting",
    "permutations", "combinations", "multinomial theorem",
    "binomial theorem", "Vandermonde's identity", "Hockey Stick identity",
    "Pascal's triangle", "linearity of expectation",
    "conditional probability", "Bayes' theorem",
    "geometric probability", "expected value", "variance",
    "random walks", "Markov chains", "principle of reflection",
    "Catalan numbers", "Stirling numbers", "derangements",
    "double counting", "extremal principle", "invariants",
    "monovariant", "game theory", "strategy stealing",
    # Number Theory
    "divisibility", "GCD/LCM", "prime factorization",
    "Euclidean algorithm", "Bezout's identity",
    "modular arithmetic", "Chinese Remainder Theorem",
    "Fermat's little theorem", "Euler's theorem", "Wilson's theorem",
    "quadratic residues", "Legendre symbol", "order of an element",
    "primitive roots", "lifting the exponent (LTE)", "p-adic valuation",
    "floor sums", "Legendre's formula", "digit problems",
    "base conversion", "Chicken McNugget theorem",
    "Diophantine equations", "Pell equation", "Vieta jumping",
    "infinite descent", "well-ordering principle",
]

TOOL = {
    "name": "record_branches",
    "description": "Record the proposed branch taxonomy for competition math.",
    "input_schema": {
        "type": "object",
        "properties": {
            "taxonomy": {
                "type": "object",
                "description": (
                    "Keys are the four areas. Values are objects mapping "
                    "branch name -> list of technique names from the provided list."
                ),
                "additionalProperties": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                }
            },
            "unassigned": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Techniques you could not confidently assign to a branch. "
                    "Include any cross-area ones here with a note."
                )
            }
        },
        "required": ["taxonomy", "unassigned"]
    }
}

PROMPT = f"""You are organizing a list of competition math technique tags into
branches within each of the four math olympiad areas.

AREAS: {AREAS}

TECHNIQUES TO CLASSIFY (assign every one to exactly one branch — or to
"unassigned" if it genuinely spans multiple areas with no clear home):

{json.dumps(TECHNIQUES, indent=2)}

INSTRUCTIONS:
- A branch is a broad topic like "Circle Geometry", "Polynomials",
  "Counting & Combinatorics", "Modular Arithmetic" — NOT as specific as
  the techniques themselves, and NOT as broad as the areas.
- Aim for 4-8 branches per area. Merge thin categories rather than leaving
  singletons.
- Every technique in the list must appear in exactly one branch (or in
  "unassigned"). Do not invent techniques; use the exact name strings above.
- Cross-area techniques (e.g. "complex numbers" appears in both Geometry and
  Algebra) should be placed in whichever area most naturally owns them in
  competition math usage, and noted in "unassigned" only if truly ambiguous.
- Branch names should be short noun phrases (2-4 words), title-cased.
- The output must cover all {len(TECHNIQUES)} techniques above — check your
  work before calling the tool.

Call the record_branches tool with your proposed taxonomy."""


def main():
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        sys.exit("Set ANTHROPIC_API_KEY first")

    client = Anthropic(api_key=key)
    print(f"Asking Claude to classify {len(TECHNIQUES)} techniques into branches...",
          flush=True)

    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        temperature=0,
        tools=[TOOL],
        tool_choice={"type": "tool", "name": "record_branches"},
        messages=[{"role": "user", "content": PROMPT}],
    )

    result = None
    for b in msg.content:
        if b.type == "tool_use" and b.name == "record_branches":
            result = dict(b.input)
            break

    if not result:
        sys.exit("No tool call returned — something went wrong")

    taxonomy = result.get("taxonomy", {})
    unassigned = result.get("unassigned", [])

    # Verify coverage
    assigned = {t for area in taxonomy.values()
                  for branch in area.values()
                  for t in branch}
    missing = [t for t in TECHNIQUES if t not in assigned and t not in unassigned]
    if missing:
        print(f"\nWARNING: {len(missing)} techniques not covered in the output:",
              file=sys.stderr)
        for t in missing:
            print(f"  - {t}", file=sys.stderr)

    out = {
        "taxonomy": taxonomy,
        "unassigned": unassigned,
        "_coverage": {
            "total_techniques": len(TECHNIQUES),
            "assigned": len(assigned),
            "unassigned": len(unassigned),
            "missing": missing,
        }
    }

    with open("branches.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # Pretty-print summary
    print("\n=== Proposed branch taxonomy ===\n")
    for area, branches in taxonomy.items():
        print(f"{area}:")
        for branch, techs in branches.items():
            print(f"  {branch} ({len(techs)})")
            for t in techs:
                print(f"    - {t}")
    if unassigned:
        print(f"\nUnassigned ({len(unassigned)}):")
        for t in unassigned:
            print(f"  - {t}")
    print(f"\nWrote branches.json — review and edit, then run apply_branches.py")
    print(f"Coverage: {len(assigned)}/{len(TECHNIQUES)} assigned, "
          f"{len(unassigned)} unassigned, {len(missing)} missing")


if __name__ == "__main__":
    main()
