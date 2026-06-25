"""
apply_branches.py — After reviewing branches.json, run this to:

  1. Stamp every record in tagged.json with a `branches` list field based
     on which branches its techniques belong to. A technique can appear in
     multiple branches, so `branches` is always a list (possibly length 1).
     Records with no matching technique get `branches` inferred from area
     alone (first branch for that area), or [] if area is also absent.
  2. Inserts/replaces a BRANCHES constant in tag_and_compare.py.
  3. Backs up tagged.json before writing.

Run:
    python3 apply_branches.py --dry-run   # preview without writing
    python3 apply_branches.py             # apply
"""
import argparse, json, os, shutil, sys, re
from collections import defaultdict


def load_taxonomy(path="branches.json"):
    with open(path) as f:
        return json.load(f)["taxonomy"]


def build_lookup(taxonomy):
    """
    technique_to_branches: lowercase technique -> list of branch names
      (list because a technique can appear in multiple branches)
    area_default_branch: area -> first branch name (fallback when no
      technique matches)
    """
    technique_to_branches = defaultdict(list)
    area_default_branch = {}
    for area, branches in taxonomy.items():
        first = True
        for branch, techs in branches.items():
            if first:
                area_default_branch[area] = branch
                first = False
            for t in techs:
                key = t.strip().lower()
                if branch not in technique_to_branches[key]:
                    technique_to_branches[key].append(branch)
    return dict(technique_to_branches), area_default_branch


def infer_branches(record, technique_to_branches, area_default_branch):
    """Return a deduped list of branch names for a record, preserving
    insertion order (first technique's branches first)."""
    seen = []
    for t in record.get("techniques", []):
        for branch in technique_to_branches.get(t.strip().lower(), []):
            if branch not in seen:
                seen.append(branch)
    if seen:
        return seen
    # Fallback: use the first branch for the record's primary area
    for a in record.get("area", []):
        if a in area_default_branch:
            return [area_default_branch[a]]
    return []


def build_branches_block(taxonomy):
    lines = ["BRANCHES = {"]
    for area, branches in taxonomy.items():
        lines.append(f"    {area!r}: {{")
        for branch, techs in branches.items():
            lines.append(f"        {branch!r}: [")
            for t in techs:
                lines.append(f"            {t!r},")
            lines.append("        ],")
        lines.append("    },")
    lines.append("}")
    return "\n".join(lines)


def update_tag_and_compare(taxonomy, tc_path="tag_and_compare.py"):
    with open(tc_path) as f:
        src = f.read()
    block = build_branches_block(taxonomy)
    existing = re.search(
        r"^BRANCHES\s*=\s*\{.*?^\}", src, re.M | re.S
    )
    if existing:
        new_src = src[:existing.start()] + block + src[existing.end():]
    else:
        # Insert after the closing ] of the TECHNIQUES list
        techs_end = re.search(r"^TECHNIQUES\s*=\s*\[.*?^\]", src, re.M | re.S)
        if techs_end:
            new_src = src[:techs_end.end()] + "\n\n" + block + src[techs_end.end():]
        else:
            new_src = src + "\n\n" + block + "\n"
    with open(tc_path, "w") as f:
        f.write(new_src)
    print(f"Updated {tc_path} with BRANCHES block.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="tagged.json")
    ap.add_argument("--branches", default="branches.json")
    ap.add_argument("--tc", default="tag_and_compare.py")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    for p in [args.branches, args.corpus]:
        if not os.path.exists(p):
            sys.exit(f"File not found: {p}")

    taxonomy = load_taxonomy(args.branches)
    technique_to_branches, area_default_branch = build_lookup(taxonomy)

    total_techs = sum(len(v) for v in technique_to_branches.values())
    print(f"Loaded taxonomy: {sum(len(b) for b in taxonomy.values())} branches, "
          f"{total_techs} technique mappings (including multi-branch duplicates).")

    records = json.load(open(args.corpus))
    print(f"Corpus: {len(records)} records.")

    updated = []
    stamped = no_branch = 0
    sample_changes = []

    for r in records:
        branches = infer_branches(r, technique_to_branches, area_default_branch)
        u = dict(r)
        old = r.get("branches", [])
        if branches:
            u["branches"] = branches
            stamped += 1
        else:
            u.pop("branches", None)
            no_branch += 1
        if old != branches and len(sample_changes) < 20:
            sample_changes.append((r["id"], old, branches))
        updated.append(u)

    print(f"Stamped: {stamped} records with branches, {no_branch} left without.")

    if args.dry_run:
        print("\n-- DRY RUN — nothing written --")
        print(f"\nSample of up to 20 changes (id: old -> new):")
        for pid, old, new in sample_changes:
            print(f"  {pid}: {old or []} -> {new}")
        return

    backup = args.corpus + ".bak"
    shutil.copy2(args.corpus, backup)
    print(f"Backed up corpus to {backup}")

    with open(args.corpus, "w") as f:
        json.dump(updated, f, indent=2, ensure_ascii=False)
    print(f"Wrote updated corpus to {args.corpus}")

    if os.path.exists(args.tc):
        update_tag_and_compare(taxonomy, args.tc)
    else:
        print(f"Note: {args.tc} not found — skipping tag_and_compare.py update")

    print("\nDone. Commit tagged.json and tag_and_compare.py, then redeploy.")
    print("\nAlso add `branches` to TAG_TOOL in tag_and_compare.py:")
    print("""
    \"branches\": {
        \"type\": \"array\",
        \"items\": {\"type\": \"string\"},
        \"description\": \"Topic branches this problem falls under, e.g. ['Circle Geometry', 'Trigonometry']. Pick from the BRANCHES dict.\"
    },""")
    print("And add 'branches' to the required list in TAG_TOOL.")


if __name__ == "__main__":
    main()
