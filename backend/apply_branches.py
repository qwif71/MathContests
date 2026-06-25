"""
apply_branches.py — After you've reviewed and edited branches.json produced
by classify_branches.py, run this to:

  1. Update tag_and_compare.py's BRANCHES constant with the approved taxonomy.
  2. Stamp every record in tagged.json with a `branch` field based on which
     branch its techniques belong to. Records with no matching technique get
     branch inferred from their area alone (falls back to the most common
     branch for that area, or None if the area itself is absent).
  3. Writes updated tagged.json in place (backs up the original first).

Run:
    python apply_branches.py [--corpus tagged.json] [--dry-run]
"""
import argparse, json, os, shutil, sys
from collections import Counter, defaultdict


def load_branches(path="branches.json"):
    with open(path) as f:
        data = json.load(f)
    # taxonomy: {area: {branch: [technique, ...]}}
    return data["taxonomy"]


def build_lookup(taxonomy):
    """Returns two dicts:
      technique_to_branch: technique name -> (area, branch)
      area_to_branches: area -> list of branch names (for fallback)
    """
    technique_to_branch = {}
    area_to_branches = defaultdict(list)
    for area, branches in taxonomy.items():
        for branch, techs in branches.items():
            area_to_branches[area].append(branch)
            for t in techs:
                # Lowercase key for case-insensitive lookup
                technique_to_branch[t.strip().lower()] = (area, branch)
    return technique_to_branch, area_to_branches


def infer_branch(record, technique_to_branch, area_to_branches):
    """Return the best branch for a record:
    1. First technique that maps to a known branch wins.
    2. Fallback: the first branch for the record's primary area.
    3. Final fallback: None.
    """
    for t in record.get("techniques", []):
        hit = technique_to_branch.get(t.strip().lower())
        if hit:
            return hit[1]  # just the branch name

    # No technique matched — fall back on area
    for a in record.get("area", []):
        branches = area_to_branches.get(a)
        if branches:
            return None  # area known but branch ambiguous; leave unset
    return None


def build_tag_and_compare_block(taxonomy):
    """Generate the BRANCHES = {...} Python literal to insert into
    tag_and_compare.py, replacing the old placeholder (or adding if absent)."""
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
    """Insert or replace a BRANCHES block in tag_and_compare.py.
    Looks for an existing BRANCHES = { ... } block and replaces it;
    if not found, inserts it right after the TECHNIQUES list."""
    with open(tc_path) as f:
        src = f.read()

    block = build_tag_and_compare_block(taxonomy)

    import re
    # Match an existing BRANCHES = { ... } block (possibly multiline)
    existing = re.search(r"^BRANCHES\s*=\s*\{[^}]*(?:\{[^}]*\}[^}]*)?\}", src, re.M)
    if existing:
        new_src = src[:existing.start()] + block + src[existing.end():]
    else:
        # Insert after the TECHNIQUES list closes (first line that is just "]")
        # after "TECHNIQUES = ["
        techs_end = re.search(r"^TECHNIQUES\s*=\s*\[.*?^]", src, re.M | re.S)
        if techs_end:
            insert_at = techs_end.end()
            new_src = src[:insert_at] + "\n\n" + block + src[insert_at:]
        else:
            new_src = src + "\n\n" + block + "\n"

    with open(tc_path, "w") as f:
        f.write(new_src)

    print(f"Updated {tc_path} with BRANCHES block.")


def stamp_corpus(records, technique_to_branch, area_to_branches, dry_run=False):
    """Add a `branch` field to every record. Returns (updated_records, stats)."""
    stats = Counter()
    out = []
    for r in records:
        branch = infer_branch(r, technique_to_branch, area_to_branches)
        updated = dict(r)
        if branch:
            updated["branch"] = branch
            stats["stamped"] += 1
        else:
            # Remove stale branch if the technique list changed
            updated.pop("branch", None)
            stats["no_branch"] += 1
        out.append(updated)
    return out, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="tagged.json",
                    help="Path to the live corpus JSON (default: tagged.json)")
    ap.add_argument("--branches", default="branches.json",
                    help="Path to the reviewed branches.json (default: branches.json)")
    ap.add_argument("--tc", default="tag_and_compare.py",
                    help="Path to tag_and_compare.py (default: tag_and_compare.py)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would change without writing anything")
    args = ap.parse_args()

    if not os.path.exists(args.branches):
        sys.exit(f"branches.json not found at {args.branches} — run classify_branches.py first")
    if not os.path.exists(args.corpus):
        sys.exit(f"Corpus not found at {args.corpus}")

    taxonomy = load_branches(args.branches)
    technique_to_branch, area_to_branches = build_lookup(taxonomy)

    total_branches = sum(len(b) for b in taxonomy.values())
    total_mapped = len(technique_to_branch)
    print(f"Loaded {total_branches} branches covering {total_mapped} techniques.")

    records = json.load(open(args.corpus))
    print(f"Corpus: {len(records)} records.")

    updated, stats = stamp_corpus(records, technique_to_branch, area_to_branches, dry_run=args.dry_run)
    print(f"Branch stamping: {stats['stamped']} stamped, {stats['no_branch']} could not be assigned.")

    if args.dry_run:
        print("\n-- DRY RUN: nothing written --")
        # Show a sample of what would change
        changes = [(r["id"], u.get("branch")) for r, u in zip(records, updated)
                   if r.get("branch") != u.get("branch")][:20]
        if changes:
            print(f"Sample of {len(changes)} changes (id -> new branch):")
            for pid, branch in changes:
                print(f"  {pid}: {branch or '(none)'}")
        return

    # Backup
    backup = args.corpus + ".bak"
    shutil.copy2(args.corpus, backup)
    print(f"Backed up original corpus to {backup}")

    # Write updated corpus
    with open(args.corpus, "w") as f:
        json.dump(updated, f, indent=2, ensure_ascii=False)
    print(f"Wrote updated corpus to {args.corpus}")

    # Update tag_and_compare.py
    if os.path.exists(args.tc):
        update_tag_and_compare(taxonomy, args.tc)
    else:
        print(f"Warning: {args.tc} not found — skipping tag_and_compare.py update")

    print("\nDone. Next steps:")
    print("  1. Commit the updated tagged.json and tag_and_compare.py")
    print("  2. Add `branch` to the TAG_TOOL schema in tag_and_compare.py")
    print("     so new imports also get a branch tag (see below)")
    print("\nAdd this to TAG_TOOL['input_schema']['properties']:")
    print('    "branch": {"type": "string",')
    print('               "description": "The topic branch this problem falls under"},')
    print("\nAnd add 'branch' to the required list in TAG_TOOL.")


if __name__ == "__main__":
    main()
