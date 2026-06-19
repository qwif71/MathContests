#!/bin/zsh
# add_contest.sh — extract, combine, and tag a new contest PDF in one command.
#
# Usage:
#   ./add_contest.sh <pdf_path> <contest_name> <format> [round]
#
# Examples:
#   ./add_contest.sh "Problem Archive/2025 AMC 10B Solutions.pdf" "AMC 2025 10B" amc
#   ./add_contest.sh "Problem Archive/ARML-2024-Contest.pdf" "ARML 2024" arml
#   ./add_contest.sh "Problem Archive/mmaths_2024.pdf" "MMATHS 2024" mmaths

set -e

PDF="$1"
CONTEST="$2"
FORMAT="$3"
ROUND="${4:-}"

if [[ -z "$ROUND" ]]; then
    if [[ "$FORMAT" == "arml" ]]; then
        ROUND="all"
    else
        ROUND="Individual Round"
    fi
fi

if [[ -z "$PDF" || -z "$CONTEST" || -z "$FORMAT" ]]; then
    echo "Usage: ./add_contest.sh <pdf> <contest_name> <format> [round]"
    exit 1
fi

SLUG=$(echo "$CONTEST" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
RAW="${SLUG}_raw.json"
CORPUS="corpus_raw.json"
TAGGED="tagged.json"

echo "==> Extracting: $CONTEST ($FORMAT)"
python3 backend/extract.py "$PDF" "$ROUND" --contest "$CONTEST" --format "$FORMAT" -o "$RAW"

echo ""
echo "==> Combining into corpus"
RAW_FILES=(*_raw.json)
python3 backend/tag_and_compare.py combine "${RAW_FILES[@]}" -o "$CORPUS"

echo ""
echo "==> Tagging new problems (skipping already-tagged)"
python3 backend/tag_and_compare.py tag "$CORPUS" -o "$TAGGED"

echo ""
echo "Done. Corpus: $(python3 -c "import json; d=json.load(open('$TAGGED')); print(len(d), 'problems')")"
