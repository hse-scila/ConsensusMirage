#!/usr/bin/env bash
# Reproduce all analyses end-to-end from a fresh clone.
set -e
cd "$(dirname "$0")"
for s in build_tidy core_metrics descriptives consensus_mirage make_fig1 stability \
         prompt_effects expert_disagreement qualitative_crossref mitigation mixed_model; do
  echo "== $s =="
  python3 "code/$s.py"
done
echo "Done. Tables and the figure are in outputs/."
