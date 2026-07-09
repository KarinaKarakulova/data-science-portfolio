#!/usr/bin/env bash
# ============================================================================
# publish_to_github.sh — publish this portfolio to your GitHub account with a
# clean, per-project commit history attributed to YOU.
#
# One-time prerequisites (2 minutes):
#   1. On github.com: create a new EMPTY repository named
#      data-science-portfolio (no README, no .gitignore, no license).
#   2. Make sure git knows who you are (uses your GitHub-registered email so
#      commits count toward your contribution graph):
#         git config --global user.name  "Karina Karakulova"
#         git config --global user.email "your-github-email@example.com"
#
# Then, from inside the unzipped portfolio folder:
#         bash publish_to_github.sh
#
# Authentication: when git asks for a password, use a GitHub Personal Access
# Token (Settings → Developer settings → Fine-grained tokens; scope it to just
# this one repo, set a short expiry). Never share that token with anyone —
# including AI assistants.
# ============================================================================
set -euo pipefail

REMOTE="https://github.com/KarinaKarakulova/data-science-portfolio.git"

if [ -d .git ]; then
  echo "A .git directory already exists here — refusing to re-initialize."; exit 1
fi

git init -b main
git add .gitignore README.md shared publish_to_github.sh
git commit -m "Portfolio scaffold: overview, shared requirements"

for p in \
  tier-1-foundational/01-sql-data-cleaning \
  tier-1-foundational/02-eda-financial \
  tier-1-foundational/03-python-pipeline \
  tier-2-intermediate/04-ml-classification \
  tier-2-intermediate/05-timeseries-forecasting \
  tier-2-intermediate/06-ab-testing \
  tier-2-intermediate/07-unsupervised-learning \
  tier-3-advanced/08-cloud-ml-pipeline \
  tier-3-advanced/09-nlp \
  tier-3-advanced/10-domain-capstone
do
  git add "$p"
  git commit -m "Project ${p##*/}: complete, runnable, with reports and figures"
done

git remote add origin "$REMOTE"
git push -u origin main
echo
echo "Done — https://github.com/KarinaKarakulova/data-science-portfolio"
