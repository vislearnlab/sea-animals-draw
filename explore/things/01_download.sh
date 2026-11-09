#!/usr/bin/env bash
# Download the open datasets used in explore/things into explore/things/raw/ (git-ignored, ~2.2 GB).
# Run from the repo root:  bash explore/things/01_download.sh
set -euo pipefail
R=explore/things/raw
mkdir -p "$R/dot" "$R/thingsplus"

# Drawings of THINGS (OSF s9yta): production data, recognition data, PNGs
curl -sSL -o "$R/dot/README.md"            https://osf.io/download/zpfy3/
curl -sSL -o "$R/dot/recog.csv"            https://osf.io/download/xjnts/
curl -sSL -o "$R/dot/recognizability.csv"  https://osf.io/download/zfuyr/
curl -sSL -o "$R/dot/prod.csv"             https://osf.io/download/pmafv/   # ~800 MB
curl -sSL -o "$R/dot/pngs.zip"             https://osf.io/download/qz2yn/
unzip -q -o "$R/dot/pngs.zip" -d "$R/dot/pngs"

# THINGSplus norms (OSF jum2f)
curl -sSL -o "$R/thingsplus/props.tsv" https://osf.io/download/7cz69/   # property ratings, nameability, size
curl -sSL -o "$R/thingsplus/meta.tsv"  https://osf.io/download/5uvc2/   # concreteness, frequency, % known
curl -sSL -o "$R/thingsplus/typ.tsv"   https://osf.io/download/qj7ec/   # typicality within 53 categories
curl -sSL -o "$R/thingsplus/cat53.tsv" https://osf.io/download/pa9ry/   # 53 higher-level categories

# GPT-3 semantic feature norms for all THINGS concepts (Hansen & Hebart, 2022)
[ -d "$R/gpt3feat" ] || git clone -q --depth 1 https://github.com/ViCCo-Group/semantic_features_gpt_3.git "$R/gpt3feat"

echo "done"; du -sh "$R"/*
