"""
Cross-modal RSA (drawing RDM vs. description RDM), corrected for missing categories.

Why this script exists
----------------------
07_rsm_toself_analysis (CLIP) and 07g_rsm_toself_analysis_gemini build each child's
6x6 matrix from the softmax `*_probability_{cat}` columns. The softmax is taken over
whichever of the child's OTHER categories exist, so a child missing one category gets
4-way probabilities (~0.25) instead of 5-way (~0.20). Missingness is correlated across
modalities within a child (a skipped trial usually loses both the drawing and the
description), so averaging these matrices plants the same missing-data pattern in
both group RDMs, and the Spearman correlation between them picks it up.

The `null` rows below show the size of the problem. They replace every probability
with 1 / (number of other categories that child has), which keeps the missingness and
removes all embedding information. That null alone reproduces most of the reported
cross-modal correlation.

Corrected estimate
------------------
- children with all 6 categories in BOTH modalities (complete cases)
- raw cosine similarities (no softmax, so no dependence on how many categories exist),
  symmetrised
- median split recomputed within complete cases
- p-values from a label permutation test (all 720 permutations of the 6 categories)
  instead of `cor.test`, which assumes independent cells that 15 RDM cells are not
- bootstrap over children for the older-minus-younger difference
- per-child RSA vs. continuous age, avoiding the median split

Runs from any directory:
    python analysis/07c_rsa_missingness_check.py
"""

import warnings
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, ttest_1samp

REPO_ROOT = Path(__file__).resolve().parent.parent
CATS = ["crab", "octopus", "seahorse", "shark", "turtle", "whale"]
LT = np.tril_indices(6, -1)
N_BOOT = 2000
RNG = np.random.default_rng(0)

# all-NaN cells (a category the child never produced) are expected and stay NaN
warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)

TIDY_DRAW = REPO_ROOT / "data/tidy/drawings.csv"
TIDY_UTT = REPO_ROOT / "data/tidy/utterances.csv"

# (label, drawing csv, utterance csv). The CLIP values live in the tidy files.
BRANCHES = [
    ("CLIP", TIDY_DRAW, TIDY_UTT),
    ("Gemini",
     REPO_ROOT / "data/preprocessed/gemini/draw_cossim_toself.csv",
     REPO_ROOT / "data/preprocessed/gemini/talk_cossim_toself.csv"),
]


def interference_keys(path, col):
    t = pd.read_csv(path)
    bad = t[t[col] == 1]
    return set(zip(bad["participant_id"], bad["target_category"].str.lower().str.strip()))


DRAW_BAD = interference_keys(TIDY_DRAW, "draw_interference")
UTT_BAD = interference_keys(TIDY_UTT, "utterance_interference")


def load(path, prefix, bad):
    df = pd.read_csv(path)
    df["target_category"] = df["target_category"].str.lower().str.strip()
    if "is_adult" in df:
        df = df[~df["is_adult"].astype(bool)]
    else:
        df = df[df["participant_id"].str.startswith("BD")]
    exists_col = f"{'draw' if prefix == 'draw' else 'utterance'}_exists"
    if exists_col in df:
        df = df[df[exists_col] == 1]
    df = df[[k not in bad for k in zip(df["participant_id"], df["target_category"])]]
    # same sample as 07/07g: every valid child trial with an age, including children
    # with a single category (their rows are all-NaN and drop out of the averages)
    return df.dropna(subset=["age_months"]).copy()


def child_matrices(df, prefix, kind):
    """kind: 'prob' (as in 07/07g), 'null' (uniform over available), 'cos' (symmetrised raw cosine)."""
    out = {}
    for pid, s in df.groupby("participant_id"):
        m = np.full((6, 6), np.nan)
        have = set(s["target_category"])
        for i, c in enumerate(CATS):
            r = s[s["target_category"] == c]
            if r.empty:
                continue
            if kind == "prob":
                m[i] = r[[f"{prefix}_probability_{k}" for k in CATS]].to_numpy()[0]
            elif kind == "cos":
                m[i] = r[[f"{prefix}_cossim_{k}" for k in CATS]].to_numpy()[0]
            elif len(have) > 1:  # null; a lone category has no siblings
                m[i] = [1 / (len(have) - 1) if (k in have and k != c) else np.nan for k in CATS]
        if kind == "cos":
            m = np.nanmean(np.stack([m, m.T]), axis=0)
        out[pid] = m
    return out


def group_rho(D, U, ids, perm=None):
    a = np.nanmean([D[p] for p in ids if p in D], axis=0)
    b = np.nanmean([U[p] for p in ids if p in U], axis=0)
    if perm is not None:
        b = b[np.ix_(perm, perm)]
    return spearmanr(a[LT], b[LT]).correlation


def median_split(ages):
    med = ages.median()
    return list(ages[ages <= med].index), list(ages[ages > med].index)


def run(label, draw_path, utt_path):
    d = load(draw_path, "draw", DRAW_BAD)
    u = load(utt_path, "utterance", UTT_BAD)
    ages = (pd.concat([d[["participant_id", "age_months"]], u[["participant_id", "age_months"]]])
              .drop_duplicates("participant_id").set_index("participant_id")["age_months"])
    print(f"\n=== {label} ===")

    # 1. as in 07/07g, and the missingness-only null on the same children
    young, old = median_split(ages)
    for kind in ("prob", "null"):
        D, U = child_matrices(d, "draw", kind), child_matrices(u, "utterance", kind)
        print(f"{'as reported (softmax)' if kind == 'prob' else 'missingness-only null':<24}"
              f" younger rho = {group_rho(D, U, young):.3f}   older rho = {group_rho(D, U, old):.3f}"
              f"   (n = {len(young)}/{len(old)})")

    # 2. corrected: complete cases, raw cosine
    D, U = child_matrices(d, "draw", "cos"), child_matrices(u, "utterance", "cos")
    complete = [p for p in ages.index
                if p in D and p in U
                and not np.isnan(D[p][LT]).any() and not np.isnan(U[p][LT]).any()]
    cy, co = median_split(ages[complete])
    perms = [np.array(p) for p in permutations(range(6))]
    for name, ids in (("younger", cy), ("older", co)):
        obs = group_rho(D, U, ids)
        null = np.array([group_rho(D, U, ids, p) for p in perms])
        print(f"corrected {name:<8} n = {len(ids):>2}  rho = {obs:.3f}  "
              f"permutation p = {np.mean(null >= obs):.4f}")

    diffs = [group_rho(D, U, RNG.choice(co, len(co))) - group_rho(D, U, RNG.choice(cy, len(cy)))
             for _ in range(N_BOOT)]
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
    print(f"older - younger: {group_rho(D, U, co) - group_rho(D, U, cy):+.3f}  "
          f"bootstrap 95% CI [{lo:+.3f}, {hi:+.3f}]")

    per_child = np.array([(ages[p], spearmanr(D[p][LT], U[p][LT]).correlation) for p in complete])
    t = ttest_1samp(per_child[:, 1], 0)
    r = pearsonr(per_child[:, 0], per_child[:, 1])
    print(f"per-child RSA: mean rho = {per_child[:, 1].mean():.3f} (t = {t.statistic:.2f}, "
          f"p = {t.pvalue:.3f}); r(age, rho) = {r.statistic:.3f}, p = {r.pvalue:.3f}")


if __name__ == "__main__":
    for branch in BRANCHES:
        run(*branch)
