"""
Leave-one-out adult similarity on Gemini embeddings, for children AND adults.

Why this script exists
----------------------
The cossim-to-adults columns in data/preprocessed compare each production to the adult
prototype. For adults, that prototype includes the adult's own production, so adults
can't be analysed the same way as children. Here each production (child or adult) is
compared with the mean of all OTHER adults' productions of the same animal. Children
get the same scores as before (they're never in the prototype); adults get a fair score.

Output: data/derived/loo_adult_sims.csv with columns
    pid, cat, d (drawing similarity), u (description similarity), adult (bool)
Used by 17b (adult within-person model) and 17d (category profile).

Run from the repo root:  python analysis/17a_loo_adult_similarity.py
"""
import ast
from pathlib import Path

import numpy as np
import pandas as pd

CATS = ["crab", "octopus", "seahorse", "shark", "turtle", "whale"]
OUT = Path("data/derived/loo_adult_sims.csv")


def load(path, flagcol, tidy):
    df = pd.read_csv(path)
    df["target_category"] = df.target_category.str.lower().str.strip()
    t = pd.read_csv(tidy)
    bad = set(zip(t.loc[t[flagcol] == 1, "participant_id"], t.loc[t[flagcol] == 1, "target_category"]))
    keep = [k not in bad for k in zip(df.participant_id, df.target_category)] & df.target_category.isin(CATS)
    df = df[keep].copy()
    ecol = [c for c in df.columns if "embedding" in c and "half" not in c][0]
    df = df[df[ecol].notna()]
    df["e"] = [(lambda v: v / np.linalg.norm(v))(np.array(ast.literal_eval(x), float)) for x in df[ecol]]
    print(f"{Path(path).name}: column {ecol}, {len(df)} productions")
    return df


def adult_sim(df):
    """Cosine of each production with the mean of the other adults' productions of that animal."""
    out = {}
    A = df[df.is_adult]
    for r in df.itertuples():
        pool = A[(A.target_category == r.target_category) & (A.participant_id != r.participant_id)]
        p = np.mean(np.stack(pool.e), 0)
        out[(r.participant_id, r.target_category)] = float(r.e @ p / np.linalg.norm(p))
    return out


if __name__ == "__main__":
    D = load("data/embeddings/gemini/draw_gemini_embeddings_full.csv", "draw_interference", "data/tidy/drawings.csv")
    U = load("data/embeddings/gemini/utterance_gemini_embeddings.csv", "utterance_interference", "data/tidy/utterances.csv")
    ds, us = adult_sim(D), adult_sim(U)
    x = pd.DataFrame([(k[0], k[1], ds[k], us[k]) for k in ds if k in us], columns=["pid", "cat", "d", "u"])
    x["adult"] = x.pid.str.startswith("AD")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    x.to_csv(OUT, index=False)
    print(x.groupby("adult").agg(n=("d", "size"), people=("pid", "nunique")))
    print("saved", OUT)
