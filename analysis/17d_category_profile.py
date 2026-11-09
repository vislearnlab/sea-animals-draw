"""
Which animals carry the within-child coupling, and are they the less "template-like" ones?

Per animal: drawing homogeneity (mean pairwise Gemini cosine among drawings, kids and
adults separately), entropy and count of hand-coded drawing features, the most common
feature, and the within-person drawing-description r (from 17a's leave-one-out scores).
Then Spearman correlations across the 6 animals. With n = 6 these are descriptive only.

Run from the repo root (after 17a):  python analysis/17d_category_profile.py
"""
import ast
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
CATS = ["crab", "octopus", "seahorse", "shark", "turtle", "whale"]

t = pd.read_csv("data/tidy/drawings.csv")
bad = set(zip(t.loc[t.draw_interference == 1, "participant_id"], t.loc[t.draw_interference == 1, "target_category"]))
D = pd.read_csv("data/embeddings/gemini/draw_gemini_embeddings_full.csv")
D = D[[k not in bad for k in zip(D.participant_id, D.target_category)]]
D["e"] = [(lambda v: v / np.linalg.norm(v))(np.array(ast.literal_eval(x))) for x in D.embedding]
x = pd.read_csv("data/derived/loo_adult_sims.csv")
kids = t[(t.draw_exists == 1) & (t.draw_interference == 0) & t.is_child]

rows = []
for c in CATS:
    out = {"cat": c}
    for grp, mask in [("kids", ~D.is_adult), ("adults", D.is_adult)]:
        E = np.stack(D[(D.target_category == c) & mask].e)
        S = E @ E.T
        out[f"homog_{grp}"] = S[np.triu_indices(len(E), 1)].mean()
    feats = [f.strip().lower() for s in kids[kids.target_category == c].draw_full_features.dropna()
             for f in s.split(",") if f.strip()]
    n = kids[kids.target_category == c].draw_full_features.notna().sum()
    cnt = Counter(feats)
    p = np.array(list(cnt.values())) / sum(cnt.values())
    out["feat_entropy"] = -(p * np.log2(p)).sum()
    out["n_feat_types"] = len(cnt)
    out["top_feature"] = f"{cnt.most_common(1)[0][0]} ({cnt.most_common(1)[0][1] / n:.0%})"
    for grp, a in [("kids", False), ("adults", True)]:
        s = x[x.adult == a].copy()
        for col in ["d", "u"]:
            s[col + "z"] = s.groupby("cat")[col].transform(lambda v: (v - v.mean()) / v.std())
        s["dw"] = s.dz - s.groupby("pid").dz.transform("mean")
        s["uw"] = s.uz - s.groupby("pid").uz.transform("mean")
        q = s[s.cat == c]
        out[f"within_r_{grp}"] = np.corrcoef(q.dw, q.uw)[0, 1]
    rows.append(out)

r = pd.DataFrame(rows).set_index("cat").round(3)
print(r.sort_values("within_r_kids", ascending=False).to_string())
print(f"\nkids vs adults within-r profile: rho={spearmanr(r.within_r_kids, r.within_r_adults).correlation:+.2f}")
for m in ["homog_kids", "homog_adults", "feat_entropy", "n_feat_types"]:
    print(f"{m:>14} vs within_r_kids rho={spearmanr(r[m], r.within_r_kids).correlation:+.2f} | "
          f"vs within_r_adults rho={spearmanr(r[m], r.within_r_adults).correlation:+.2f}")
