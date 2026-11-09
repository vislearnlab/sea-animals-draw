"""
Build the concept- and drawing-level tables for the THINGS moderator analyses.

Outputs (explore/things/derived/, committed; small):
  dot_drawings.csv          one row per DoT drawing: subject_id, sketch_id, concept, familiar,
                            recog (share of recognizers who named the concept), n_rec
  dot_concepts.csv          one row per concept: n drawings, homog (template-likeness = mean
                            pairwise cosine of mean-centred CLIP embeddings), recog, fam, homog_rank_pct
  thingsplus_concept.csv    THINGSplus norms per concept (nameability, properties, size, ...)
  gpt3_concept_features.csv GPT-3 feature counts per concept (robust = produced in >=5/30 runs;
                            part = "has ..."; distinctive = held by <=10 of 1,854 concepts)

Run from the repo root (after 01 and 02):  python explore/things/03_build_tables.py
"""
import ast
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
RAW, OUT = Path("explore/things/raw"), Path("explore/things/derived")
OUT.mkdir(parents=True, exist_ok=True)
SIX = ["seahorse", "octopus", "turtle", "whale", "shark", "crab"]

# ---- Drawings of THINGS: recognizability and template-likeness
p = pd.read_csv(RAW / "dot/prod.csv", usecols=["subject_id", "sketch_id", "concept", "familiar", "rt"])
print("prod", p.shape, "drawers", p.subject_id.nunique(), "familiar share", p.familiar.mean().round(3))
r = pd.read_csv(RAW / "dot/recog.csv")
r["labels"] = r.response_list.map(lambda s: [x.strip().lower() for x in ast.literal_eval(s)] if isinstance(s, str) else [])
r["hit"] = [c.lower() in L for c, L in zip(r.concept, r.labels)]
rec = r.groupby("sketch_id").agg(recog=("hit", "mean"), n_rec=("hit", "size")).reset_index()
z = np.load(RAW / "dot_clip.npz")
E = z["emb"] / np.linalg.norm(z["emb"], axis=1, keepdims=True)
emb = pd.DataFrame({"sketch_id": z["sketch_id"], "i": range(len(E))})
d = p.merge(rec, on="sketch_id").merge(emb, on="sketch_id")
print("joined drawings", len(d))
Ec = E - E[d.i].mean(0)  # centre to reduce CLIP anisotropy
Ec /= np.linalg.norm(Ec, axis=1, keepdims=True)
rows = []
for c, g in d.groupby("concept"):
    if len(g) < 5:
        continue
    X = Ec[g.i]
    S = X @ X.T
    rows.append((c, len(g), S[np.triu_indices(len(g), 1)].mean(), g.recog.mean(), g.familiar.mean()))
C = pd.DataFrame(rows, columns=["concept", "n", "homog", "recog", "fam"])
C["homog_rank_pct"] = C.homog.rank(pct=True)
print("concepts", len(C), " r(homog, recog)=%.2f" % C[["homog", "recog"]].corr().iloc[0, 1])
print(C.set_index("concept").loc[SIX].round(3).to_string())
C.to_csv(OUT / "dot_concepts.csv", index=False)
d[["subject_id", "sketch_id", "concept", "familiar", "recog", "n_rec"]].to_csv(OUT / "dot_drawings.csv", index=False)

# ---- THINGSplus norms
pr = pd.read_csv(RAW / "thingsplus/props.tsv", sep="\t")
m = pd.read_csv(RAW / "thingsplus/meta.tsv", sep="\t")
keep = pr[["uniqueID", "Word", "image-label_nameability_mean", "image-label_consistency_mean", "property_manmade_mean",
           "property_lives_mean", "property_moves_mean", "property_natural_mean", "property_heavy_mean",
           "property_grasp_mean", "property_pleasant_mean", "property_arousal_mean", "size_mean"]].merge(
    m[["uniqueID", "Word", "Concreteness (M)", "SUBTLEX freq", "Percent_known"]], on=["uniqueID", "Word"], how="left")
keep.columns = ["uniqueID", "Word", "nameability", "name_consistency", "manmade", "lives", "moves", "natural", "heavy",
                "grasp", "pleasant", "arousal", "size", "concreteness", "subtlex", "pct_known"]
keep.to_csv(OUT / "thingsplus_concept.csv", index=False)
print("\nTHINGSplus matched by uniqueID:", C.concept.str.replace(" ", "_").isin(keep.uniqueID).sum(), "/", len(C))
print(keep.set_index("uniqueID").loc[SIX, ["nameability", "size", "heavy", "moves"]].round(3).to_string())
typ = pd.read_csv(RAW / "thingsplus/typ.tsv", sep="\t")
sea = typ[(typ.category == "sea animal") & typ.uniqueID.isin(SIX)]
print("\nTHINGSplus typicality within 'sea animal':\n", sea.set_index("uniqueID").typicality_score.round(2).sort_values().to_string())

# ---- GPT-3 feature norms
f = pd.read_csv(RAW / "gpt3feat/data/gpt_3_feature_norm/mcrae_priming/gpt3-davinci/all_things_concepts/30/decoded_answers.csv")
g = f.groupby(["concept_id", "decoded_feature"]).run_nr.nunique().reset_index(name="runs")
g = g[g.runs >= 5]
g["is_part"] = g.decoded_feature.str.match(r"^has (a |an |two |four |six |eight |many |\w+ )?") & \
    ~g.decoded_feature.str.contains(r"has (?:been|to|no)\b")
g["n_concepts"] = g.decoded_feature.map(g.groupby("decoded_feature").concept_id.nunique())
g["distinct"] = g.n_concepts <= 10
G = g.groupby("concept_id").agg(
    n_feat=("decoded_feature", "size"), n_part=("is_part", "sum"),
    n_distinct_part=("is_part", lambda s: (s & g.loc[s.index, "distinct"]).sum()),
    n_distinct_feat=("distinct", "sum")).reset_index()
G["prop_part"] = G.n_part / G.n_feat
G.to_csv(OUT / "gpt3_concept_features.csv", index=False)
print("\n", G.set_index("concept_id").loc[SIX].to_string())
for a in SIX:
    s = g[(g.concept_id == a) & g.is_part].sort_values("runs", ascending=False)
    print(f"{a}: " + "; ".join(f"{r.decoded_feature} ({r.runs}/30, {r.n_concepts} concepts)" for r in s.head(8).itertuples()))
