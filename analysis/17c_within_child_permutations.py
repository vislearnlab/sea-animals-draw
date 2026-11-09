"""
Is the drawing-description coupling specific to the child? Two permutation tests.

1. Child-swap permutation (embedding scores)
   Pair each child's drawings with ANOTHER child's descriptions (a derangement within
   age quartile, so ages stay matched), matched on animal. Recompute the within-child
   slope (drawing deviation ~ description deviation, both centred within the drawer).
   If the effect were just "some animals are drawn and described better by everyone"
   or an age-wide pattern, swapped pairs would show it too. 5,000 permutations, for each
   of the 4 drawing-model x description-model pairings.

2. Yoked feature overlap (hand-coded features)
   Dice overlap between a child's drawing features and their OWN description features,
   vs. descriptions of the same animal by other children: (a) the 5 nearest in age
   ("yoked"), with a child-level paired test; (b) 2,000 shuffles of descriptions within
   animal x age tertile.

Run from the repo root:  python analysis/17c_within_child_permutations.py
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, ttest_rel

N_SWAP, N_SHUFFLE = 5000, 2000

d = pd.read_csv("data/tidy/drawings.csv")
u = pd.read_csv("data/tidy/utterances.csv")
d = d[(d.draw_exists == 1) & (d.draw_interference == 0) & d.is_child & d.age_months.notna()]
u = u[(u.utterance_exists == 1) & (u.utterance_interference == 0) & u.is_child]


# ---------------------------------------------------------------- 1. child swap
def swap_test(rng):
    gd = pd.read_csv("data/preprocessed/gemini/draw_cossim_adults.csv")[["participant_id", "target_category", "draw_cossim_adults"]] \
        .rename(columns={"draw_cossim_adults": "g_draw"})
    gu = pd.read_csv("data/preprocessed/gemini/talk_cossim_adults.csv")[["participant_id", "target_category", "utterance_full_cossim_adults"]] \
        .rename(columns={"utterance_full_cossim_adults": "g_utt"})
    D = d[["participant_id", "target_category", "age_months", "draw_cossim_adults"]] \
        .rename(columns={"draw_cossim_adults": "c_draw"}).merge(gd, how="left")
    U = u[["participant_id", "target_category", "utterance_full_cossim_adults"]] \
        .rename(columns={"utterance_full_cossim_adults": "c_utt"}).merge(gu, how="left")
    z = lambda s: (s - s.mean()) / s.std()
    for col in ["c_draw", "g_draw"]:
        D[col] = D.groupby("target_category")[col].transform(z)
    for col in ["c_utt", "g_utt"]:
        U[col] = U.groupby("target_category")[col].transform(z)
    both = sorted(set(D.participant_id) & set(U.participant_id))
    age = D.drop_duplicates("participant_id").set_index("participant_id").age_months[both]
    band = pd.qcut(age, 4, labels=False)

    def within_beta(pairs, dc, uc):
        m = D[["participant_id", "target_category", dc]].merge(pairs, left_on="participant_id", right_on="drawer").merge(
            U[["participant_id", "target_category", uc]].rename(columns={"participant_id": "describer"}),
            on=["describer", "target_category"]).dropna()
        m = m[m.groupby("drawer").target_category.transform("size") >= 2]
        dw = m[dc] - m.groupby("drawer")[dc].transform("mean")
        uw = m[uc] - m.groupby("drawer")[uc].transform("mean")
        return (dw * uw).sum() / (uw * uw).sum(), len(m)

    ident = pd.DataFrame({"drawer": both, "describer": both})
    print("== 1. Child-swap permutation (derangements within age quartile) ==")
    for dc, uc, lab in [("c_draw", "c_utt", "CLIP~CLIP"), ("g_draw", "g_utt", "Gemini~Gemini"),
                        ("c_draw", "g_utt", "CLIP~Gemini"), ("g_draw", "c_utt", "Gemini~CLIP")]:
        obs, n = within_beta(ident, dc, uc)
        null = []
        for _ in range(N_SWAP):
            desc = []
            for q in range(4):
                ids = list(band[band == q].index)
                perm = rng.permutation(ids)
                while any(a == c for a, c in zip(ids, perm)):  # never paired with self
                    perm = rng.permutation(ids)
                desc += list(zip(ids, perm))
            null.append(within_beta(pd.DataFrame(desc, columns=["drawer", "describer"]), dc, uc)[0])
        null = np.array(null)
        print(f"{lab:14s} own-child slope={obs:.3f} (n={n}) | swapped: mean={null.mean():+.3f}, "
              f"95% [{np.percentile(null, 2.5):+.3f},{np.percentile(null, 97.5):+.3f}] | p={np.mean(null >= obs):.4f}")


# ---------------------------------------------------------------- 2. yoked features
def feature_test(rng):
    sp = lambda x: {t.strip().lower() for t in str(x).split(",") if t.strip() and str(x) != "nan"}
    b = d[["participant_id", "target_category", "age_months", "draw_full_features"]].merge(
        u[["participant_id", "target_category", "utterance_full_visual_features", "utterance_full_semantic_features"]],
        on=["participant_id", "target_category"])
    b["D"] = b.draw_full_features.map(sp)
    b["U"] = b.apply(lambda r: sp(r.utterance_full_visual_features) | sp(r.utterance_full_semantic_features), axis=1)
    b = b[(b.D.str.len() > 0) & (b.U.str.len() > 0)].reset_index(drop=True)
    dice = lambda A, B: 2 * len(A & B) / (len(A) + len(B))
    b["own"] = [dice(r.D, r.U) for r in b.itertuples()]

    def yoked(r, k=5):
        pool = b[(b.target_category == r.target_category) & (b.participant_id != r.participant_id)].copy()
        pool["dist"] = (pool.age_months - r.age_months).abs()
        return np.mean([dice(r.D, U) for U in pool.nsmallest(k, "dist").U])

    b["yoked"] = [yoked(r) for r in b.itertuples()]
    print("\n== 2. Feature overlap: own vs. other children's descriptions ==")
    print(f"trials {len(b)}, kids {b.participant_id.nunique()} | own Dice {b.own.mean():.3f}, yoked Dice {b.yoked.mean():.3f}")
    c = b.groupby("participant_id")[["own", "yoked", "age_months"]].mean()
    diff = c.own - c.yoked
    t, p = ttest_rel(c.own, c.yoked)
    print(f"child-level own-yoked = {diff.mean():.3f}, t={t:.2f} p={p:.4f}; own>yoked in {(diff > 0).sum()}/{len(diff)} kids")
    print("diff ~ age: r=%.2f p=%.3f" % pearsonr(c.age_months, diff))
    b["band"] = pd.qcut(b.age_months, 3, labels=False)
    obs, null = b.own.mean(), []
    for _ in range(N_SHUFFLE):
        s = b.groupby(["target_category", "band"], group_keys=False).apply(
            lambda g: g.assign(Up=list(g.U.sample(frac=1, random_state=int(rng.integers(1e9))))))
        null.append(np.mean([dice(D_, U_) for D_, U_ in zip(s.D, s.Up)]))
    null = np.array(null)
    print("shuffle within animal x age tertile: own %.3f vs null %.3f [95%%: %.3f-%.3f], p=%.4f"
          % (obs, null.mean(), *np.percentile(null, [2.5, 97.5]), np.mean(null >= obs)))


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    swap_test(np.random.default_rng(7))
    feature_test(np.random.default_rng(1))
