"""
Build data/test_pairs/{drawings,utterances}.csv — small hand-picked good/bad
contrast sets, one pair per sea animal per modality, for prompt/metric sanity checks.


Run from the preprocessing/ directory, after 05b_add_trial_num.py:
    python 05c_create_test_pairs.py
"""

from pathlib import Path

import pandas as pd

TIDY_DIR = Path("../data/tidy")
OUT_DIR = Path("../data/test_pairs")

# Keeps rows in a predictable order within each status block.
CATEGORY_ORDER = ["crab", "octopus", "whale", "shark", "seahorse", "turtle"]
STATUS_ORDER = ["good", "bad"]

# (participant_id, target_category) -> status, per modality.
TEST_PAIRS = {
    "drawings": {
        "good": [
            ("BD069", "crab"),
            ("BD058", "octopus"),
            ("BD091", "whale"),
            ("BD096", "shark"),
            ("BD120", "seahorse"),
            ("BD026", "turtle"),
        ],
        "bad": [
            ("BD061", "crab"),
            ("BD062", "octopus"),
            ("BD054", "whale"),
            ("BD113", "shark"),
            ("BD109", "seahorse"),
            ("BD103", "turtle"),
        ],
    },
    "utterances": {
        "good": [
            ("BD060", "crab"),
            ("BD035", "octopus"),
            ("BD060", "whale"),
            ("BD037", "shark"),
            ("BD047", "seahorse"),
            ("BD046", "turtle"),
        ],
        "bad": [
            ("BD048", "crab"),
            ("BD056", "octopus"),
            ("BD124", "whale"),
            ("BD077", "shark"),
            ("BD116", "seahorse"),
            ("BD123", "turtle"),
        ],
    },
}


def build(modality: str, groups: dict[str, list[tuple[str, str]]]) -> pd.DataFrame:
    tidy = pd.read_csv(TIDY_DIR / f"{modality}.csv")
    tidy["participant_id"] = tidy["participant_id"].astype(str).str.strip()

    wanted = {(pid, cat): status for status, pairs in groups.items() for pid, cat in pairs}
    keys = list(zip(tidy["participant_id"], tidy["target_category"]))

    missing = sorted(set(wanted) - set(keys))
    if missing:
        raise SystemExit(
            f"{modality}: {len(missing)} requested pair(s) not in {TIDY_DIR / f'{modality}.csv'}: "
            + ", ".join(f"{p} {c}" for p, c in missing)
        )

    sub = tidy[[k in wanted for k in keys]].copy()
    sub["status"] = [wanted[k] for k in sub[["participant_id", "target_category"]].itertuples(index=False, name=None)]

    # tidy column order, with status sitting next to the identifier columns
    cols = list(tidy.columns)
    cols.insert(cols.index("target_category") + 1, "status")
    sub = sub[cols]

    sub["_s"] = pd.Categorical(sub["status"], STATUS_ORDER, ordered=True)
    sub["_c"] = pd.Categorical(sub["target_category"], CATEGORY_ORDER, ordered=True)
    sub = sub.sort_values(["_s", "_c"]).drop(columns=["_s", "_c"]).reset_index(drop=True)

    n_expected = sum(len(p) for p in groups.values())
    if len(sub) != n_expected:
        raise SystemExit(
            f"{modality}: expected {n_expected} rows, got {len(sub)} — "
            "duplicate (participant_id, target_category) in the tidy file?"
        )
    return sub


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for modality, groups in TEST_PAIRS.items():
        out = build(modality, groups)
        path = OUT_DIR / f"{modality}.csv"
        out.to_csv(path, index=False)
        counts = out["status"].value_counts().reindex(STATUS_ORDER).to_dict()
        print(f"{path}: {len(out)} rows ({counts}), {len(out.columns)} cols")
        for _, r in out.iterrows():
            print(f"  {r['status']:4s} {r['participant_id']} {r['target_category']:9s} age={r['age_yrs']}")


if __name__ == "__main__":
    main()
