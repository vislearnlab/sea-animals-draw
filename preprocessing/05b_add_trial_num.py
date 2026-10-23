"""
Add presentation-order columns to data/tidy/{drawings,utterances}.csv.

Three columns are added after target_category:

  trial_num          raw trial index from the descriptives (5-12 for the animals)
  trial_num_animal   1-6: rank of trial_num among the six sea animals only
  n_animals_present  how many of the six that child has a trial_num for

trial_num is NOT 1-6. Each session opens with a fixed warm-up block — trial 2 is
square, 3 is shape, 4 is this_square, and trial 5 is a_penguin for 143 of 152
children — after which the six sea animals appear in randomised order across
trials 6-11 (a few land at 5 or 12). trial_num_animal is therefore the column to
plot against; it is ranked from the descriptives rather than from tidy, so
excluding a bad drawing does not shift the remaining trials' positions.

For a child missing one of the six, the ranks below the gap are shifted down by one.

Idempotent — rerunning overwrites the three columns rather than duplicating them.

    python 05b_add_trial_num.py
    python 05b_add_trial_num.py --dry-run
"""

import argparse
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW = REPO_ROOT / "data/raw/mongo_output"
IMAGES_CSV = RAW / "AllDescriptives_images_final_birch_run_v1.csv"
AUDIO_CSV = RAW / "AllDescriptives_audio_final_birch_run_v1.csv"
TIDY_DIR = REPO_ROOT / "data/tidy"

CATEGORIES = ["crab", "octopus", "seahorse", "shark", "turtle", "whale"]
NEW_COLS = ["trial_num", "trial_num_animal", "n_animals_present"]


def normalise(df):
    df = df.copy()
    df["cat"] = (df.category.str.replace(r"^(a|an|this)_", "", regex=True)
                   .str.replace("_", " ").str.lower().str.strip())
    df["pid"] = df.participant_id.astype(str).str.strip().str.upper()
    return df


def order_table():
    """(pid, cat) -> trial_num / trial_num_animal / n_animals_present."""
    # Filter to the six animals before the duplicate check. On the full table the
    # warm-up categories "square" and "this_square" both normalise to "square", so
    # every participant would look like a duplicate.
    img = normalise(pd.read_csv(IMAGES_CSV))
    aud = normalise(pd.read_csv(AUDIO_CSV))
    img = img[img.cat.isin(CATEGORIES)]
    aud = aud[aud.cat.isin(CATEGORIES)]

    # A participant re-run across sessions has several trial_nums per category, and
    # tidy has no session column to disambiguate with, so they cannot be attached.
    def dedupe(name, df):
        dup = df[df.duplicated(["pid", "cat"], keep=False)]
        if len(dup):
            print(f"  {name}: dropping {len(dup)} rows from "
                  f"{dup.pid.nunique()} participant(s) with repeat sessions "
                  f"({', '.join(sorted(dup.pid.unique()))})")
        return df.drop_duplicates(["pid", "cat"], keep=False)

    img = dedupe("images", img)
    aud = dedupe("audio", aud)

    order = img[["pid", "cat", "trial_num"]].copy()
    # audio-only trials, if any, keep the same numbering
    extra = aud.merge(order[["pid", "cat"]], on=["pid", "cat"], how="left",
                      indicator=True)
    extra = extra[extra._merge == "left_only"][["pid", "cat", "trial_num"]]
    if len(extra):
        print(f"  {len(extra)} (participant, category) pairs came from audio only")
        order = pd.concat([order, extra], ignore_index=True)

    six = order.copy()
    six["trial_num_animal"] = (six.groupby("pid").trial_num
                                  .rank(method="dense").astype(int))
    six["n_animals_present"] = six.groupby("pid").trial_num.transform("size")
    return six.set_index(["pid", "cat"])[NEW_COLS]


def attach(path, lookup, dry_run):
    t = pd.read_csv(path)
    pid = t.participant_id.astype(str).str.strip().str.upper()
    cat = t.target_category.str.lower().str.strip()
    keys = list(zip(pid, cat))

    got = lookup.reindex(keys)
    t = t.drop(columns=[c for c in NEW_COLS if c in t.columns])  # keep idempotent
    at = t.columns.get_loc("target_category") + 1
    for i, col in enumerate(NEW_COLS):
        t.insert(at + i, col, got[col].values)

    n = len(t)
    matched = t.trial_num.notna().sum()
    print(f"\n{path.name}: {matched}/{n} rows matched ({matched/n:.1%})")
    if matched < n:
        miss = t[t.trial_num.isna()]
        print(f"  unmatched participants: "
              f"{', '.join(sorted(miss.participant_id.astype(str).unique())[:8])}"
              f"{' ...' if miss.participant_id.nunique() > 8 else ''}")
    complete = t[t.n_animals_present == 6]
    print(f"  rows whose child has all six animals: {len(complete)}/{n}")
    print("  trial_num_animal distribution:",
          t.trial_num_animal.value_counts().sort_index().to_dict())

    if dry_run:
        print("  [dry run] not written")
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(f"{path.stem}.bak_{stamp}{path.suffix}")
    shutil.copy2(path, backup)
    t.to_csv(path, index=False)
    print(f"  wrote {path.name} ({len(t.columns)} cols); backup at {backup.name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    lookup = order_table()
    print(f"\norder table: {len(lookup)} (participant, category) pairs across "
          f"{lookup.index.get_level_values(0).nunique()} participants")
    for name in ["drawings.csv", "utterances.csv"]:
        attach(TIDY_DIR / name, lookup, args.dry_run)


if __name__ == "__main__":
    main()
