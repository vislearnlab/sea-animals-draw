"""
Ask Gemini how strongly each child's own description of an animal matches their own
drawing of that animal.

This is the within-participant counterpart to 10_judge_alignment_test_pairs.py, but includes age analyses. That
script pairs drawings and utterances from *different* participants to check that the
prompt separates matches from mismatches; it measures the instrument. This one points
the validated instrument at the real data: one judgment per (participant, category)
trial, drawing and description both from the same child on the same animal.

Because there is no same/different-animal manipulation here, there are no conditions
and no good/bad status — the outcome is the alignment score itself, and how it moves
with age.

Only the alignment call is made just as a first pass and to check the prompt and schema. But would like to run the recognizability prompt as well.

The prompt and the response schema are IMPORTED from script 10 rather than copied, so
the two scripts are guaranteed to have the same prompt and schema.  

Results are appended to data/judgments/alignment_within_child.csv after every call and
keyed on (participant, category, prompt_version), so an interrupted or quota-killed run
resumes for free.

Runs from any directory (paths are anchored to the repo root):
    python 15_judge_alignment_within_child.py              # judge everything outstanding
    python 15_judge_alignment_within_child.py --dry-run    # print prompts, no API calls
    python 15_judge_alignment_within_child.py --limit 5    # smoke-test 5 trials
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "preprocessing"))

import gemini_embed as ge  # noqa: E402

ge.CACHE_DIR = REPO_ROOT / "data/embeddings/gemini/_cache"


def _load_test_pair_script():
    """Import 10_judge_alignment_test_pairs.py, whose name starts with a digit.

    Reuse its ALIGNMENT_PROMPT, ALIGNMENT_SCHEMA, PROMPT_VERSION
    """
    path = REPO_ROOT / "analysis/10_judge_alignment_test_pairs.py"
    spec = importlib.util.spec_from_file_location("judge_test_pairs", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["judge_test_pairs"] = mod
    spec.loader.exec_module(mod)
    return mod


tp = _load_test_pair_script()

PROMPT_VERSION = tp.PROMPT_VERSION
ALIGNMENT_PROMPT = tp.ALIGNMENT_PROMPT
ALIGNMENT_SCHEMA = tp.ALIGNMENT_SCHEMA
CATEGORY_ORDER = tp.CATEGORY_ORDER

# Masked text, matching the smoke test: utterance_ns names the animal outright in many
# trials ("It's a whale and it..."), which would let the judge score off the noun rather
# than the description.
UTTERANCE_COL = "utterance_full_masked_filtered"

DRAWINGS_CSV = REPO_ROOT / "data/tidy/drawings.csv"
UTTERANCES_CSV = REPO_ROOT / "data/tidy/utterances.csv"
HIGHRES_CSV = REPO_ROOT / "data/clean/drawing_highres_clean.csv"
FULL_IMAGE_ROOT = REPO_ROOT / "data/raw/mongo_output"
OUT_CSV = REPO_ROOT / "data/judgments/alignment_within_child.csv"

KEY_COLS = ["participant_id", "target_category", "prompt_version"]

# Carried through to the CSV untouched so filtering decisions stay downstream in R.
# Interference trials are judged rather than dropped: judging them costs cents, and
# excluding them here would mean a second API run to get them back.
META_COLS = ["trial_num", "trial_num_animal", "n_animals_present",
             "age_months", "age_yrs", "is_adult", "is_child",
             "draw_interference", "utterance_interference"]


def drawing_image_paths() -> dict[tuple[str, str], str]:
    """(participant_id, target_category) -> on-disk PNG path. Same lookup as script 10."""
    full = pd.read_csv(HIGHRES_CSV)
    full["target_category"] = ge.target_category_from_text1(full["text1"])
    full["participant_id"] = full["participant_id"].astype(str).str.strip()
    out = {}
    for _, r in full.iterrows():
        p = ge.resolve_image_path(r["image1"], FULL_IMAGE_ROOT)
        if p and p.exists():
            out[(r["participant_id"], r["target_category"])] = str(p)
    return out


def build_trials() -> pd.DataFrame:
    """One row per (participant, category) with both a drawing on disk and an utterance."""
    draw = pd.read_csv(DRAWINGS_CSV)
    talk = pd.read_csv(UTTERANCES_CSV)

    # Take the metadata from the drawings side and only the text from the utterances side
    keep_talk = ["participant_id", "target_category",
                 "utterance_exists", "utterance_interference", UTTERANCE_COL]
    m = draw.merge(talk[keep_talk], on=["participant_id", "target_category"],
                   how="inner", validate="one_to_one")

    have_both = m[(m["draw_exists"] == 1)
                  & (m["utterance_exists"] == 1)
                  & m[UTTERANCE_COL].notna()
                  & (m[UTTERANCE_COL].astype(str).str.strip() != "")].copy()

    images = drawing_image_paths()
    have_both["image_path"] = [
        images.get((str(p).strip(), c))
        for p, c in zip(have_both["participant_id"], have_both["target_category"])
    ]

    missing = have_both[have_both["image_path"].isna()]
    for _, r in missing.iterrows():
        print(f"  ! no image on disk for {r['participant_id']} {r['target_category']} — skipping")

    out = have_both[have_both["image_path"].notna()].copy()
    out["utterance_text"] = out[UTTERANCE_COL]
    out["prompt_version"] = PROMPT_VERSION
    out = out[["participant_id", "target_category"] + META_COLS
              + ["utterance_text", "image_path", "prompt_version"]]


    # Within a category the order is a seeded shuffle, just in case it stops early can stilll get a representative sample of ages.
    out["_c"] = pd.Categorical(out["target_category"], CATEGORY_ORDER, ordered=True)
    out = (out.sample(frac=1, random_state=0)
              .sort_values(["_c"], kind="stable"))
    out["_r"] = out.groupby("_c", observed=True).cumcount()
    return (out.sort_values(["_r", "_c"])
               .drop(columns=["_c", "_r"])
               .reset_index(drop=True))


def load_progress() -> tuple[pd.DataFrame, set]:
    if OUT_CSV.exists():
        done = pd.read_csv(OUT_CSV)
        if all(c in done.columns for c in KEY_COLS):
            return done, set(map(tuple, done[KEY_COLS].astype(str).values))
        print(f"  ! {OUT_CSV} has an older column layout — ignoring it for resume")
    return pd.DataFrame(), set()


def summarize(cur: pd.DataFrame) -> None:
    cur = cur[(cur["prompt_version"].astype(str) == PROMPT_VERSION) & cur["score"].notna()]
    if cur.empty:
        return
    print(f"\nalignment score by category ({PROMPT_VERSION}, within-child):")
    aggs = {"n": ("score", "size"), "mean_score": ("score", "mean"),
            "median": ("score", "median")}
    if cur["referents_agree"].notna().any():
        cur = cur.copy()
        cur["referents_agree"] = cur["referents_agree"].astype(float)
        aggs["agree_rate"] = ("referents_agree", "mean")
    print(cur.groupby("target_category").agg(**aggs).round(2).to_string())
    print(f"\noverall: n={len(cur)} mean={cur['score'].mean():.1f} "
          f"median={cur['score'].median():.1f} "
          f"at 100={(cur['score'] >= 100).mean():.1%} "
          f"at 0={(cur['score'] <= 0).mean():.1%}")
    print("\nscore distribution (deciles):")
    print(cur["score"].describe(percentiles=[.1, .25, .5, .75, .9]).round(1).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the rendered prompts and exit without calling the API")
    ap.add_argument("--limit", type=int, default=None,
                    help="judge at most N outstanding trials (cheap smoke test)")
    args = ap.parse_args()

    trials = build_trials()
    done_df, done_keys = load_progress()
    outstanding = trials[[
        tuple(str(v) for v in row) not in done_keys for row in trials[KEY_COLS].values
    ]]
    n_done = len(trials) - len(outstanding)
    if args.limit is not None:
        outstanding = outstanding.head(args.limit)

    print(f"{len(trials)} within-child trials at prompt {PROMPT_VERSION}, "
          f"{n_done} already judged, {len(outstanding)} to do")
    print(f"utterance column: {UTTERANCE_COL}")
    print(f"participants: {trials['participant_id'].nunique()}  "
          f"age {trials['age_yrs'].min():.1f}-{trials['age_yrs'].max():.1f} yrs")
    print(trials["target_category"].value_counts().to_string())

    if args.dry_run:
        for _, r in outstanding.iterrows():
            print("\n" + "=" * 78)
            print(f"{r['participant_id']} ({r['target_category']}, {r['age_yrs']} yrs)")
            print(f"image: {r['image_path']}")
            print("-" * 78)
            print(ALIGNMENT_PROMPT.format(utterance=r["utterance_text"]))
        print(f"\n[dry run] {len(outstanding)} trials shown, 0 API calls made")
        return

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    print("cache before run:", ge.cache_stats())
    records = done_df.to_dict("records")
    n_new = 0
    stopped_early = False
    try:
        for _, r in outstanding.iterrows():
            ali = ge.judge_drawing_text(
                r["image_path"],
                ALIGNMENT_PROMPT.format(utterance=r["utterance_text"]),
                ALIGNMENT_SCHEMA,
            )
            rec = r.to_dict()

            d_ref = tp._norm(ali.get("drawing_referent")) if ali else ""
            u_ref = tp._norm(ali.get("utterance_referent")
                             or ali.get("description_referent")) if ali else ""
            rec["score"] = (ali.get("alignment_score")
                            if ali.get("alignment_score") is not None
                            else ali.get("score")) if ali else None
            rec["reasoning"] = (ali.get("reasoning") or None) if ali else None
            rec["drawing_referent"] = d_ref or None
            rec["utterance_referent"] = u_ref or None
            rec["drawing_confidence"] = ali.get("drawing_confidence") if ali else None
            rec["utterance_confidence"] = (ali.get("description_confidence")
                                           if ali else None)

            # "unclear" on either side is not a match, and a missing referent is unknown
            # rather than False — same convention as script 10
            rec["referents_agree"] = (
                None if not ali or not d_ref else
                (d_ref == u_ref and d_ref != "unclear")
            )
            rec["drawing_referent_correct"] = (
                None if not ali or not d_ref else d_ref == r["target_category"]
            )
            rec["utterance_referent_correct"] = (
                None if not ali or not u_ref else u_ref == r["target_category"]
            )
            records.append(rec)
            n_new += 1
            pd.DataFrame(records).to_csv(OUT_CSV, index=False)
            print(f"  [{n_new}/{len(outstanding)}] {r['participant_id']:>6} "
                  f"{r['target_category']:<9} {r['age_yrs']:>5.1f}y "
                  f"align={rec['score']!s:>4} "
                  f"draw={d_ref!r}/{rec['drawing_confidence']!s} "
                  f"utt={u_ref!r}/{rec['utterance_confidence']!s}")
    except KeyboardInterrupt:
        stopped_early = True
        print("\nInterrupted by user.")
    except Exception as exc:  # noqa: BLE001 — quota errors, network errors, etc.
        stopped_early = True
        print(f"\nStopped: {type(exc).__name__}: {exc}")
    finally:
        if records:
            cur = pd.DataFrame(records)
            cur.to_csv(OUT_CSV, index=False)
            print(f"\nsaved {len(cur)} rows -> {OUT_CSV} ({n_new} new this run)")
            print("cache after run:", ge.cache_stats())
            if "score" in cur:
                summarize(cur)
        if stopped_early:
            print("\nRe-run to pick up where it left off (judged trials are cached).")


if __name__ == "__main__":
    main()
