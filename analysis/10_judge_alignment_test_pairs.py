"""
Ask Gemini how strongly each test-pair utterance matches each test-pair drawing,
and whether it reads the drawing and the description as the same animal.

Pairs come in two flavours, giving three expected levels of score:

  same animal, good utterance  -> ceiling
  same animal, bad utterance   -> floor
  different animal             -> zero  (run with both good and bad utterances, to
                                         see whether a *good* description of the
                                         wrong animal still scores low)

Drawings and utterances in the test pairs come from different participants, so this
measures whether the prompt works, not within-child alignment.

Results are appended to data/test_pairs/alignment_judgments.csv after every call and
keyed on (drawing, utterance, PROMPT_VERSION), so an interrupted or quota-killed run
resumes for free, and bumping PROMPT_VERSION re-judges everything while keeping the
old rows around for comparison.

Runs from any directory (paths are anchored to the repo root):
    python 10_judge_alignment_test_pairs.py              # judge everything outstanding
    python 10_judge_alignment_test_pairs.py --dry-run    # print prompts, no API calls
    python 10_judge_alignment_test_pairs.py --limit 2    # smoke-test 2 pairs
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# gemini_embed lives in preprocessing/; everything below is anchored to the repo
# root rather than the cwd, so this runs from any directory.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "preprocessing"))

import gemini_embed as ge  # noqa: E402

# keep the shared response cache in one place no matter where this is run from,
# otherwise already-paid-for judgments look uncached and get re-requested
ge.CACHE_DIR = REPO_ROOT / "data/embeddings/gemini/_cache"

# =============================================================================
# EDIT ME — prompt, response shape, pairing, and which utterance text to judge.
# Bump PROMPT_VERSION whenever you change PROMPT_TEMPLATE or JUDGE_SCHEMA, so old
# results stay in the CSV instead of being silently mixed with new ones.
# =============================================================================

PROMPT_VERSION = "v6"

# NOTE: the literal JSON braces below are doubled ({{ }}) because this string goes
# through .format(utterance=...) — single braces there would be read as format
# fields and raise KeyError. The rendered prompt shows single braces.
ALIGNMENT_PROMPT = """The image is a drawing of an animal made by a participant in a psychology study. Below is a spoken description that may or may not be about the same animal.

Description: "{utterance}"

Work through these steps in order:

1. DRAWING: Looking only at the image, decide which animal is drawn. Be generous about drawing skill. If you genuinely cannot tell, use "unclear".
2. DESCRIPTION: Looking only at the text, decide which animal it describes. Be generous about vocabulary and grammatical skill. If the text fits no specific animal, use "unclear".
3. COMPARE: Judge how strongly the spoken description is aligned with the drawing, using an alignment score of 0-100. 

Confidence (0-100) for each referent means how certain you are of your label, given only that input. A clear drawing or an explicit animal name is high; a vague scribble or a generic remark is low.

Rules:
- Judge only whether the two are about the same animal, not the quality of the drawing or the description.
- Do not infer alignment that isn't there. 
- If either referent is "unclear", the alignment score must reflect that uncertainty.

Return JSON only, with integers as bare numbers (not strings):
{{"drawing_referent": "<single lowercase word or unclear>",
  "drawing_confidence": <integer 0-100>,
  "description_referent": "<single lowercase word or unclear>",
  "description_confidence": <integer 0-100>,
  "reasoning": "<one sentence on the level of alignment>",
  "alignment_score": <integer 0-100>}}"""


# The two recognizability prompts are fully isolated: the drawing call never
# receives the description, and the description call never receives the image.

DRAWING_TARGET_PROMPT = """The image is a drawing made by a participant in a psychology study. The participant was meant to draw a {drawing_category}.

How well does the drawing convey a {drawing_category} specifically? Be generous about drawing skill.

Rules:
- Score the drawing only against that target.
- A clear depiction of a different animal scores low, not high.
- Do not judge the quality of the drawing beyond whether the target is recognizable.

Return JSON only, with integers as bare numbers (not strings):
{{"drawing_recognizability": <integer 0-100>,
  "reasoning": "<one sentence>"}}"""


DESCRIPTION_TARGET_PROMPT = """Below is a spoken description from a participant in a psychology study. The participant was meant to describe a {utterance_category}.

Description: "{utterance}"

How well does the description convey a {utterance_category} specifically? Be generous about vocabulary and grammar.

Rules:
- Score the description only against that target.
- A clear description of a different animal scores low, not high.
- Do not judge the quality of the description beyond whether the target is recognizable.

Return JSON only, with integers as bare numbers (not strings):
{{"description_recognizability": <integer 0-100>,
  "reasoning": "<one sentence>"}}"""



# Each pair gets three calls, deliberately kept apart:
#   ALIGNMENT_PROMPT           image + description, never names either target, so
#                              the target cannot leak into the referents or the score
#   DRAWING_TARGET_PROMPT      image + its target, no description
#   DESCRIPTION_TARGET_PROMPT  text + its target, no image
# propertyOrdering makes the model emit each referent before its confidence, and
# the reasoning before the score, so the numbers follow the labels rather than
# being rationalised after the fact.
ALIGNMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "drawing_referent": {"type": "string"},
        "drawing_confidence": {"type": "number"},
        "description_referent": {"type": "string"},
        "description_confidence": {"type": "number"},
        "reasoning": {"type": "string"},
        "alignment_score": {"type": "number"},
    },
    "required": ["drawing_referent", "drawing_confidence",
                 "description_referent", "description_confidence",
                 "reasoning", "alignment_score"],
    "propertyOrdering": ["drawing_referent", "drawing_confidence",
                         "description_referent", "description_confidence",
                         "reasoning", "alignment_score"],
}

DRAWING_TARGET_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "drawing_recognizability": {"type": "number"},
    },
    "required": ["reasoning", "drawing_recognizability"],
    "propertyOrdering": ["reasoning", "drawing_recognizability"],
}

DESCRIPTION_TARGET_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "description_recognizability": {"type": "number"},
    },
    "required": ["reasoning", "description_recognizability"],
    "propertyOrdering": ["reasoning", "description_recognizability"],
}

# Using full masked filtered utterances here. Note this is a choice as currently we have embedded only masked utterances.

UTTERANCE_COL = "utterance_full_masked_filtered"

# How to build different-animal pairs for testing without going oer API budget:
#   "rotate" — each drawing meets the good and bad utterance of ONE other animal,
#              paired as crab/shark, octopus/seahorse, whale/turtle.
#              24 same-animal + 24 different-animal = 48 judgments.
#   "full"   — every drawing x every utterance across all animals = 144 judgments.
#              Balanced and thorough, but 144 calls per prompt version.
#   "off"    — same-animal pairs only = 24 judgments.
CROSS_CATEGORY_MODE = "rotate"

# =============================================================================

# Circular order; "rotate" pairs each animal with the one 3 steps along, so the
# mismatches come out as crab/shark, octopus/seahorse, whale/turtle. 
CATEGORY_ORDER = ge.CATEGORY_ORDER
STATUS_ORDER = ["good", "bad"]

# Judging order cycles through these, so partial runs cover every condition.
CONDITION_ORDER = [
    "same_animal_good_utterance",       # ceiling
    "same_animal_bad_utterance",        # floor
    "different_animal_good_utterance",  # zero, but the description is clear
    "different_animal_bad_utterance",   # zero, and the description is not
]

TEST_PAIRS_DIR = REPO_ROOT / "data/test_pairs"
HIGHRES_CSV = REPO_ROOT / "data/clean/drawing_highres_clean.csv"
FULL_IMAGE_ROOT = REPO_ROOT / "data/raw/mongo_output"
OUT_CSV = TEST_PAIRS_DIR / "alignment_judgments.csv"

# utterance_participant alone is not unique — BD060 supplies both a crab and a
# whale utterance — so the utterance category is part of the key.
KEY_COLS = ["drawing_participant", "drawing_category",
            "utterance_participant", "utterance_category", "prompt_version"]


def drawing_image_paths() -> dict[tuple[str, str], str]:
    """(participant_id, target_category) -> on-disk PNG path."""
    full = pd.read_csv(HIGHRES_CSV)
    full["target_category"] = ge.target_category_from_text1(full["text1"])
    full["participant_id"] = full["participant_id"].astype(str).str.strip()
    out = {}
    for _, r in full.iterrows():
        p = ge.resolve_image_path(r["image1"], FULL_IMAGE_ROOT)
        if p and p.exists():
            out[(r["participant_id"], r["target_category"])] = str(p)
    return out


def wanted_utterance_categories(drawing_cat: str, all_cats: list[str]) -> list[str]:
    """Which utterance animals this drawing gets paired with."""
    if CROSS_CATEGORY_MODE == "off":
        return [drawing_cat]
    if CROSS_CATEGORY_MODE == "full":
        return list(all_cats)
    if CROSS_CATEGORY_MODE == "rotate":
        i = CATEGORY_ORDER.index(drawing_cat)
        other = CATEGORY_ORDER[(i + len(CATEGORY_ORDER) // 2) % len(CATEGORY_ORDER)]
        return [c for c in (drawing_cat, other) if c in all_cats]
    raise ValueError(f"CROSS_CATEGORY_MODE must be off/rotate/full, got {CROSS_CATEGORY_MODE!r}")


def build_pairs() -> pd.DataFrame:
    draw = pd.read_csv(TEST_PAIRS_DIR / "drawings.csv")
    talk = pd.read_csv(TEST_PAIRS_DIR / "utterances.csv")
    images = drawing_image_paths()
    all_cats = sorted(set(talk["target_category"]))

    rows = []
    for _, d in draw.iterrows():
        d_cat = d["target_category"]
        img = images.get((str(d["participant_id"]).strip(), d_cat))
        if img is None:
            print(f"  ! no image on disk for {d['participant_id']} {d_cat} — skipping")
            continue
        for t_cat in wanted_utterance_categories(d_cat, all_cats):
            for _, t in talk[talk["target_category"] == t_cat].iterrows():
                same = d_cat == t_cat
                rows.append({
                    "drawing_participant": d["participant_id"],
                    "drawing_category": d_cat,
                    "drawing_status": d["status"],
                    "drawing_age_yrs": d["age_yrs"],
                    "utterance_participant": t["participant_id"],
                    "utterance_category": t_cat,
                    "utterance_status": t["status"],
                    "utterance_age_yrs": t["age_yrs"],
                    "category_match": same,
                    "condition": ("same_animal" if same else "different_animal")
                                 + f"_{t['status']}_utterance",
                    "pair_type": f"{'same' if same else 'diff'}_animal_"
                                 f"{d['status']}_drawing_x_{t['status']}_utterance",
                    "utterance_text": t[UTTERANCE_COL],
                    "image_path": img,
                    "prompt_version": PROMPT_VERSION,
                })
    out = pd.DataFrame(rows)
    # Round-robin the four conditions rather than grouping them, so a quota-limited
    # or --limit run still lands a balanced mix: every consecutive block of 4 rows
    # holds one pair from each condition.
    out["_c"] = pd.Categorical(out["condition"], CONDITION_ORDER, ordered=True)
    out["_d"] = pd.Categorical(out["drawing_category"], CATEGORY_ORDER, ordered=True)
    out = out.sort_values(["_c", "_d", "drawing_status"])
    out["_r"] = out.groupby("_c", observed=True).cumcount()
    return (out.sort_values(["_r", "_c"])
               .drop(columns=["_c", "_d", "_r"])
               .reset_index(drop=True))


def load_progress() -> tuple[pd.DataFrame, set]:
    if OUT_CSV.exists():
        done = pd.read_csv(OUT_CSV)
        if all(c in done.columns for c in KEY_COLS):
            return done, set(map(tuple, done[KEY_COLS].astype(str).values))
        print(f"  ! {OUT_CSV} has an older column layout — ignoring it for resume")
    return pd.DataFrame(), set()


def _norm(x) -> str:
    return str(x or "").strip().lower()


def render_alignment(r) -> str:
    """Call 1: the utterance only. Never mentions either target category."""
    return ALIGNMENT_PROMPT.format(utterance=r["utterance_text"])


def render_drawing_target(r) -> str:
    """Call 2: the image and its target only — the description is not included."""
    return DRAWING_TARGET_PROMPT.format(drawing_category=r["drawing_category"])


def render_description_target(r) -> str:
    """Call 3: the text and its target only — no image is sent."""
    return DESCRIPTION_TARGET_PROMPT.format(
        utterance=r["utterance_text"],
        utterance_category=r["utterance_category"],
    )


def summarize(cur: pd.DataFrame) -> None:
    cur = cur[cur["prompt_version"].astype(str) == PROMPT_VERSION]
    cur = cur[cur["score"].notna()]
    if cur.empty:
        return
    print(f"\nscore and referent agreement by condition ({PROMPT_VERSION}):")
    aggs = {"n": ("score", "size"), "mean_score": ("score", "mean")}
    # a prompt version that does not ask for a given referent leaves its column
    # all-empty; averaging that raises on object dtype, so skip it entirely
    for name, col in [("agree_rate", "referents_agree"),
                      ("drawing_ref_correct", "drawing_referent_correct"),
                      ("utterance_ref_correct", "utterance_referent_correct")]:
        if col in cur and cur[col].notna().any():
            aggs[name] = (col, "mean")
            cur[col] = cur[col].astype(float)
    print(cur.groupby("condition").agg(**aggs).round(2).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the rendered prompts and exit without calling the API")
    ap.add_argument("--limit", type=int, default=None,
                    help="judge at most N outstanding pairs (cheap prompt smoke test)")
    args = ap.parse_args()

    pairs = build_pairs()
    done_df, done_keys = load_progress()
    outstanding = pairs[[
        tuple(str(v) for v in row) not in done_keys for row in pairs[KEY_COLS].values
    ]]
    n_done = len(pairs) - len(outstanding)
    if args.limit is not None:
        outstanding = outstanding.head(args.limit)

    print(f"{len(pairs)} pairs at prompt {PROMPT_VERSION} "
          f"(cross-category mode: {CROSS_CATEGORY_MODE}), "
          f"{n_done} already judged, {len(outstanding)} to do")
    print(f"utterance column: {UTTERANCE_COL}")
    print(pairs["condition"].value_counts().to_string())

    if args.dry_run:
        for _, r in outstanding.iterrows():
            print("\n" + "=" * 78)
            print(f"{r['pair_type']}  |  drawing {r['drawing_participant']} ({r['drawing_category']}) "
                  f"+ utterance {r['utterance_participant']} ({r['utterance_category']})")
            print(f"image: {r['image_path']}")
            print("-" * 78)
            print("[CALL 1 — alignment: image + description, no target]")
            print(render_alignment(r))
            print("\n" + "-" * 78)
            print("[CALL 2 — drawing recognizability: image + target, NO description]")
            print(render_drawing_target(r))
            print("\n" + "-" * 78)
            print("[CALL 3 — description recognizability: text + target, NO image]")
            print(render_description_target(r))
        print(f"\n[dry run] {len(outstanding)} pairs x 3 prompts shown, "
              f"0 API calls made")
        return

    print("cache before run:", ge.cache_stats())
    records = done_df.to_dict("records")
    n_new = 0
    stopped_early = False
    try:
        for _, r in outstanding.iterrows():
            # Two independent calls. The alignment call never sees the target
            # categories; only the recognizability call does. Both are cached on
            # (image, prompt, schema), so a resumed run re-requests neither.
            ali = ge.judge_drawing_text(r["image_path"], render_alignment(r),
                                        ALIGNMENT_SCHEMA)
            # image + target, no description
            d_tgt = ge.judge_drawing_text(r["image_path"], render_drawing_target(r),
                                          DRAWING_TARGET_SCHEMA)
            # text + target, no image
            u_tgt = ge.judge_text(render_description_target(r),
                                  DESCRIPTION_TARGET_SCHEMA)
            rec = r.to_dict()

            # v2 named the utterance referent "utterance_referent"; v3-v5 call it
            # "description_referent". The CSV keeps one column across versions.
            d_ref = _norm(ali.get("drawing_referent")) if ali else ""
            u_ref = _norm(ali.get("utterance_referent")
                          or ali.get("description_referent")) if ali else ""
            # v5 renamed "score" to "alignment_score"; keep writing "score" so the
            # downstream scripts read one column for every prompt version
            rec["score"] = (ali.get("alignment_score")
                            if ali.get("alignment_score") is not None
                            else ali.get("score")) if ali else None
            rec["reasoning"] = (ali.get("reasoning") or None) if ali else None
            rec["drawing_referent"] = d_ref or None
            rec["utterance_referent"] = u_ref or None
            rec["drawing_confidence"] = ali.get("drawing_confidence") if ali else None
            rec["utterance_confidence"] = (ali.get("description_confidence")
                                           if ali else None)
            rec["drawing_recognizability"] = (d_tgt.get("drawing_recognizability")
                                              if d_tgt else None)
            rec["utterance_recognizability"] = (u_tgt.get("description_recognizability")
                                                if u_tgt else None)
            rec["drawing_recog_reasoning"] = (d_tgt.get("reasoning") or None) if d_tgt else None
            rec["utterance_recog_reasoning"] = (u_tgt.get("reasoning") or None) if u_tgt else None

            # agreement needs two actual animals — "unclear" on either side is not a
            # match, and an absent drawing referent is unknown rather than False
            rec["referents_agree"] = (
                None if not ali or not d_ref else
                (d_ref == u_ref and d_ref != "unclear")
            )
            rec["drawing_referent_correct"] = (
                None if not ali or not d_ref else d_ref == r["drawing_category"]
            )
            rec["utterance_referent_correct"] = (
                None if not ali or not u_ref else u_ref == r["utterance_category"]
            )
            records.append(rec)
            n_new += 1
            pd.DataFrame(records).to_csv(OUT_CSV, index=False)
            # score is None when the image is missing or the JSON failed to parse,
            # which "!s:>5" survives and ":>5" would not
            print(f"  [{n_new}/{len(outstanding)}] {r['pair_type']:48s} "
                  f"align={rec['score']!s:>4} "
                  f"draw={d_ref!r}/{rec['drawing_confidence']!s} "
                  f"utt={u_ref!r}/{rec['utterance_confidence']!s} "
                  f"recog={rec['drawing_recognizability']!s}/"
                  f"{rec['utterance_recognizability']!s}")
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
            print("\nRe-run to pick up where it left off (judged pairs are cached).")


if __name__ == "__main__":
    main()
