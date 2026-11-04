"""
Name the animal in each child's drawing and in their description, in SEPARATE calls.

Why this script exists
----------------------
15_judge_alignment_within_child.py gets `drawing_referent` and `utterance_referent`
from a single call that sees the drawing and the description together. The prompt
says "looking only at the image", but the description is in the same context, so a
scribble paired with "it has eight legs" is likely to be read as an octopus. That
couples the two labels by construction. `referents_agree`, `drawing_referent_correct`
and any analysis of drawing/description coupling inherit the bias, and it is
largest for the youngest children, whose drawings are the most ambiguous.

This script asks for the two referents in two isolated calls (image only; text only).
Neither call names a target category. It also runs the target-given recognizability
prompts from script 10, which 15 does not yet run ("would like to run the
recognizability prompt as well").

Output: data/judgments/referents_isolated_within_child.csv, keyed on
(participant_id, target_category, isolated_version), with resume-on-rerun like 15.
Join it to alignment_within_child.csv on (participant_id, target_category) to compare
joint vs. isolated labels.

    python 15b_judge_referents_isolated.py --dry-run
    python 15b_judge_referents_isolated.py --limit 5
    python 15b_judge_referents_isolated.py
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load(name: str, filename: str):
    path = REPO_ROOT / "analysis" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# 15 imports 10 and sets up gemini_embed's cache dir; reuse both, so trials, image
# lookup and caching are identical to the joint-judgment run.
wc = _load("judge_within_child", "15_judge_alignment_within_child.py")
tp = wc.tp
ge = wc.ge

ISOLATED_VERSION = "iso1"
OUT_CSV = REPO_ROOT / "data/judgments/referents_isolated_within_child.csv"
KEY_COLS = ["participant_id", "target_category", "isolated_version"]

DRAWING_ONLY_PROMPT = """The image is a drawing of an animal made by a participant in a psychology study.

Which animal is drawn? Be generous about drawing skill. If you genuinely cannot tell, use "unclear".

Confidence (0-100) means how certain you are of your label. A clear drawing is high; a vague scribble is low.

Return JSON only, with integers as bare numbers (not strings):
{"drawing_referent": "<single lowercase word or unclear>",
  "drawing_confidence": <integer 0-100>}"""

DESCRIPTION_ONLY_PROMPT = """Below is a spoken description of an animal from a participant in a psychology study.

Description: "{utterance}"

Which animal does it describe? Be generous about vocabulary and grammatical skill. If the text fits no specific animal, use "unclear".

Confidence (0-100) means how certain you are of your label. An explicit, specific description is high; a generic remark is low.

Return JSON only, with integers as bare numbers (not strings):
{{"description_referent": "<single lowercase word or unclear>",
  "description_confidence": <integer 0-100>}}"""

DRAWING_ONLY_SCHEMA = {
    "type": "object",
    "properties": {"drawing_referent": {"type": "string"},
                   "drawing_confidence": {"type": "number"}},
    "required": ["drawing_referent", "drawing_confidence"],
    "propertyOrdering": ["drawing_referent", "drawing_confidence"],
}

DESCRIPTION_ONLY_SCHEMA = {
    "type": "object",
    "properties": {"description_referent": {"type": "string"},
                   "description_confidence": {"type": "number"}},
    "required": ["description_referent", "description_confidence"],
    "propertyOrdering": ["description_referent", "description_confidence"],
}


def render(r) -> dict[str, str]:
    return {
        "drawing_only": DRAWING_ONLY_PROMPT,
        "description_only": DESCRIPTION_ONLY_PROMPT.format(utterance=r["utterance_text"]),
        "drawing_target": tp.DRAWING_TARGET_PROMPT.format(drawing_category=r["target_category"]),
        "description_target": tp.DESCRIPTION_TARGET_PROMPT.format(
            utterance=r["utterance_text"], utterance_category=r["target_category"]),
    }


def judge(r) -> dict:
    p = render(r)
    d = ge.judge_drawing_text(r["image_path"], p["drawing_only"], DRAWING_ONLY_SCHEMA)
    u = ge.judge_text(p["description_only"], DESCRIPTION_ONLY_SCHEMA)
    dt = ge.judge_drawing_text(r["image_path"], p["drawing_target"], tp.DRAWING_TARGET_SCHEMA)
    ut = ge.judge_text(p["description_target"], tp.DESCRIPTION_TARGET_SCHEMA)

    d_ref = tp._norm(d.get("drawing_referent")) if d else ""
    u_ref = tp._norm(u.get("description_referent")) if u else ""
    cat = r["target_category"]
    return {
        "drawing_referent_iso": d_ref or None,
        "drawing_confidence_iso": d.get("drawing_confidence") if d else None,
        "utterance_referent_iso": u_ref or None,
        "utterance_confidence_iso": u.get("description_confidence") if u else None,
        # same conventions as 15: "unclear" never agrees; a missing label is unknown
        "referents_agree_iso": (None if not d_ref or not u_ref
                                else (d_ref == u_ref and d_ref != "unclear")),
        "drawing_referent_correct_iso": None if not d_ref else d_ref == cat,
        "utterance_referent_correct_iso": None if not u_ref else u_ref == cat,
        "drawing_recognizability": dt.get("drawing_recognizability") if dt else None,
        "description_recognizability": (ut.get("description_recognizability")
                                        if ut else None),
    }


def load_progress() -> tuple[pd.DataFrame, set]:
    if OUT_CSV.exists():
        done = pd.read_csv(OUT_CSV)
        return done, set(map(tuple, done[KEY_COLS].astype(str).values))
    return pd.DataFrame(), set()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    trials = wc.build_trials().drop(columns=["prompt_version"])
    trials["isolated_version"] = ISOLATED_VERSION
    done_df, done_keys = load_progress()
    todo = trials[[tuple(str(v) for v in row) not in done_keys
                   for row in trials[KEY_COLS].values]]
    if args.limit is not None:
        todo = todo.head(args.limit)
    print(f"{len(trials)} trials, {len(trials) - len(todo)} done, {len(todo)} to do "
          f"(4 calls each)")

    if args.dry_run:
        for _, r in todo.iterrows():
            print("\n" + "=" * 78 + f"\n{r['participant_id']} {r['target_category']}")
            for name, text in render(r).items():
                print(f"--- {name} ---\n{text}")
        print(f"\n[dry run] {len(todo)} trials shown, 0 API calls made")
        return

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    records = done_df.to_dict("records")
    try:
        for i, (_, r) in enumerate(todo.iterrows(), 1):
            rec = {**r.to_dict(), **judge(r)}
            records.append(rec)
            pd.DataFrame(records).to_csv(OUT_CSV, index=False)
            print(f"  [{i}/{len(todo)}] {r['participant_id']:>6} {r['target_category']:<9} "
                  f"draw={rec['drawing_referent_iso']!r} utt={rec['utterance_referent_iso']!r}")
    except KeyboardInterrupt:
        print("\nInterrupted; re-run to resume.")
    except Exception as exc:  # noqa: BLE001 — quota, network
        print(f"\nStopped: {type(exc).__name__}: {exc}\nRe-run to resume.")
    finally:
        if records:
            pd.DataFrame(records).to_csv(OUT_CSV, index=False)
            print(f"saved {len(records)} rows -> {OUT_CSV}")


if __name__ == "__main__":
    main()
