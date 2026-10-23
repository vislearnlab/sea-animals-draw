"""
Contact sheet of the same-animal test pairs: the drawing, the utterance, and the
alignment score the judge gave them, laid out so each cell can be eyeballed.

The same-animal half of the test set is a clean 6 x 4 grid — one pair per
(animal, quality cell) — so rows are animals and columns are the four crossings
of drawing quality x utterance quality. Within a row the two left columns share
the same good drawing and the two right columns share the same bad drawing,
which is why a drawing image appears twice: the pair, not the drawing, is the
unit here.

Thumbnails are produced by gemini_embed.prep_image with the same arguments
judge_drawing_text uses, so what you see is what the model was sent —
content-cropped, not the raw canvas.

Runs from any directory (paths are anchored to the repo root):
    python 12_pair_gallery.py                      # current prompt version
    python 12_pair_gallery.py --prompt-version v2  # same layout, older scores
"""

import argparse
import importlib.util
import sys
import textwrap
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "preprocessing"))

import gemini_embed as ge  # noqa: E402

ge.CACHE_DIR = REPO_ROOT / "data/embeddings/gemini/_cache"

JUDGMENTS_CSV = REPO_ROOT / "data/test_pairs/alignment_judgments.csv"

CATEGORY_ORDER = ge.CATEGORY_ORDER


def _current_prompt_version() -> str:
    """PROMPT_VERSION as script 10 currently defines it.

    Read rather than hardcoded: this used to default to "v3" and was still saying so
    three prompt versions later, which silently drew the gallery from stale rows.
    """
    path = REPO_ROOT / "analysis/10_judge_alignment_test_pairs.py"
    spec = importlib.util.spec_from_file_location("judge_test_pairs", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.PROMPT_VERSION

# (drawing_status, utterance_status) in the reading order of figure 2 in script 11
QUALITY_CELLS = [
    ("good", "good", "good drawing\ngood utterance"),
    ("good", "bad",  "good drawing\nbad utterance"),
    ("bad",  "good", "bad drawing\ngood utterance"),
    ("bad",  "bad",  "bad drawing\nbad utterance"),
]

WRAP = 52          # characters per line in the utterance block
IMG_RATIO = 2.3    # image sub-row height relative to the text sub-row
TXT_RATIO = 2.9
ROW_H = 4.15       # inches per animal row (image sub-row + text sub-row)
FS_UTT, FS_REASON = 7.2, 6.8
TEXT_TOP = 0.88    # leaves room for the score badge hanging under the image

# The utterance and the reasoning are separate text objects (different styles), so
# the reasoning has to be placed below however many lines the utterance took. One
# line as a fraction of the text sub-row's height:
_TXT_IN = TXT_RATIO / (IMG_RATIO + TXT_RATIO) * ROW_H
LINE_FRAC = (FS_UTT * 1.45 / 72.0) / _TXT_IN


def score_colours(score):
    """Viridis fill for a 0-100 score, with legible text on top of it."""
    if pd.isna(score):
        return "#DDDDDD", "black"
    rgba = plt.cm.viridis(float(score) / 100.0)
    # relative luminance decides black vs white text; viridis runs dark to light
    lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
    return rgba, ("black" if lum > 0.5 else "white")


def metric_line(label, referent, confidence, recognizability):
    """One aligned row of per-input metrics, tolerating missing values."""
    def num(v):
        return "  -" if pd.isna(v) else f"{float(v):3.0f}"
    ref = "-" if pd.isna(referent) else str(referent)
    return f"{label:<11} {ref:<9} conf {num(confidence)}  recog {num(recognizability)}"


def load_thumb(path):
    """The exact bytes judge_drawing_text sent for this drawing."""
    try:
        return Image.open(BytesIO(ge.prep_image(path, content_crop=True, max_side=1024)))
    except Exception as exc:  # noqa: BLE001 — a missing file should not kill the sheet
        print(f"  ! could not load {path}: {exc}")
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompt-version", default=None,
                    help="which prompt_version's scores to show "
                         "(default: whatever script 10 is currently set to)")
    ap.add_argument("--out", default=None,
                    help="output PNG (default figures/alignment_pair_gallery_<version>.png)")
    args = ap.parse_args()
    version = args.prompt_version or _current_prompt_version()
    out = Path(args.out) if args.out else \
        REPO_ROOT / f"figures/alignment_pair_gallery_{version}.png"

    d = pd.read_csv(JUDGMENTS_CSV)
    d = d[(d.prompt_version.astype(str) == version) & (d.category_match)]
    if d.empty:
        sys.exit(f"no same-animal rows at prompt_version={version!r} in {JUDGMENTS_CSV}")
    # v5 onwards reports a referent, a target-blind confidence and a
    # target-informed recognizability for each input; earlier versions do not.
    METRIC_COLS = ["drawing_referent", "drawing_confidence", "drawing_recognizability",
                   "utterance_referent", "utterance_confidence",
                   "utterance_recognizability"]
    has_metrics = any(c in d.columns and d[c].notna().any() for c in METRIC_COLS)
    print(f"{len(d)} same-animal pairs at prompt {version}"
          f"{' (with per-input metrics)' if has_metrics else ''}")

    cats = [c for c in CATEGORY_ORDER if c in set(d.drawing_category)]
    nrow, ncol = len(cats), len(QUALITY_CELLS)

    fig = plt.figure(figsize=(4.15 * ncol + 0.8, ROW_H * nrow))
    gs = fig.add_gridspec(
        nrows=2 * nrow, ncols=ncol + 1,
        width_ratios=[0.22] + [1] * ncol,
        height_ratios=[IMG_RATIO, TXT_RATIO] * nrow,
        hspace=0.06, wspace=0.06,
        left=0.012, right=0.995, top=0.952, bottom=0.008,
    )

    header_axes = {}
    for i, cat in enumerate(cats):
        lab = fig.add_subplot(gs[2 * i:2 * i + 2, 0])
        lab.set_axis_off()
        lab.text(0.5, 0.5, cat, rotation=90, ha="center", va="center",
                 fontsize=13, fontweight="bold")

        for j, (d_stat, u_stat, _) in enumerate(QUALITY_CELLS):
            row = d[(d.drawing_category == cat)
                    & (d.drawing_status == d_stat)
                    & (d.utterance_status == u_stat)]
            img_ax = fig.add_subplot(gs[2 * i, j + 1])
            txt_ax = fig.add_subplot(gs[2 * i + 1, j + 1])
            for ax in (img_ax, txt_ax):
                ax.set_axis_off()
            if i == 0:
                header_axes[j] = img_ax
            if row.empty:
                img_ax.text(0.5, 0.5, "no pair", ha="center", va="center",
                            fontsize=9, color="#999999")
                continue
            r = row.iloc[0]

            thumb = load_thumb(r["image_path"])
            if thumb is not None:
                img_ax.imshow(thumb)
            fill, txtcol = score_colours(r["score"])
            # score sits on the image so the number and the drawing read together
            img_ax.text(
                0.5, -0.015, f"  score {r['score']:g}  ",
                transform=img_ax.transAxes, ha="center", va="top",
                fontsize=12, fontweight="bold", color=txtcol,
                bbox=dict(facecolor=fill, edgecolor="none", pad=3.5),
            )
            img_ax.text(0.5, 1.01, f"drawing {r['drawing_participant']} ({d_stat})",
                        transform=img_ax.transAxes, ha="center", va="bottom",
                        fontsize=8, color="#555555")

            body = [f"utterance {r['utterance_participant']} ({u_stat}):"]
            body += textwrap.wrap(f"“{r['utterance_text']}”", WRAP)
            txt_ax.text(0.02, TEXT_TOP, "\n".join(body), transform=txt_ax.transAxes,
                        ha="left", va="top", fontsize=FS_UTT, linespacing=1.45)
            if has_metrics:
                # Referent and confidence come from the target-blind call;
                # recognizability from the target-informed one. Keeping them on one
                # line each makes the two inputs directly comparable down the column.
                metrics = "\n".join([
                    metric_line("drawing", r.get("drawing_referent"),
                                r.get("drawing_confidence"),
                                r.get("drawing_recognizability")),
                    metric_line("description", r.get("utterance_referent"),
                                r.get("utterance_confidence"),
                                r.get("utterance_recognizability")),
                ])
                # one blank line of gap below the utterance block
                txt_ax.text(0.02, TEXT_TOP - (len(body) + 0.6) * LINE_FRAC, metrics,
                            transform=txt_ax.transAxes, ha="left", va="top",
                            fontsize=FS_REASON + 0.4, family="monospace",
                            color="#38618C", linespacing=1.6)

    for j, (_, _, label) in enumerate(QUALITY_CELLS):
        pos = header_axes[j].get_position()
        fig.text((pos.x0 + pos.x1) / 2, 0.985, label, ha="center", va="top",
                 fontsize=12, fontweight="bold")

    fig.suptitle(
        f"Same-animal test pairs at prompt {version}: drawing, utterance, and alignment score"
        + ("  —  per-input referent, target-blind confidence, target-informed recognizability"
           if has_metrics else ""),
        x=0.012, ha="left", y=0.997, fontsize=14, va="top",
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
