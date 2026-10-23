"""
6 x 6 grid of children's drawings for easy to see prompt evaluation on test pairs. 

Columns are the six sea-animal categories in alphabetical order; rows are six
equal-count age bins over the pooled child drawings, youngest at the top. Each
cell holds one drawing, tinted with the viridis colour for its age bin, so the
developmental progression reads as purple -> green across the rows.

Cell selection is deterministic: the drawing whose age sits closest to the median
age of its (category, bin) cell. That is deliberately blind to how good the
drawing is — picking by recognisability would manufacture the age trend the
figure is supposed to show. Pass --pick random for a seeded random draw instead.

Children only (is_child), excluding draw_exists == 0 and draw_interference == 1,
matching the filters the other analyses use.

    python 13_drawing_grid.py
    python 13_drawing_grid.py --pick random --seed 7
    python 13_drawing_grid.py --no-crop          # keep the original canvas
"""

import argparse
import sys
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image, ImageChops  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "preprocessing"))

import gemini_embed as ge  # noqa: E402

TIDY_CSV = REPO_ROOT / "data/tidy/drawings.csv"
# tidy/drawings.csv carries age and the exclusion flags but no file path, so the
# image column is joined in from the clean table the same way script 10 does it
HIGHRES_CSV = REPO_ROOT / "data/clean/drawing_highres_clean.csv"
IMAGE_ROOT = REPO_ROOT / "data/raw/mongo_output"

CATEGORIES = ge.SEA_ANIMALS   # alphabetical, which is also this figure's column order
N_BINS = 6

# Viridis runs to #FDE725, which is close to invisible on white. Capping the top
# keeps the oldest row a legible yellow-green while preserving the purple->green
# progression. Raise to 1.0 if the figure will sit on a dark slide.
VIRIDIS_END = 0.88


def ink_mask(img):
    """Opacity map of the strokes: transparency if present, else dark pixels."""
    if img.mode in ("RGBA", "LA") and img.getchannel("A").getextrema()[0] < 255:
        return img.getchannel("A")
    return ImageChops.invert(img.convert("L"))


def tinted(path, rgb, crop=True):
    """The drawing with its strokes recoloured to rgb, on a square white field."""
    if crop:
        img = Image.open(BytesIO(ge.prep_image(path, content_crop=True, max_side=700)))
    else:
        img = Image.open(path)
    img.load()
    colour = Image.new("RGBA", img.size, tuple(int(255 * c) for c in rgb[:3]) + (0,))
    colour.putalpha(ink_mask(img))
    # Pad to square so every cell has the same aspect ratio — otherwise the axes
    # are different heights and the column titles and row labels stop lining up.
    side = max(img.size)
    out = Image.new("RGBA", (side, side), (255, 255, 255, 255))
    out.alpha_composite(colour, ((side - img.width) // 2, (side - img.height) // 2))
    return out


def image_paths():
    """(participant_id, category) -> image1 path, from the clean table."""
    full = pd.read_csv(HIGHRES_CSV)
    full["cat"] = ge.target_category_from_text1(full["text1"])
    full["pid"] = full.participant_id.astype(str).str.strip().str.upper()
    return full.drop_duplicates(["pid", "cat"]).set_index(["pid", "cat"])["image1"]


def load_cells(pick, seed):
    d = pd.read_csv(TIDY_CSV)
    d = d[(d.draw_exists == 1) & (d.draw_interference == 0) & d.age_yrs.notna()]
    d = d[d.is_child == True]  # noqa: E712 — pandas mask, not a bool test
    d["cat"] = d.target_category.str.lower()
    d = d[d.cat.isin(CATEGORIES)].copy()

    paths = image_paths()
    d["pid"] = d.participant_id.astype(str).str.strip().str.upper()
    d["image1"] = [paths.get((p, c)) for p, c in zip(d.pid, d.cat)]
    missing = d.image1.isna().sum()
    if missing:
        print(f"  ! {missing} child drawings have no row in the clean table — dropped")
        d = d[d.image1.notna()].copy()

    # bins are cut once over all categories pooled, so a row means the same age
    # range in every column
    d["bin"], edges = pd.qcut(d.age_yrs, N_BINS, labels=False, retbins=True)
    print(f"{len(d)} child drawings, ages {d.age_yrs.min():.1f}-{d.age_yrs.max():.1f}")

    # resolve once into a column; the render loop reads it back rather than redoing
    # the string surgery per cell
    d["image_path"] = [ge.resolve_image_path(p, IMAGE_ROOT) for p in d.image1]
    d = d[[p is not None and p.exists() for p in d.image_path]].copy()

    chosen = {}
    for b in range(N_BINS):
        rows = d[d["bin"] == b]
        star = None
        if pick == "one-child":
            # the child in this bin who drew the most of the six categories,
            # ties broken by closeness to the bin's median age
            target = rows.age_yrs.median()
            cand = (rows.groupby("participant_id")
                        .agg(n=("cat", "nunique"), age=("age_yrs", "first"))
                        .assign(d=lambda x: (x.age - target).abs())
                        .sort_values(["n", "d"], ascending=[False, True]))
            if not cand.empty:
                star = cand.index[0]
                print(f"  bin {b} ({edges[b]:.1f}-{edges[b+1]:.1f}y): {star}, "
                      f"{cand.iloc[0].n}/6 categories")
        for cat in CATEGORIES:
            cell = rows[rows.cat == cat]
            if cell.empty:
                print(f"  ! no drawing on disk for {cat} bin {b}")
                continue
            mine = cell[cell.participant_id == star] if star else cell.iloc[0:0]
            if not mine.empty:
                chosen[(b, cat)] = mine.iloc[0]
            elif pick == "random":
                chosen[(b, cat)] = cell.sample(1, random_state=seed + b).iloc[0]
            else:
                # one-child fallback, and the median-age rule itself
                target = cell.age_yrs.median()
                chosen[(b, cat)] = cell.iloc[(cell.age_yrs - target).abs().argmin()]
    return chosen, edges


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pick", choices=["one-child", "median-age", "random"],
                    default="one-child",
                    help="one-child: a single child per age row wherever possible")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-crop", dest="crop", action="store_false",
                    help="keep the original canvas instead of cropping to the strokes")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out) if args.out else REPO_ROOT / "figures/drawing_grid_6x6.png"

    chosen, edges = load_cells(args.pick, args.seed)
    colours = [plt.cm.viridis(VIRIDIS_END * b / (N_BINS - 1)) for b in range(N_BINS)]

    fig, axes = plt.subplots(N_BINS, len(CATEGORIES), figsize=(13.5, 14.2))
    for b in range(N_BINS):
        for j, cat in enumerate(CATEGORIES):
            ax = axes[b, j]
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor("#DDDDDD")
            r = chosen.get((b, cat))
            if r is None:
                ax.text(0.5, 0.5, "—", ha="center", va="center", color="#BBBBBB")
                continue
            ax.imshow(tinted(r.image_path, colours[b], crop=args.crop))
            ax.set_xlabel(f"{r.participant_id}  {r.age_yrs:.1f}y",
                          fontsize=6.5, color="#888888", labelpad=1.5)
            if b == 0:
                ax.set_title(cat, fontsize=14, fontweight="bold", pad=10)
            if j == 0:
                # youngest row on top; label carries the bin's actual age range
                ax.set_ylabel(f"{edges[b]:.1f}–{edges[b+1]:.1f} yrs", fontsize=11,
                              fontweight="bold", color=colours[b], labelpad=8)

    fig.suptitle("Children's drawings by category and age", fontsize=19,
                 fontweight="bold", x=0.5, y=0.985)
    fig.text(0.5, 0.962,
             {"one-child": "each row is one child drawing all six categories; "
                           "colour marks the age bin (viridis, purple = youngest)",
              "median-age": "one drawing per cell, closest to the cell's median age; "
                            "colour marks the age bin (viridis, purple = youngest)",
              "random": f"one drawing per cell, randomly sampled (seed {args.seed}); "
                        "colour marks the age bin"}[args.pick],
             ha="center", fontsize=10.5, color="#555555")
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
