"""Embed Drawings of THINGS PNGs with OpenAI CLIP ViT-B/32, same crop/pad as the Birch pipeline.

Needs open_clip and the OpenAI ViT-B/32 weights at ~/.cache/clip/ViT-B-32.pt.
Output: explore/things/raw/dot_clip.npz (29,893 x 512; git-ignored, ~60 MB). ~20 min on Apple MPS.

Run from the repo root:  python explore/things/02_embed_dot.py
"""
import os
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image

png_dir = Path("explore/things/raw/dot/pngs")
out = "explore/things/raw/dot_clip.npz"
paths = sorted(png_dir.rglob("*.png"))
model = open_clip.load_openai_model(os.path.expanduser("~/.cache/clip/ViT-B-32.pt"), device="cpu")
model.eval()
dev = "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(dev)
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)


def prep(p, pad=8):
    """Crop to ink + pad, centre on a white square, resize to 224, CLIP-normalize."""
    img = Image.open(p)
    img.load()
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGBA", img.size, "white")
        bg.alpha_composite(img.convert("RGBA"))
        img = bg
    img = img.convert("RGB")
    bbox = Image.eval(img.convert("L"), lambda v: 255 - v).getbbox()
    if bbox:
        l, t, r, b = bbox
        img = img.crop((max(0, l - pad), max(0, t - pad), min(img.width, r + pad), min(img.height, b + pad)))
    side = max(img.size)
    sq = Image.new("RGB", (side, side), "white")
    sq.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
    a = np.asarray(sq.resize((224, 224), Image.BICUBIC), np.float32) / 255
    return ((a - MEAN) / STD).transpose(2, 0, 1)


embs = []
with torch.no_grad():
    for i in range(0, len(paths), 64):
        x = torch.from_numpy(np.stack([prep(p) for p in paths[i:i + 64]])).to(dev)
        embs.append(model.encode_image(x).float().cpu().numpy())
        if i % 3200 == 0:
            print(i, len(paths), flush=True)
E = np.concatenate(embs)
np.savez(out, emb=E, sketch_id=np.array([p.stem for p in paths]))
print("saved", E.shape)
