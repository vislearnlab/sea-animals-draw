"""
gemini_embed.py — helpful wrappers around the Gemini embedding and vision
models

Embeddings (EMBED_MODEL):
  embed_images()        drawing image vectors
  embed_texts()         utterance / label text vectors
  label_embeddings()    "a drawing of a crab" etc, for cross-modal recognizability

Vision reads (VISION_MODEL). Four prompts over one schema, differing only in how much
they constrain the feature vocabulary — see each section's header for why:
  describe_drawing()                  literal: only what is visibly present
  describe_drawing_anatomical()       told the target category, part names inferred
  describe_drawing_descriptive()      as above, in the human coders' phrase style
  describe_drawing_coding_scheme()    as above, against the lab's fixed word bank
Each has a describe_drawings_*() batch sibling that preserves input positions.

Judgments (VISION_MODEL). Caller owns the prompt and the response schema, so prompt
variants can be iterated on without editing this module:
  judge_drawing_text()  one drawing + one prompt -> structured JSON
  judge_text()          one prompt, no image     -> structured JSON

Dataset conventions, here because every caller needs them and this is the one module
they all already import: SEA_ANIMALS, CATEGORY_ORDER, target_category_from_text1(),
resolve_image_path().

Requires:  pip install google-genai pillow numpy pandas python-dotenv
Env:       GEMINI_API_KEY
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image


class _DropAFCWarning(logging.Filter):
    """Hide google-genai's "Direct use of automatic function calling (AFC)..." notice.

    """

    def filter(self, record: logging.LogRecord) -> bool:
        return "automatic function calling" not in record.getMessage()


logging.getLogger("google_genai.models").addFilter(_DropAFCWarning())

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

EMBED_MODEL = "gemini-embedding-2"

VISION_MODEL = "gemini-3.6-flash"

# 768 dimensions for efficiency, without losing semantic resolution
DEFAULT_DIM = 768

# gemini-embedding-2 caps image inputs at 6 per request.
IMAGE_BATCH = 6
TEXT_BATCH = 16

CACHE_DIR = Path(os.environ.get("GEMINI_CACHE_DIR", "../data/embeddings/gemini/_cache"))

MAX_RETRIES = 6
BASE_BACKOFF = 2.0

_client = None


# ----------------------------------------------------------------------------
# Dataset conventions
# ----------------------------------------------------------------------------

SEA_ANIMALS = ["crab", "octopus", "seahorse", "shark", "turtle", "whale"]

# Presentation and pairing order. 10_judge_alignment_test_pairs.py order matters
CATEGORY_ORDER = ["crab", "octopus", "whale", "shark", "seahorse", "turtle"]

# Where the lab's stored absolute paths turn into this checkout's relative ones.
_PATH_ANCHOR = "sketches_full_dataset/"


def _is_missing(value) -> bool:
    """True for None, NaN and the string "nan" that CSV round-trips can produce."""
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    return str(value).strip().lower() in ("", "nan")


def target_category_from_text1(text1):
    """ "an_octopus" -> "octopus".

    Note "square" and "this_square" both collapse to "square"
    """
    if hasattr(text1, "str"):  # pandas Series
        return (text1.str.replace(r"^(a|an|this)_", "", regex=True)
                     .str.replace("_", " ")
                     .str.lower().str.strip())
    import re

    return re.sub(r"^(a|an|this)_", "", str(text1)).replace("_", " ").lower().strip()


def resolve_image_path(stored_path, root) -> Path | None:
    """Map a drawing path stored by the lab onto a local `root`.

    """
    if _is_missing(stored_path):
        return None
    p = str(stored_path)
    i = p.find(_PATH_ANCHOR)
    return Path(root) / (p[i:] if i != -1 else p)


def get_client():
    """Construct a genai client. Reads GEMINI_API_KEY from env or .env."""
    global _client
    if _client is None:
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass
        from google import genai

        if not os.environ.get("GEMINI_API_KEY"):
            raise RuntimeError(
                "GEMINI_API_KEY not set. Add it to your .env file."
            )
        _client = genai.Client()
    return _client


# ----------------------------------------------------------------------------
# Caching
# ----------------------------------------------------------------------------


def _hash(*parts: object) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(repr(p).encode() if not isinstance(p, bytes) else p)
        h.update(b"\x00")
    return h.hexdigest()[:32]


def _vec_cache_get(key: str) -> np.ndarray | None:
    p = CACHE_DIR / f"{key}.npy"
    if p.exists():
        try:
            return np.load(p)
        except Exception:
            p.unlink(missing_ok=True)
    return None


def _vec_cache_put(key: str, vec: np.ndarray) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_DIR / f"{key}.npy.tmp"
    # np.save appends ".npy" to any string/Path filename that lacks it, which would
    # silently write to "<tmp>.npy" instead of "<tmp>" and break the replace() below.
    # Passing an open file object bypasses that renaming.
    with open(tmp, "wb") as f:
        np.save(f, vec.astype(np.float32))
    tmp.replace(CACHE_DIR / f"{key}.npy")


def _json_cache_get(key: str) -> dict | None:
    p = CACHE_DIR / f"{key}.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            p.unlink(missing_ok=True)
    return None


def _json_cache_put(key: str, obj: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_DIR / f"{key}.json.tmp"
    tmp.write_text(json.dumps(obj))
    tmp.replace(CACHE_DIR / f"{key}.json")


def cache_stats() -> dict:
    if not CACHE_DIR.exists():
        return {"vectors": 0, "json": 0, "mb": 0.0}
    vecs = list(CACHE_DIR.glob("*.npy"))
    js = list(CACHE_DIR.glob("*.json"))
    mb = sum(f.stat().st_size for f in vecs + js) / 1e6
    return {"vectors": len(vecs), "json": len(js), "mb": round(mb, 1)}


# ----------------------------------------------------------------------------
# Retry
# ----------------------------------------------------------------------------


def _is_retryable(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    if "perday" in text:
        # A per-day quota (e.g. EmbedContentRequestsPerDayPerProjectPerModel-FreeTier)
        # won't clear within a retry loop's timescale — fail fast instead of burning
        # ~6 backoff attempts (and possibly more quota) against a wall that can't move
        # until the next daily reset.
        return False
    return any(
        s in text
        for s in ("429", "resource_exhausted", "500", "502", "503", "504",
                  "unavailable", "deadline", "internal", "timeout",
                  # Transport-level failures. These carry no HTTP status, so none of
                  # the codes above match them and a flaky minute of wifi or DNS would
                  # otherwise kill a long run outright — which is exactly what ended a
                  # 542-call judging run at call 98 with ConnectError [Errno 8].
                  "connect", "connection", "nodename", "name resolution",
                  "name or service not known", "reset by peer", "broken pipe",
                  "remoteprotocolerror", "readerror", "ssl")
    )


def _with_retry(fn, what: str = "request"):
    last = None
    for attempt in range(MAX_RETRIES):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — SDK error classes vary by version
            last = exc
            if not _is_retryable(exc) or attempt == MAX_RETRIES - 1:
                raise
            wait = BASE_BACKOFF * (2**attempt) + random.uniform(0, 1.0)
            print(f"  [retry {attempt + 1}/{MAX_RETRIES}] {what}: {exc} — sleeping {wait:.1f}s")
            time.sleep(wait)
    raise last  # unreachable


# ----------------------------------------------------------------------------
# Vector helpers
# ----------------------------------------------------------------------------


def l2_normalize(v: np.ndarray) -> np.ndarray:
    """MRL-truncated vectors are not re-normalized by the API, so do it here."""
    n = np.linalg.norm(v)
    return v if n == 0 else v / n


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


# ----------------------------------------------------------------------------
# Image preparation
# ----------------------------------------------------------------------------


def prep_image(
    path: str | Path,
    content_crop: bool = True,
    pad: int = 8,
    max_side: int = 1024,
) -> bytes:
    """
    Load a drawing PNG and return PNG bytes ready to send.

    Mirrors apply_content_crop=True from the CLIP branch's ImgExtractionSettings,
    but deliberately does NOT downscale to 224px — Gemini can use the extra stroke
    detail. max_side only caps pathologically large canvases to control image tokens.
    """
    img = Image.open(path)
    img.load()

    if content_crop:
        if img.mode in ("RGBA", "LA") and img.getchannel("A").getextrema()[0] < 255:
            bbox = img.getchannel("A").getbbox()
        else:
            gray = img.convert("L")
            # non-white ink; invert so getbbox() finds the strokes
            bbox = Image.eval(gray, lambda p: 255 - p).getbbox()
        if bbox:
            l, t, r, b = bbox
            l, t = max(0, l - pad), max(0, t - pad)
            r, b = min(img.width, r + pad), min(img.height, b + pad)
            img = img.crop((l, t, r, b))

    # flatten transparency onto white so strokes read as dark-on-light
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        flat = Image.new("RGBA", img.size, (255, 255, 255, 255))
        flat.alpha_composite(img)
        img = flat.convert("RGB")
    else:
        img = img.convert("RGB")

    if max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))),
                         Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


# ----------------------------------------------------------------------------
# Embedding
# ----------------------------------------------------------------------------


def _embed_batch(contents: list, dim: int, model: str) -> list[np.ndarray]:
    """One API call. Each element of `contents` must be its own Content object.

    gemini-embedding-2 aggregates multiple raw inputs into a SINGLE vector; wrapping
    each input in a Content object is what makes it return one vector per input.
    """
    from google.genai import types

    client = get_client()

    def call():
        return client.models.embed_content(
            model=model,
            contents=contents,
            config=types.EmbedContentConfig(output_dimensionality=dim),
        )

    resp = _with_retry(call, what=f"embed x{len(contents)}")
    vecs = [l2_normalize(np.array(e.values, dtype=np.float32)) for e in resp.embeddings]
    if len(vecs) != len(contents):
        raise RuntimeError(
            f"Expected {len(contents)} embeddings, got {len(vecs)}. "
            "The API aggregated the batch — reduce batch size to 1."
        )
    return vecs


def embed_texts(
    texts: Sequence[str | None],
    dim: int = DEFAULT_DIM,
    task_prefix: str | None = None,
    model: str = EMBED_MODEL,
    batch_size: int = TEXT_BATCH,
    verbose: bool = True,
) -> list[np.ndarray | None]:
    """
    Embed a sequence of strings. None/blank inputs return None (preserving position).

    task_prefix: gemini-embedding-2 has no task_type parameter — task intent goes in
    the prompt text instead. Leave as None for plain symmetric similarity, which is
    what the cossim analyses want. If you do set one, use the SAME prefix for every
    item being compared, or the geometry is not comparable.
    """
    from google.genai import types

    out: list[np.ndarray | None] = [None] * len(texts)
    todo: list[tuple[int, str]] = []

    for i, t in enumerate(texts):
        if t is None or str(t).strip() == "" or str(t).lower() == "nan":
            continue
        payload = str(t).strip()
        if task_prefix:
            payload = f"{task_prefix}\n{payload}"
        key = _hash(model, dim, "text", task_prefix, payload)
        cached = _vec_cache_get(key)
        if cached is not None:
            out[i] = cached
        else:
            todo.append((i, payload))

    if verbose:
        print(f"  texts: {len(texts)} total, {len(todo)} to fetch, "
              f"{sum(v is not None for v in out)} cached")

    for start in range(0, len(todo), batch_size):
        chunk = todo[start : start + batch_size]
        contents = [types.Content(parts=[types.Part.from_text(text=p)]) for _, p in chunk]
        try:
            vecs = _embed_batch(contents, dim, model)
        except RuntimeError:
            vecs = [_embed_batch([c], dim, model)[0] for c in contents]
        for (i, payload), v in zip(chunk, vecs):
            _vec_cache_put(_hash(model, dim, "text", task_prefix, payload), v)
            out[i] = v
        if verbose and (start // batch_size) % 5 == 0:
            print(f"    {min(start + batch_size, len(todo))}/{len(todo)}")

    return out


def embed_images(
    paths: Sequence[str | Path | None],
    dim: int = DEFAULT_DIM,
    model: str = EMBED_MODEL,
    batch_size: int = IMAGE_BATCH,
    content_crop: bool = True,
    max_side: int = 1024,
    verbose: bool = True,
) -> list[np.ndarray | None]:
    """
    Embed drawing PNGs. Missing files / None paths return None (position preserved).
    Cache key is the hash of the *prepared* image bytes, so identical drawings
    across the full and half-time sets share a cache entry.
    """
    from google.genai import types

    out: list[np.ndarray | None] = [None] * len(paths)
    todo: list[tuple[int, bytes, str]] = []
    missing = 0

    for i, p in enumerate(paths):
        if p is None or (isinstance(p, float) and np.isnan(p)):
            missing += 1
            continue
        fp = Path(str(p))
        if not fp.exists():
            missing += 1
            continue
        try:
            data = prep_image(fp, content_crop=content_crop, max_side=max_side)
        except Exception as exc:  # noqa: BLE001
            print(f"  [skip] {fp.name}: {exc}")
            missing += 1
            continue
        key = _hash(model, dim, "image", content_crop, max_side, data)
        cached = _vec_cache_get(key)
        if cached is not None:
            out[i] = cached
        else:
            todo.append((i, data, key))

    if verbose:
        print(f"  images: {len(paths)} rows, {missing} missing/unreadable, "
              f"{len(todo)} to fetch, {sum(v is not None for v in out)} cached")

    for start in range(0, len(todo), batch_size):
        chunk = todo[start : start + batch_size]
        contents = [
            types.Content(parts=[types.Part.from_bytes(data=d, mime_type="image/png")])
            for _, d, _ in chunk
        ]
        try:
            vecs = _embed_batch(contents, dim, model)
        except RuntimeError:
            vecs = [_embed_batch([c], dim, model)[0] for c in contents]
        for (i, _, key), v in zip(chunk, vecs):
            _vec_cache_put(key, v)
            out[i] = v
        if verbose:
            print(f"    {min(start + batch_size, len(todo))}/{len(todo)}")

    return out


def label_embeddings(
    categories: Iterable[str],
    template: str = "a drawing of a {}",
    dim: int = DEFAULT_DIM,
    model: str = EMBED_MODEL,
) -> dict[str, np.ndarray]:
    """
    Text vectors for category labels, for cross-modal recognizability.

    Because gemini-embedding-2 puts images and text in one space, cosine(drawing,
    label) is a direct recognizability score — the clean version of what the CLIP
    branch approximated with text_prompt="a drawing of ".
    """
    cats = list(categories)
    vecs = embed_texts([template.format(c) for c in cats], dim=dim, model=model,
                       verbose=False)
    return {c: v for c, v in zip(cats, vecs) if v is not None}


# ----------------------------------------------------------------------------
# Vision descriptions
# ----------------------------------------------------------------------------

_DESCRIBE_SCHEMA = {
    "type": "object",
    "properties": {
        "description": {"type": "string"},
        "features": {"type": "array", "items": {"type": "string"}},
        "best_guess": {"type": "string"},
        "confidence": {"type": "number"},
        "completeness": {"type": "string",
                         "enum": ["scribble", "partial", "recognizable", "detailed"]},
    },
    "required": ["description", "features", "best_guess", "confidence", "completeness"],
}

_DESCRIBE_PROMPT = """This is a drawing made by a child or an adult in a psychology study.

Describe what has been drawn, then list the discrete depicted parts or attributes you
can actually see, as short lowercase noun phrases (for example: "eight legs", "shell",
"pincers", "water"). List only what is visibly present — do not infer parts that a
typical member of the category would have. Give your best guess at the intended
subject as a single lowercase word, your confidence from 0 to 1, and how complete the
drawing looks.

Do not speculate about the age or identity of the person who drew it.
Return JSON only."""


def describe_drawing(
    path: str | Path,
    model: str = VISION_MODEL,
    prompt: str = _DESCRIBE_PROMPT,
    content_crop: bool = True,
    max_side: int = 1024,
) -> dict | None:
    """
    Structured VISION_MODEL read of one drawing. Cached like the embeddings.

    The `features` list is deliberately shaped to be comparable to the hand-coded
    draw_full_features / draw_half_features columns in the tidy data, so you can ask
    how well model-coded features track human-coded ones.
    """
    from google.genai import types

    fp = Path(str(path))
    if not fp.exists():
        return None
    try:
        data = prep_image(fp, content_crop=content_crop, max_side=max_side)
    except Exception:  # noqa: BLE001
        return None

    key = _hash(model, "describe", prompt, data)
    cached = _json_cache_get(key)
    if cached is not None:
        return cached

    client = get_client()

    def call():
        return client.models.generate_content(
            model=model,
            contents=[
                types.Part.from_bytes(data=data, mime_type="image/png"),
                types.Part.from_text(text=prompt),
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=_DESCRIBE_SCHEMA,
                temperature=0.0,
            ),
        )

    resp = _with_retry(call, what=f"describe {fp.name}")
    try:
        obj = json.loads(resp.text)
    except Exception:  # noqa: BLE001
        return None
    _json_cache_put(key, obj)
    return obj


def describe_drawings(
    paths: Sequence[str | Path | None],
    verbose: bool = True,
    **kwargs,
) -> list[dict | None]:
    out = []
    for n, p in enumerate(paths, 1):
        out.append(None if p is None else describe_drawing(p, **kwargs))
        if verbose and n % 25 == 0:
            print(f"    described {n}/{len(paths)}")
    return out


# ----------------------------------------------------------------------------
# Vision descriptions — anatomical-parts variant
# ----------------------------------------------------------------------------
#

_DESCRIBE_PROMPT_ANATOMICAL_TEMPLATE = """This is a drawing made by a child in a psychology study. The child was
asked to draw a {category}.

You are annotating this drawing the way a human coder in a developmental psychology
study would, not describing it as a vision system would. Identify the anatomical
parts of a {category} that the drawing depicts, using standard part names (for
example: "eyes", "legs", "shell", "pincers", "antennae", "tail", "fin"). Do not
describe the marks themselves (avoid things like "circle", "curved line", "scribble",
"zigzag pattern") — translate each mark into the body part it represents.

Some inference is expected and appropriate. Children's drawings, especially from
younger children, are often abstract. If a mark's position and shape plausibly
represent a specific part of a {category} — a round shape near the top with two dots
is a head with eyes, even if crudely drawn — label it with that part's name. Do not,
however, list a part that has no corresponding mark anywhere in the drawing just
because a typical {category} would have it. Every part you list should correspond to
something actually drawn, generously interpreted.

Describe what has been drawn, then list the anatomical parts under `features` as
short lowercase noun phrases. Give your confidence from 0 to 1 that this drawing
depicts a {category}, and how complete the drawing looks.

Do not speculate about the age or identity of the person who drew it.
Return JSON only."""


def describe_drawing_anatomical(
    path: str | Path,
    category: str,
    model: str = VISION_MODEL,
    content_crop: bool = True,
    max_side: int = 1024,
) -> dict | None:
    """Like describe_drawing, but told the intended category and asked for
    human-coder-style anatomical parts (generously inferred) instead of literal
    mark description. See module docstring above for why."""
    prompt = _DESCRIBE_PROMPT_ANATOMICAL_TEMPLATE.format(category=category)
    return describe_drawing(
        path, model=model, prompt=prompt, content_crop=content_crop, max_side=max_side
    )


def describe_drawings_anatomical(
    paths: Sequence[str | Path | None],
    categories: Sequence[str],
    verbose: bool = True,
    **kwargs,
) -> list[dict | None]:
    if len(paths) != len(categories):
        raise ValueError(
            f"paths and categories must be the same length, got {len(paths)} and {len(categories)}"
        )
    out = []
    for n, (p, cat) in enumerate(zip(paths, categories), 1):
        out.append(None if p is None else describe_drawing_anatomical(p, cat, **kwargs))
        if verbose and n % 25 == 0:
            print(f"    described {n}/{len(paths)}")
    return out


# ----------------------------------------------------------------------------
# Vision descriptions — descriptive (1-2 word) variant
# ----------------------------------------------------------------------------
#


_DESCRIBE_PROMPT_DESCRIPTIVE_TEMPLATE = """This is a drawing made by a child in a psychology study. The child was
asked to draw a {category}.

You are annotating this drawing the way a human coder in a developmental psychology
study would, not describing it as a vision system would. Identify the anatomical
parts of a {category} that the drawing depicts, using standard part names (for
example: "eyes", "legs", "shell", "pincers", "antennae", "tail", "fin"). Do not
describe the marks themselves (avoid things like "circle", "curved line", "scribble",
"zigzag pattern") — translate each mark into the body part it represents.

List each part under `features` as a short noun phrase of ONE OR TWO words, lowercase.
Human coders in this study wrote features this way — not just "tentacles" but "eight
tentacles", not just "tail" but "long tail" or "curly tail", not just "fins" but "side
fins" or "dorsal fin". Whenever a part has a visible count, size, shape, or position
that distinguishes it, fold that descriptor into the phrase as the first word. If a
part has no such distinguishing detail actually visible, a single word (e.g. "eyes")
is fine — do not invent a descriptor that isn't really there.

Some inference is expected and appropriate. Children's drawings, especially from
younger children, are often abstract. If a mark's position and shape plausibly
represent a specific part of a {category} — a round shape near the top with two dots
is a head with eyes, even if crudely drawn — label it with that part's name. Do not,
however, list a part that has no corresponding mark anywhere in the drawing just
because a typical {category} would have it. Every part you list should correspond to
something actually drawn, generously interpreted.

Describe what has been drawn, then list the anatomical parts under `features` as
described above. Give your confidence from 0 to 1 that this drawing depicts a
{category}, and how complete the drawing looks.

Do not speculate about the age or identity of the person who drew it.
Return JSON only."""


def describe_drawing_descriptive(
    path: str | Path,
    category: str,
    model: str = VISION_MODEL,
    content_crop: bool = True,
    max_side: int = 1024,
) -> dict | None:
    """Like describe_drawing_anatomical, but explicitly asks for one-or-two-word
    feature phrases with a count/size/position descriptor folded in when visible
    (e.g. "eight tentacles"), matching the style human coders used."""
    prompt = _DESCRIBE_PROMPT_DESCRIPTIVE_TEMPLATE.format(category=category)
    return describe_drawing(
        path, model=model, prompt=prompt, content_crop=content_crop, max_side=max_side
    )


def describe_drawings_descriptive(
    paths: Sequence[str | Path | None],
    categories: Sequence[str],
    verbose: bool = True,
    **kwargs,
) -> list[dict | None]:
    if len(paths) != len(categories):
        raise ValueError(
            f"paths and categories must be the same length, got {len(paths)} and {len(categories)}"
        )
    out = []
    for n, (p, cat) in enumerate(zip(paths, categories), 1):
        out.append(None if p is None else describe_drawing_descriptive(p, cat, **kwargs))
        if verbose and n % 25 == 0:
            print(f"    described {n}/{len(paths)}")
    return out


# ----------------------------------------------------------------------------
# Vision descriptions — fixed coding scheme variant that human coders utilized
# ----------------------------------------------------------------------------
#

_FEATURE_CODING_SCHEME = {
    "whale": """- fins
- long body (as in a whale-like body)
- body (as in an animal like/square body)
- eyes
- baleen
- side fins
- dorsal fin
- tail
- blowhole
- squirt water
- nose (basically round front, not a circular head)
- head
- barnacles (look like bumps)
- legs (if the legs are human-like)""",
    "turtle": """- shell
- legs (eg flippers, fins, legs… appendages basically)
- head
- tail
- eyes
- mouth
- neck""",
    "shark": """- long body
- body (if it doesn't look like a shark body)
- pointy nose (basically head… doesn't have to be pointy… just want to distinguish
  between round seahorse head for ex)
- eyes
- teeth
- side fins (side)
- dorsal fin (top)
- tail
- species (if drew hammerhead or other recognizable type)
- gills
- bumps
- head (if a round head was drawn)
- mouth (eg if smiling like a cartoon)""",
    "seahorse": """- curly tail
- long nose
- belly
- fins
- vertical
- head
- eyes
- mouth
- long body
- antennae
- bumps
- live in seaweed
- bubbles
- face (eyes, mouth)""",
    "octopus": """- (eight) tentacles
- bulbous head
- suction cups
- eyes
- mouth
- body (if it looks very separate from the head, like a human body)
- ink
- live in water (if they drew background)
- small
- vertical""",
    "crab": """- antennae
- shell
- eyes
- pincers (arms with claws, or just claws)
- mouth
- legs
- head (if a separate circle)""",
}

_DESCRIBE_PROMPT_CODING_SCHEME_TEMPLATE = """This is a drawing made by a child in a psychology study. The child was
asked to draw a {category}.

You are applying a fixed human-coder coding scheme to this drawing — the same one the
developmental psychology lab that ran this study uses to code its own human
annotators' work. Below is the list of visual features coders look for in a
{category} drawing, with clarifying notes in parentheses where a feature name could
be ambiguous:

{word_bank}

Go through this list and identify which of these features are actually visible in the
drawing. List each one you find under `features`, using the exact wording from the
list above (drop any parenthetical clarification — that note is guidance for you, not
part of the label). Some inference is expected: children's drawings are often
abstract, so if a mark's position and shape plausibly represents one of these
features, count it, even if crudely drawn. Do not list a feature that has no
corresponding mark anywhere in the drawing.

If the drawing clearly shows a feature that genuinely isn't covered by the list above,
add it to `features` too, as a short lowercase noun phrase — but only for something
not already expressible with an item on the list.

Describe what has been drawn, then list the features as described above. Give your
confidence from 0 to 1 that this drawing depicts a {category}, and how complete the
drawing looks.

Do not speculate about the age or identity of the person who drew it.
Return JSON only."""


def describe_drawing_coding_scheme(
    path: str | Path,
    category: str,
    model: str = VISION_MODEL,
    content_crop: bool = True,
    max_side: int = 1024,
) -> dict | None:
    """Like describe_drawing_anatomical, but hands Gemini the lab's fixed
    per-category word bank (see _FEATURE_CODING_SCHEME) instead of leaving feature
    vocabulary up to the model."""
    if category not in _FEATURE_CODING_SCHEME:
        raise ValueError(
            f"No coding scheme for category {category!r}. "
            f"Have: {sorted(_FEATURE_CODING_SCHEME)}"
        )
    prompt = _DESCRIBE_PROMPT_CODING_SCHEME_TEMPLATE.format(
        category=category, word_bank=_FEATURE_CODING_SCHEME[category]
    )
    return describe_drawing(
        path, model=model, prompt=prompt, content_crop=content_crop, max_side=max_side
    )


def describe_drawings_coding_scheme(
    paths: Sequence[str | Path | None],
    categories: Sequence[str],
    verbose: bool = True,
    **kwargs,
) -> list[dict | None]:
    if len(paths) != len(categories):
        raise ValueError(
            f"paths and categories must be the same length, got {len(paths)} and {len(categories)}"
        )
    out = []
    for n, (p, cat) in enumerate(zip(paths, categories), 1):
        out.append(None if p is None else describe_drawing_coding_scheme(p, cat, **kwargs))
        if verbose and n % 25 == 0:
            print(f"    described {n}/{len(paths)}")
    return out


# ----------------------------------------------------------------------------
# Cross-modal judging (drawing + utterance -> alignment score)
# ----------------------------------------------------------------------------


def judge_drawing_text(
    path: str | Path,
    prompt: str,
    schema: dict,
    model: str = VISION_MODEL,
    content_crop: bool = True,
    max_side: int = 1024,
) -> dict | None:
    """
    Show Gemini one drawing plus one already-formatted prompt and return structured JSON.

    Generic on purpose: the caller owns both the prompt text and the response schema,
    so prompt variants can be iterated on without touching this module. The cache key
    covers the prompt and the schema as well as the image, so an edited prompt
    re-requests automatically while an unchanged one stays free.
    """
    from google.genai import types

    fp = Path(str(path))
    if not fp.exists():
        return None
    try:
        data = prep_image(fp, content_crop=content_crop, max_side=max_side)
    except Exception:  # noqa: BLE001
        return None

    key = _hash(model, "judge", prompt, json.dumps(schema, sort_keys=True), data)
    cached = _json_cache_get(key)
    if cached is not None:
        return cached

    client = get_client()

    def call():
        return client.models.generate_content(
            model=model,
            contents=[
                types.Part.from_bytes(data=data, mime_type="image/png"),
                types.Part.from_text(text=prompt),
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
                temperature=0.0,
            ),
        )

    resp = _with_retry(call, what=f"judge {fp.name}")
    try:
        obj = json.loads(resp.text)
    except Exception:  # noqa: BLE001
        return None
    _json_cache_put(key, obj)
    return obj


def judge_text(
    prompt: str,
    schema: dict,
    model: str = VISION_MODEL,
) -> dict | None:
    """
    Send one already-formatted prompt with no image and return structured JSON.

    The text-only sibling of judge_drawing_text, for judgments that must not see
    the drawing at all. The cache key covers the prompt and the schema; there is
    no image component, so two callers passing the same prompt share one response.
    """
    from google.genai import types

    key = _hash(model, "judge_text", prompt, json.dumps(schema, sort_keys=True))
    cached = _json_cache_get(key)
    if cached is not None:
        return cached

    client = get_client()

    def call():
        return client.models.generate_content(
            model=model,
            contents=[types.Part.from_text(text=prompt)],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
                temperature=0.0,
            ),
        )

    resp = _with_retry(call, what="judge_text")
    try:
        obj = json.loads(resp.text)
    except Exception:  # noqa: BLE001
        return None
    _json_cache_put(key, obj)
    return obj
