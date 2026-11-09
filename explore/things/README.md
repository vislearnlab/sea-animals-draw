# Exploratory: which concepts show knowledge → drawing effects? (THINGS)

**Status: exploratory, not for the main paper.** All results here are null or uncorrected, except one we didn't predict. Kept so the reasoning and the code aren't lost.

## Why this exists

In the Birch data, the within-child drawing–description coupling (`analysis/17b`) is carried by **octopus, seahorse and turtle**; crab, shark and whale are near 0. Adults show the same per-animal profile (ρ = .94 with children, `analysis/17d`).

With six animals we can't tell *why*. So we asked, at scale: across 1,854 objects, which concept properties make *knowing an object* help *drawing it* more? Candidate explanations:

- **Template-likeness.** Some animals have a canonical drawing schema (shark, crab) that anyone can produce, so knowledge adds little.
- **Nameability.** In our six animals it matched the coupling rank perfectly.
- **Distinctive parts.** Knowledge may help when an object has parts unique to it (octopus suckers, seahorse snout).

## Data (all open)

| Dataset | What we use | Source |
|---|---|---|
| Drawings of THINGS (DoT) | 28,627 drawings of 1,854 concepts; per-drawing `familiar` flag ("I know what this looks like"); recognition labels from separate participants | [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12858628/), [OSF s9yta](https://osf.io/s9yta/) |
| THINGSplus | Nameability, property ratings, size, typicality within 53 categories | [paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10991023/), [OSF jum2f](https://osf.io/jum2f/wiki/home) |
| GPT-3 feature norms | McRae-style features for all THINGS concepts, 30 runs each | [paper](https://arxiv.org/pdf/2202.03753), [GitHub](https://github.com/ViCCo-Group/semantic_features_gpt_3) |

## Pipeline (run from the repo root)

| Script | Does | Time |
|---|---|---|
| `01_download.sh` | Downloads everything into `raw/` (git-ignored, ~2.2 GB) | ~10 min |
| `02_embed_dot.py` | CLIP ViT-B/32 embeddings of DoT drawings, same crop/pad as the Birch pipeline → `raw/dot_clip.npz` | ~20 min |
| `03_build_tables.py` | Builds `derived/*.csv` (committed): recognizability, template-likeness, THINGSplus norms, GPT-3 part counts | ~2 min |
| `04_moderators.R` | Within-drawer glmers: `cbind(hits, miss) ~ familiar * property + (1\|drawer) + (1\|concept)` | ~25 min |

`derived/` is committed, so `04` runs without downloading anything.

**Definitions**
- *Recognizability:* share of recognizers whose label matched the concept.
- *Template-likeness (homog):* mean pairwise cosine among a concept's drawings, after mean-centring all embeddings.
- *GPT-3 features:* "robust" = produced in ≥ 5 of 30 runs; "part" = starts with "has …"; "distinctive" = held by ≤ 10 of 1,854 concepts.
- *Sample:* only the 799 drawers who flagged both familiar and unfamiliar objects (17,801 drawings), so familiarity is tested within person.

## Results

**Our six animals in THINGS:**

| | Coupling in kids (r) | DoT template-likeness (percentile) | Nameability | Distinctive parts (GPT-3) | Sea-animal typicality |
|---|---|---|---|---|---|
| seahorse | .24 | 68 | .74 | 3 (long snout, pouch) | .47 |
| octopus | .21 | 94 | .87 | 3 (eight legs, tentacles, suckers) | .65 |
| turtle | .15 | 91 | .88 | 1 (flippers) | .54 |
| whale | .10 | 88 | .93 | 3 (blowhole, dorsal fin) | .68 |
| shark | .05 | 99 | .97 | 1 (skeleton) | .78 |
| crab | .04 | 92 | .98 | 1 (pincers) | .59 |

**At scale, the familiarity × property interaction:**

| Property | Interaction | p | Holm p (15 norms) |
|---|---|---|---|
| Size | −.22 | < .001 | < .001 |
| Heaviness | −.19 | < .001 | .001 |
| Template-likeness | −.11 | .038 | .38 |
| Nameability | +.01 | .85 | 1 |
| Number of parts | −.04 | .41 | — |
| Number of distinctive parts | −.06 | .24 | — |
| Proportion of features that are parts | −.03 | .54 | — |
| Total / distinctive features | −.05 / −.04 | .35 / .47 | — |

- **Template-likeness:** familiarity's effect is OR 3.19 at −1 SD and 2.56 at +1 SD, so it points the predicted way, but it doesn't survive correction.
- **Joint model** (nameability + template-likeness + distinctive parts): no interaction reliable; template-likeness is closest (−.10, p = .068).
- **Main effect:** distinctive parts make drawings more recognizable overall (+.18, p = .012), but they don't change how much knowing the object helps.
- **The only survivors:** knowing an object helps *less* for big, heavy things. We didn't predict this and have no account of it.

## Take-aways

- **The nameability match among our six animals was almost certainly chance:** it was found after looking at about a dozen properties, and it doesn't replicate at scale.
- **No measured concept property explains which animals carry the coupling.** Report the per-animal profile descriptively.
- **If a follow-up tests moderators,** choose animals that vary on one preregistered property. Template-likeness is the only (weak) lead.

## Caveats

- DoT tests **knowledge → drawing**, not drawing ↔ description, in adults online. Our effect is a within-child *covariation* between two productions.
- The familiarity flag is a coarse self-report: 88.5% of drawings are "familiar."
- GPT-3 features are model-generated, not human norms. The "part" and "distinctive" rules are heuristic.
- Template-likeness uses CLIP; the Birch analyses use CLIP and Gemini.
