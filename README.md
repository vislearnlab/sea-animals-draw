# What do drawing and speaking capture about how children express their knowledge of the world?

How much overlap exists between children's visual and linguistic expressions of the same concept, and how does this overlap shift across development? This repository contains the analysis scripts and data for a study in which children (ages 2–12) drew and described six sea animal categories (crab, octopus, seahorse, shark, turtle, whale) at the Birch Aquarium.

## Setup

```
pip install -r requirements.txt
```

R scripts need `tidyverse`, `viridis`, `tidytext` and `lmerTest`. 

Notebooks and `.Rmd` files run from their own directory; `.py` scripts run from anywhere. Figures are written to `figures/`.

## Participants

- **Children:** BD025–BD124 (BD001–BD024 are pilots, excluded)
- **Adults:** AD001–AD020 (the adult prototype)
- BD085 and BD097 have no recorded age and are dropped from age analyses.
- Trials flagged `draw_interference` or `utterance_interference` are excluded from all analyses.

## Data

| Folder | Contents | In git |
|---|---|---|
| `data/tidy/` | `drawings.csv`, `utterances.csv`: one row per participant × animal. **Start here.** | yes |
| `data/embeddings/gemini/` | Gemini embeddings and drawing↔label similarities | yes |
| `data/preprocessed/` | Cosine similarities (`gemini/` for the Gemini branch) | yes |
| `data/test_pairs/` | Hand-picked good/bad stimuli and their judgments | yes |
| `data/judgments/` | Within-child alignment judgments | yes |
| `data/raw/`, `data/clean/` | Drawings, audio exports, transcripts, ages | no (but will release publically upon publication) |

Key tidy columns: `*_exists` and `*_interference` (exclusion flags); `trial_num_animal` (presentation order, 1–6); `*_cossim_adults` (similarity to the adult prototype); `*_probability_{animal}` (within-child similarity); `*_features` (hand-coded features). `utterance_full_masked_filtered` is the transcript with the animal's name masked and fillers removed.

## Pipeline

**Needs** marks what a script requires beyond the committed data: *local* = `data/raw/` or `data/clean/`, *API* = Gemini key.

### Preprocessing (`preprocessing/`)

| Script | Does | Needs |
|---|---|---|
| `01_draw_mongo_retrieval` | Pulls drawings from MongoDB (documentation only) | MongoDB |
| `02_clean_data` | Drops pilots, adds ages | local |
| `03_*_generate_clip_embedding` | CLIP embeddings | local, CLIP |
| `03g_draw_generate_gemini_embedding` | Gemini drawing embeddings, label similarity, vision descriptions | local, API |
| `03g_talk_generate_gemini_embedding` | Gemini utterance embeddings, label similarity | local, API |
| `04a`–`04d` | CLIP similarity to adults (`a`, `b`) and to self (`c`, `d`) | local |
| `04ag`–`04dg` | Same on Gemini embeddings, excluding interference trials | — |
| `05a_create_tidy_data` | Builds `data/tidy/` | local |
| `05b_add_trial_num.py` | Adds presentation order to tidy (run after `05a`) | local |
| `05c_create_test_pairs.py` | Builds `data/test_pairs/` from tidy | — |
| `gemini_embed.py` | Shared Gemini helpers, imported by the scripts above | — |
| `check_masking.py` | Flags masked descriptions that still contain the animal's name; exits 1 on any leak | — |

A `g` suffix means the same step with Gemini instead of CLIP.

### Analysis (`analysis/`)

| Script | Does | Needs |
|---|---|---|
| `06`–`09b` | CLIP analyses: similarity to adults, RSMs, features, 50% vs 100% timepoints | — |
| `06g_correlation_analysis_gemini` | Similarity to adults by age; drawing vs description | — |
| `07g_rsm_toself_analysis_gemini` | Within-child RSMs, drawing/description↔label RDMs | — |
| `10_judge_alignment_test_pairs.py` | Validates the alignment prompt on test pairs | local, API |
| `11_visualize_alignment_test_pairs.Rmd` | Plots `10` | — |
| `12_pair_gallery.py` | Contact sheet of test pairs and scores | local |
| `13_drawing_grid.py` | Grid of drawings by animal and age | local |
| `14_additional_diagnostics.Rmd` | Presentation-order and bounding-box checks | — |
| `15_judge_alignment_within_child.py` | Judges each child's drawing against their own description | local, API |
| `16_visualize_alignment_within_child.Rmd` | Plots `15`, with age models | — |

`PROMPT_VERSION` in `10` sets the judge prompt for `10`, `12` and `15`; `11` and `16` set it separately. Bumping it re-judges everything and keeps old rows.

## Known gaps

- Gemini half-time drawing embeddings have not been generated, so `draw_half_cossim_adults` is empty in `data/preprocessed/gemini/`.
- `05a` rebuilds tidy without the presentation-order columns; run `05b` afterwards to get the run order (todo here: adjust for efficiency).
- 13 masked descriptions (12 children, 1 adult) still name the target animal, usually as a plural or compound ("sharks", "octopuses", "seaturtles", "spider crab"). Run `preprocessing/check_masking.py` for the list. Fix them in the upstream transcripts, then re-run the embedding (`03*`), similarity (`04*`) and judge (`15`) steps.
