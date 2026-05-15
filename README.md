# What do drawing and speaking capture about how children express their knowledge of the world?

How much overlap exists between children's visual and linguistic expressions of the same concept, and how does this overlap shift across development? This repository contains the analysis scripts and data for a study in which children (ages 2–12) drew and described six sea animal categories (crab, octopus, seahorse, shark, turtle, whale) at the Birch Aquarium.

**To reproduce all analyses and figures, you only need `data/tidy/` and the scripts in `analysis/`.** The preprocessing scripts (`preprocessing/01–05`) are included for transparency and document how the tidy data was generated from raw drawings and transcripts, but are not runnable without access to the raw data.

---

## Participants

- **Children:** BD025–BD124 (BD001–BD024 are pilot participants, excluded)
- **Adults:** AD001–AD020 (used to compute adult prototype embeddings)
- BD085 and BD097 are excluded from age-based analyses due to a data collection error (missing age)

---

## Repository structure

```
data/
  raw/               # Raw MongoDB output (drawings, metadata)
  clean/             # Cleaned metadata, transcripts, age table
  embeddings/        # CLIP embeddings for drawings and utterances
  preprocessed/      # Cosine similarity outputs (pre-tidy)
  tidy/              # Canonical analysis-ready CSVs (start here)
    drawings.csv
    utterances.csv
data_intermediates/
  summaries/         # trial_summaries.txt — participant/trial counts
preprocessing/       # Scripts 01–05
analysis/            # Scripts 06–09b
figures/             # All output figures
```

---

## Tidy data

The two files in `data/tidy/` are the canonical starting point for all analyses. Each has one row per participant × sea animal category and includes adults and children, with `is_adult`/`is_child` flags.

**`drawings.csv`** — drawing metrics per participant × category:
- `draw_exists`, `draw_interference` — exclusion flags
- `trial_duration`, `num_strokes`, `draw_duration`, `mean_intensity`, `bounding_box` — drawing metadata
- `draw_cossim_adults`, `draw_half_cossim_adults` — cosine similarity to adult prototype at 100% and 50% of drawing
- `draw_cossim_{category}`, `draw_probability_{category}` — within-subject similarity for RSM analyses
- `draw_full_features`, `draw_half_features` — hand-coded visual features at 100% and 50%

**`utterances.csv`** — description metrics per participant × category:
- `utterance_exists`, `utterance_interference` — exclusion flags
- `utterance_ns`, `utterance_masked` — raw and name-scrubbed transcripts
- `utterance_full_masked_filtered`, `utterance_half_masked_filtered` — filtered transcripts
- `utterance_full_cossim_adults`, `utterance_half_cossim_adults` — cosine similarity to adult prototype at 100% and 50% of utterance
- `utterance_cossim_{category}`, `utterance_probability_{category}` — within-subject similarity for RSM analyses
- `utterance_full_visual_features`, `utterance_full_semantic_features` — hand-coded features at 100%
- `utterance_half_visual_features`, `utterance_half_semantic_features` — hand-coded features at 50%

**Standard exclusion filters used in all analysis scripts:**
```r
drawings <- read_csv("data/tidy/drawings.csv") %>%
  filter(is_child, draw_exists == 1, draw_interference == 0)

utterances <- read_csv("data/tidy/utterances.csv") %>%
  filter(is_child, utterance_exists == 1, utterance_interference == 0)
```

---

## Preprocessing pipeline

Scripts 01–05 are in `preprocessing/` and are included for documentation only — they are not runnable without the raw MongoDB data and drawing image files. Run scripts in order if you have access to the raw data.

### 01_draw_mongo_retrieval
Extracts drawings and stroke data from the MongoDB database.

| | Path |
|---|---|
| **Output** | `data/raw/mongo_output/highres/sketched_full_dataset/` |
| **Output** | `data/raw/mongo_output/highres/AllDescriptives_images_final_birch_run_v1.csv` |

### 02_clean_data
Excludes pilot participants (BD001–BD024), adds age data, renames columns to the format the embedding generator expects (`image1` for file path, `text1` for category).

| | Path |
|---|---|
| **Input** | `data/raw/mongo_output/highres/AllDescriptives_images_final_birch_run_v1.csv` |
| **Input** | `data/clean/participant_age_table.csv` |
| **Input** | `data/clean/half_and_full_utterance_features.csv` |
| **Output** | `data/clean/drawing_highres_clean.csv` |
| **Output** | `data/clean/transcripts_raw_clean.csv` |

### 03_draw_generate_clip_embedding
Generates CLIP image embeddings for all drawings. **Note:** due to a known issue with the embedding generator's `output_path` argument, embeddings are written to `preprocessing/output/image_embeddings/` and must be **moved by hand** to `data/embeddings/highres_sketch_embeddings/image_embeddings/` before running the second half of the script. The second half column-wise z-scores the embeddings, merges metadata, and filters to sea animal categories only.

| | Path |
|---|---|
| **Input** | `data/clean/drawing_highres_clean.csv` |
| **Output** (hand-moved) | `data/embeddings/highres_sketch_embeddings/image_embeddings/clip_image_embeddings_csv.csv` |
| **Output** | `data/embeddings/highres_sketch_embeddings/image_embeddings/draw_embeddings_with_metadata.csv` |

### 03_talk_generate_clip_embedding
Filters utterances, generates CLIP text embeddings from the masked utterance column, and column-wise z-scores them.

| | Path |
|---|---|
| **Input** | `data/clean/transcripts_raw_clean.csv` |
| **Output** | `data/embeddings/utterance_half_and_full_embeddings.csv` |

### 04a_draw_generate_cossim_adults
Computes adult prototype embeddings (mean over adult participants) per category, then computes each child's cosine similarity to that prototype for both the full drawing and the 50%-timepoint drawing. Half-time embeddings are column-wise z-scored here. Adults are dropped from the output.

| | Path |
|---|---|
| **Input** | `data/embeddings/highres_sketch_embeddings/image_embeddings/draw_embeddings_with_metadata.csv` |
| **Input** | `data/embeddings/half_time_sketch_embeddings/` |
| **Output** | `data/preprocessed/draw_cossim_adults.csv` |

### 04b_talk_generate_cossim_adults
Same as 04a but for utterances. Produces `utterance_full_cossim_adults` and `utterance_half_cossim_adults`.

| | Path |
|---|---|
| **Input** | `data/embeddings/utterance_half_and_full_embeddings.csv` |
| **Output** | `data/preprocessed/talk_cossim_adults.csv` |

### 04c_draw_generate_cossim_toself
For each participant × category drawing, computes cosine similarity to the participant's own drawings of the other five categories. Applies softmax to yield `draw_probability_{category}`. Adults included.

| | Path |
|---|---|
| **Input** | `data/embeddings/highres_sketch_embeddings/image_embeddings/draw_embeddings_with_metadata.csv` |
| **Output** | `data/preprocessed/draw_cossim_toself.csv` |

### 04d_talk_generate_cossim_toself
Same as 04c but for utterances. Yields `utterance_probability_{category}`. Adults included.

| | Path |
|---|---|
| **Input** | `data/embeddings/utterance_half_and_full_embeddings.csv` |
| **Output** | `data/preprocessed/talk_cossim_toself.csv` |

### 05_create_tidy_data
Combines all metadata, hand-coded features, and preprocessed cosine similarity values into the two canonical tidy CSVs. Excludes participants with missing age data (BD085, BD097).

| | Path |
|---|---|
| **Input** | `data/clean/half_and_full_utterance_features.csv` |
| **Input** | `data/clean/participant_age_table.csv` |
| **Input** | `data/clean/drawing_highres_clean.csv` |
| **Input** | `data/embeddings/half_time_sketch_embeddings/half_time_clip_embeddings.csv` |
| **Input** | `data/preprocessed/draw_cossim_adults.csv` |
| **Input** | `data/preprocessed/draw_cossim_toself.csv` |
| **Input** | `data/preprocessed/talk_cossim_adults.csv` |
| **Input** | `data/preprocessed/talk_cossim_toself.csv` |
| **Output** | `data/tidy/drawings.csv` |
| **Output** | `data/tidy/utterances.csv` |

---

## Analysis scripts

All scripts are in `analysis/` and read from `data/tidy/`. Run independently of one another (no ordering requirement).

### 06_correlation_analysis
Computes within-modality age effects and cross-modal correlation between drawing and description similarity to adults. Produces Figs 3A, 3B, 4.


### 07_rsm_toself_analysis
Writes participant/trial summary statistics, then builds representational similarity matrices (RSMs) from within-subject softmax probabilities, correlates RSMs across modalities for younger vs. older children, and plots cross-modal scatter. Produces Figs 5, 6.


### 08_features_analysis
Plots hand-coded feature proportions by category and feature counts by age, for drawings and descriptions at both 50% and 100% timepoints. Produces Figs 7, 8, 9, 10, 13, 14, 15, 16.


### 09a_draw_time_analysis
Plots drawing cosine similarity to adult prototype at 50% vs. 100% of drawing (z-scored within category), fits mixed models for timepoint and age effects, and plots developmental trajectories by age bin. Produces Figs 11A, 12A.


### 09b_talk_time_analysis
Same as 09a but for utterances. Produces Figs 11B, 12B.

