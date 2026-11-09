# Drawing and describing sea animals: findings summary

*What's robust, what isn't, and the pitch for* Developmental Science. Snapshot of `main` at `623b9e8` (Nov 4, 2026), plus fixes in open PRs #1–#4 and the within-child analyses on branch `analysis/within-child-coupling` (Nov 8, 2026).

---

## 1. Study at a glance

- **Setting:** Birch Aquarium. Each participant first **described** an animal aloud, then **drew** it.
- **Animals:** crab, octopus, seahorse, shark, turtle, whale.
- **Sample after exclusions:** 89 children, 2.6–12.8 years (M = 6.8, SD = 2.3), and 20 adults, who serve as the reference "prototype."
- **Data:** 483 child drawings and 422 child descriptions. 77 children contributed both.
- **Exclusions:** 24 pilot participants; 2 with no recorded age; trials flagged for parent or sibling interference.

## 2. What the code does

| Stage | Scripts | Output |
|---|---|---|
| Clean and tidy | `preprocessing/01`–`05c` | `data/tidy/drawings.csv`, `utterances.csv` (one row per participant × animal) |
| Embeddings | `03_*` (CLIP), `03g_*` (Gemini `gemini-embedding-2`) | Drawing and masked-description embeddings |
| Similarity | `04a`–`04d` (CLIP), `04ag`–`04dg` (Gemini) | Cosine to the adult prototype; cosine to the child's own other drawings/descriptions |
| Adult-likeness by age | `analysis/06`, `06g` | Mixed models and child-level correlations |
| Category geometry | `analysis/07`, `07g` (+ `07c` in PR #2) | Within-child 6×6 RDMs; drawing-vs-description RSA |
| Hand-coded features | `analysis/08` | Feature counts and drawing/description overlap |
| Timecourse | `analysis/09a`, `09b` | Productions at 50% vs. 100% |
| LLM judge | `analysis/10`–`16` (+ `15b` in PR #4) | Gemini names the animal in the drawing and the description; scores their alignment |
| Within-child coupling | `analysis/17a`–`17d` | Item-level drawing↔description models, child-swap permutation, yoked features, adults, per-animal profile |
| Exploratory moderators | `explore/things/01`–`04` | Open THINGS datasets: which concept properties make knowledge help drawing? |

Descriptions are **masked**: the animal's name is rewritten as a pronoun, so the models can't simply read off the label.

---

## 3. Robust findings

Each of these survived my re-analysis. Unless noted, "CLIP" means the tidy-data models in `06`.

### 3.1 Both drawings and descriptions become more adult-like with age

| | CLIP: trial-level mixed model | Gemini: child-level correlation |
|---|---|---|
| Drawings | β = 0.020 SD/month, t = 8.7, p < .001 (481 trials, 88 children) | r = .69 (n = 85) |
| Descriptions | β = 0.011 SD/month, t = 4.7, p < .001 (422 trials, 79 children) | r = .60 (n = 72) |

The direction and size agree across two very different embedding models. Drawings show the steeper age slope.

### 3.2 Children who draw like adults also talk like adults, beyond age

- Per child, adult-like drawing correlates with adult-like description: r = .60 (CLIP); r = .73 (Gemini).
- **Controlling for age** (CLIP): partial r = .47, 95% CI [.27, .63], p < .001, n = 77.

### 3.3 Headline: the link holds *within* a child, item by item, and is specific to that child

The animal a child draws most adult-like is also the one they describe most adult-like, over and above that child's overall level (`analysis/17b`).

**The null this rejects:** drawing knowledge and verbal/conceptual knowledge of an animal are independent once you account for the child (age, ability, engagement). Under that null, a child's item-level deviations in drawing and in description should be uncorrelated.

- **Model:** drawing z (within animal) ~ description, within-child centred (uw) + description, child mean (ub) + age + (1 | child) + (1 | animal). 411 trials, 77 children.
- **Holds in all four model pairings,** including cross-model pairings where one embedding space can't score both modalities:

| Drawing model ~ description model | Within-child β (SE) | p | + order + effort | Random slopes |
|---|---|---|---|---|
| CLIP ~ CLIP | .105 (.044) | .018 | .106, p = .017 | .105, p = .021 |
| Gemini ~ Gemini | .122 (.055) | .028 | .118, p = .034 | .127, p = .016 |
| CLIP ~ Gemini | .144 (.056) | .010 | .132, p = .020 | .144, p = .010 |
| Gemini ~ CLIP | .093 (.041) | .024 | .093, p = .026 | .092, p = .035 |

- **Effort controls:** presentation order, strokes, drawing time and word count leave it unchanged.
- **No age interaction** (all p > .35). The between-child effect is large (β ≈ .41–.50, p < .001).
- **Child-specific** (`analysis/17c`): pairing each child's drawings with a *different* same-age child's descriptions (5,000 derangements within age quartile) gives slopes centred at 0. Own-child slopes are .103 / .115 / .141 / .088, with p = .015 / .012 / .006 / .020. So the effect is not "some animals are easier for everyone."
- **Hand-coded features agree:** a child's drawing shares more features with their *own* description than with same-animal descriptions from other children.
  - Own Dice .207 vs. shuffled within animal × age tertile .186 [.177, .195], p < .001.
  - Vs. the 5 nearest-age children: child-level Δ = .033, p = .034.

**Not all animals carry it** (`17b`, section B). Within-child r is positive in 23/24 animal × pairing cells, but octopus, seahorse and turtle (r ≈ .15–.29) carry it, while crab, shark and whale are near 0.
- Leaving out octopus, seahorse or turtle weakens the effect (p = .04–.18 across pairings). Leaving out crab, shark or whale mostly strengthens it.
- With 6 animals, this profile is descriptive. Section 4b covers attempts to explain it.

Because children described **before** drawing, the alternative reading is priming (see §6).

### 3.4 Drawings and descriptions share category geometry, at every age

Across the 6 animals, the pattern of which ones are similar to each other (e.g., shark–whale) agrees between drawings and descriptions. Corrected analysis (PR #2): complete cases, raw cosine, permutation tests.

| | Younger | Older | Older − younger (95% CI) |
|---|---|---|---|
| CLIP | ρ = .54, p = .02 | ρ = .60, p = .02 | +.05 [−.23, +.65] |
| Gemini | ρ = .48, p = .03 | ρ = .57, p = .02 | +.09 [−.31, +.49] |

Shared structure is already there in younger children. There is **no reliable age change**.

### 3.5 Children express more features with age, in both modalities

Hand-coded features per trial, on trials with both modalities:
- drawings: β = 0.84 features per SD of age, p < .001
- descriptions: β = 0.54 features per SD of age, p < .001

### 3.6 Feature overlap grows only because output grows

Raw shared features increase with age (β = 0.16, p = .015; PR #1). But:
- controlling for total features produced: β = −0.05, p = .38
- Dice overlap: β = 0.02, p = .15

The two modalities don't converge on the *same* features. Each carries content the other doesn't.

---

## 4. Promising but provisional

**LLM-judged recognizability rises with age.** Among children (411 trials):
- alignment score: β = +15.3 points per SD of age, p < .001
- the drawing and description get the same animal label in 53% of trials for younger children and 82% for older children (adults: 92%)

*Caveat:* Gemini labelled the drawing and the description **in the same call**, so each label can borrow from the other. Rerun with isolated calls (`15b`, PR #4) and validate against human ratings before reporting.

**Adults show the same pattern, underpowered** (`17a`, `17b` section C). On leave-one-out Gemini scores, adults' within-person β = .158 (SE .122, p = .20; 20 adults, 116 trials), about the same as children's (.122).
- The per-animal profile matches children's (ρ = .94; octopus, seahorse, turtle again highest).
- Adults' scores vary half as much as children's (SD of drawing similarity .024 vs .052), so a reliable adult estimate needs roughly 100–150 adults, or more animals per person.
- *Prediction:* if the coupling reflects a shared knowledge store, it should hold in adults. This is a follow-up target, e.g. aquarium parents with counterbalanced describe/draw order.

**Timecourse.** Drawings gain more from the 50% to the 100% point (β = 0.50 SD) than descriptions do (β = 0.15). But half the strokes and half the words aren't comparable units, so this belongs in the supplement at most.

## 4b. Exploratory: why do some animals carry the coupling? (no answer yet)

Full write-up: `explore/things/README.md`. Uses open data: Drawings of THINGS (28,627 drawings, 1,854 concepts, per-drawing "I know what this looks like" flag), THINGSplus norms, GPT-3 feature norms.

Within-drawer test (799 drawers): does knowing an object help its drawing's recognizability more for concepts with property P?

| Hypothesis | In our 6 animals | At scale (familiarity × P) |
|---|---|---|
| Less template-like concepts benefit more | Mixed (octopus is template-like but couples) | −.11, p = .038; Holm p = .38 |
| Less nameable concepts benefit more | Perfect rank match | +.01, p = .85 |
| Concepts with more distinctive parts benefit more | Partial | −.06, p = .24 |
| (not predicted) Size, heaviness | — | −.22 / −.19, survive Holm |

- **Bottom line:** no measured property explains the per-animal profile. The nameability match was probably chance, found after looking at about a dozen properties.
- **Limits:** DoT tests knowledge → drawing in adults online, with a coarse yes/no flag (88.5% "yes"). That is a blunt proxy for our effect.
- **For the paper:** report the profile descriptively. Test moderators only with animals chosen in advance to vary on one property.

## 5. Findings to drop (did not survive checking)

| Original claim | Why it goes |
|---|---|
| Cross-modal geometry converges with age (ρ .56 → .90) | Mostly a missing-data artifact. A null with no embedding information gives .38 → .79 (PR #2). |
| Shared features increase with age (β = .21) | Trials with only one modality were scored as 0 shared. The rest is explained by total output (PR #1). |
| Drawing and description succeed or fail together (OR ≈ 65) | Inflated by joint labelling in the judge call (PR #4). |

---

## 6. Anticipated critiques and responses

1. **Priming.** Children described before drawing, so the description may have shaped the drawing.
   *Response:* The between-child link and the shared geometry don't depend on order. For the item-level effect, the yoked feature test helps but doesn't settle it: own-description overlap beats same-age others (Dice .207 vs .186), which is what both shared knowledge *and* priming predict. The decisive fix is a counterbalanced (draw-first) follow-up. Acknowledge as a limitation.
   **Follow-on: "talking about it should shape all the drawings."** If describing primes drawing, every animal should show coupling. It is concentrated in 3 of 6 animals, and the same 3 in adults. This argues mildly against a uniform priming account, but n = 6 animals.
2. **Adult-likeness = skill, not knowledge.**
   *Response:* Stroke, time and word-count controls don't change the within-child effect. Also report models that include them.
3. **Validity of the LLM judge.**
   *Response:* Two human raters on ~150 trials; isolated calls (PR #4); report every prompt version (v1–v6).
4. **Model dependence.**
   *Response:* Report CLIP and Gemini everywhere. They agree on every robust finding above.
5. **Non-random missingness.** Number of completed categories correlates with age (r ≈ .32).
   *Response:* Complete-case sensitivity analyses throughout.
6. **Hand-coded features.**
   *Response:* Inter-rater reliability on a subset.
7. **Masking leaks.** 13 descriptions still name the animal.
   *Response:* Fix upstream, then rerun (PR #3).
8. **Scope.** A single convenience sample, 6 familiar animals, 20 adults in the prototype.
   *Response:* State it plainly; release the raw data.

## 7. Before submission

- [ ] Merge PRs #1–#4. Re-execute `07`, `07g` and `08`. Rebuild the RSA figure from complete cases.
- [ ] Fix the masking leaks; rerun embeddings and the judge; run `15b`.
- [ ] Human validation of the judge; inter-rater reliability for features.
- [ ] Continuous age everywhere (drop the median splits).
- [x] Within-child model, swap permutation, yoked features and adult models in the repo (`analysis/17a`–`17d`).
- [ ] Rerun `17a`–`17d` after the masking-leak fix (PR #3).
- [ ] Commit the `*_gemini_xmodal.csv` files (07g can't run without them). Release the raw drawings and transcripts.

---

## 8. Pitch for *Developmental Science*

**Working title:** *Drawing and describing draw on shared, item-specific category knowledge across childhood*

**Abstract (draft):**
> Children can show what they know about a category by drawing it or by describing it, yet these routes are rarely compared within the same children. At a public aquarium, 89 children (2.6–12.8 years) and 20 adults described and then drew six sea animals, yielding 483 drawings and 422 descriptions. We measured how adult-like each production was using two multimodal embedding models (CLIP and Gemini). With age, both drawings and descriptions became steadily more similar to adults' and contained more features. The two modalities were also linked beyond age. Children whose drawings were more adult-like gave more adult-like descriptions (partial r = .47). Within individual children, the animal a child drew most adult-like was also the one they described most adult-like, independent of drawing effort and description length. Drawings and descriptions also captured similar similarity relationships among the six animals, in younger and older children alike. However, the specific features children expressed in both modalities increased only in step with how much they produced overall. Drawing and talking thus appear to draw on a common, item-specific store of category knowledge that is refined across childhood, while each modality also carries content the other does not. Scalable machine-learning tools make it possible to measure this in naturalistic settings.

**Research Highlights:**
- Children aged 2–12 described and drew six animals. Both modalities became more adult-like with age, as measured by two independent embedding models.
- How adult-like a child's drawings were predicted how adult-like their descriptions were, beyond age.
- Within individual children, the animals they drew best were the ones they described best: item-level coupling between the two modalities.
- Drawings and descriptions shared category structure from early childhood, but expressed partly different features.

**Why *Developmental Science*:** The paper asks a core cognitive-development question: is one body of knowledge expressed through two output systems? It answers it within children, with computational methods the journal regularly publishes.
- Lead with the developmental question, not the tools.
- Put the within-child result in the first paragraph of the cover letter.
- Pre-empt the skill and priming objections in the Methods rather than the Discussion.

**Fallback venue:** *Open Mind* (fast, open access, computational-friendly).

---

## 9. Context: how we got here (analysis log, Nov 4–8, 2026)

Kept so the reasoning behind each analysis is recoverable.

1. **Full re-analysis and sanity check** → PRs #1–#4 (shared-features bug, RSA missingness artifact, masking leaks, joint-labelling confound in the judge).
2. **DINO embeddings** were tried as a tiebreaker between CLIP and Gemini, then dropped at Bria's request. Not in the repo.
3. **The individual-differences result looks like the most interesting finding.** Bria's framing of the null: drawing knowledge should be independent of conceptual knowledge. → within-child models (§3.3).
4. **"Talking about it should shape ALL the drawings. Is this driven by one or two categories?"** → per-animal r and leave-one-animal-out (§3.3). Answer: carried by octopus, seahorse and turtle.
5. **"Is this true in adults? It should be, if it's true in children."** → leave-one-out adult models (§4). Same direction and profile, underpowered.
6. **"So animals that AREN'T template-like are carrying the effect?"** → in-sample homogeneity (mixed) and DoT template-likeness (§4b). Weak, uncorrected.
7. **Exploratory THINGS analysis**, then THINGSplus norms, then GPT-3 part features → `explore/things/` (§4b). No moderator explains the profile; size and heaviness are unexpected survivors.
8. **"Pair different children's descriptions with each other's drawings"** → child-swap permutation (§3.3). Effect is child-specific.

**Planned follow-up (adults and kids):** more animals per person (≥ 12, chosen to vary on template-likeness), counterbalanced describe/draw order, ~100–150 adults (e.g. aquarium parents).
