# Does knowing an object help drawing it more for some kinds of concepts?
#
# Within-drawer test on Drawings of THINGS: each drawer flagged per object whether they knew
# what it looks like (`familiar`). Outcome = recognizability of each drawing (hits/misses
# from independent recognizers). Concept property P is a moderator if fam x P is reliable:
#   cbind(hits, miss) ~ fam * scale(P) + (1 | drawer) + (1 | concept)
# Restricted to drawers who flagged both familiar and unfamiliar objects.
#
# Motivation: in the Birch data, octopus/seahorse/turtle carry the within-child
# drawing-description coupling and crab/shark/whale don't. Candidate explanations tested
# here at scale: template-likeness (homog), nameability, part features, other norms.
#
# Run from the repo root (after 03):  Rscript explore/things/04_moderators.R
# Slow: ~25 min (one glmer per property).

suppressMessages({library(tidyverse); library(lme4)})
setwd("explore/things/derived")
tp <- read_csv("thingsplus_concept.csv", show_col_types = FALSE) %>% mutate(log_freq = log10(subtlex + 1))
gf <- read_csv("gpt3_concept_features.csv", show_col_types = FALSE) %>% rename(uniqueID = concept_id)
d <- read_csv("dot_drawings.csv", show_col_types = FALSE) %>%
  mutate(uniqueID = str_replace_all(concept, " ", "_")) %>%
  left_join(read_csv("dot_concepts.csv", show_col_types = FALSE) %>% select(concept, homog), by = "concept") %>%
  left_join(tp, by = "uniqueID") %>% left_join(gf, by = "uniqueID") %>%
  filter(!is.na(homog)) %>%
  mutate(hits = round(recog * n_rec), miss = n_rec - hits, fam = as.integer(familiar))
vary <- d %>% group_by(subject_id) %>% summarise(v = n_distinct(fam)) %>% filter(v > 1)
d <- semi_join(d, vary, by = "subject_id")
cat("drawers with both familiar & unfamiliar:", nrow(vary), " drawings:", nrow(d), "\n")

ctl <- glmerControl(optimizer = "bobyqa")
fit1 <- function(p) {
  x <- d %>% filter(!is.na(.data[[p]])) %>% mutate(z = scale(.data[[p]])[, 1])
  m <- glmer(cbind(hits, miss) ~ fam * z + (1 | subject_id) + (1 | concept), x, family = binomial, control = ctl)
  cs <- summary(m)$coef
  tibble(property = p, n = nrow(x), main_b = cs["z", 1], fam_x_b = cs["fam:z", 1], se = cs["fam:z", 2], p = cs["fam:z", 4])
}

# 1. Template-likeness, with simple effects
x <- d %>% mutate(hz = scale(homog)[, 1])
m <- glmer(cbind(hits, miss) ~ fam * hz + (1 | subject_id) + (1 | concept), x, family = binomial, control = ctl)
cat("\n== 1. template-likeness ==\n"); print(round(summary(m)$coef, 3))
for (h in c(-1, 1)) {
  mm <- glmer(cbind(hits, miss) ~ fam * hc + (1 | subject_id) + (1 | concept), mutate(x, hc = hz - h),
              family = binomial, control = ctl)
  cat(sprintf("familiarity effect at homog %+d SD: b=%.3f (OR %.2f)\n", h, fixef(mm)["fam"], exp(fixef(mm)["fam"])))
}

# 2. THINGSplus norms, Holm-corrected across the family
props <- c("nameability", "name_consistency", "homog", "manmade", "lives", "moves", "natural", "heavy", "grasp",
           "pleasant", "arousal", "size", "concreteness", "log_freq", "pct_known")
res <- map_dfr(props, fit1) %>% mutate(p_holm = p.adjust(p, "holm")) %>% arrange(p)
cat("\n== 2. THINGSplus norms (fam x property) ==\n")
print(res %>% mutate(across(c(main_b, fam_x_b, se, p, p_holm), ~ round(., 3))), n = 30)

# 3. GPT-3 part features
cat("\n== 3. GPT-3 part features (fam x property) ==\n")
print(map_dfr(c("n_part", "n_distinct_part", "prop_part", "n_feat", "n_distinct_feat"), fit1) %>%
        mutate(across(c(main_b, fam_x_b, se, p), ~ round(., 3))))

# 4. Joint: nameability + template-likeness + distinctive parts
x <- d %>% filter(!is.na(n_distinct_part), !is.na(nameability)) %>%
  mutate(nz = scale(nameability)[, 1], hz = scale(homog)[, 1], pz = scale(n_distinct_part)[, 1])
m <- glmer(cbind(hits, miss) ~ fam * (nz + hz + pz) + (1 | subject_id) + (1 | concept), x, family = binomial, control = ctl)
cat("\n== 4. joint model ==\n"); print(round(summary(m)$coef, 3))
