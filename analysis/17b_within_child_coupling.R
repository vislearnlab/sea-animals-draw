# Within-child (item-level) coupling between drawing and description.
#
# Question: over and above a child's overall level, is the animal they DESCRIBE most
# adult-like also the one they DRAW most adult-like?
#
# The null we're testing against: drawing knowledge and verbal/conceptual knowledge of an
# animal are independent once you account for the child (age, ability, engagement).
#
# Method
#   - Adult-likeness = cosine to the adult prototype, z-scored within animal.
#   - Description score split into a within-child part (uw = z minus that child's mean)
#     and a between-child part (ub = child's mean).
#   - drawing_z ~ uw + ub + age + (1 | child) + (1 | animal)
#   - All four model pairings (drawing model x description model: CLIP/Gemini), so the
#     result can't come from one embedding space scoring both modalities.
#   - Robustness: + presentation order + effort (strokes, drawing time, word count);
#     random slopes for uw; uw x age.
#   - Per animal: within-child r in each animal; leave-one-animal-out betas.
#   - Adults: same model on leave-one-out Gemini scores (needs 17a).
#
# Run from the repo root:  Rscript analysis/17b_within_child_coupling.R
# Child-swap permutation and yoked feature tests are in 17c.

suppressMessages({library(tidyverse); library(lmerTest)})

d0 <- read_csv("data/tidy/drawings.csv", show_col_types = FALSE) %>%
  filter(draw_exists == 1, draw_interference == 0, is_child, !is.na(age_months))
u0 <- read_csv("data/tidy/utterances.csv", show_col_types = FALSE) %>%
  filter(utterance_exists == 1, utterance_interference == 0, is_child, !is.na(age_months))
gd <- read_csv("data/preprocessed/gemini/draw_cossim_adults.csv", show_col_types = FALSE) %>%
  select(participant_id, target_category, g_draw = draw_cossim_adults)
gu <- read_csv("data/preprocessed/gemini/talk_cossim_adults.csv", show_col_types = FALSE) %>%
  select(participant_id, target_category, g_utt = utterance_full_cossim_adults)

b <- inner_join(
  d0 %>% select(participant_id, target_category, age_months, trial_num_animal, num_strokes, draw_duration,
                c_draw = draw_cossim_adults),
  u0 %>% select(participant_id, target_category, c_utt = utterance_full_cossim_adults,
                txt = utterance_full_masked_filtered),
  by = c("participant_id", "target_category")) %>%
  left_join(gd, by = c("participant_id", "target_category")) %>%
  left_join(gu, by = c("participant_id", "target_category")) %>%
  mutate(nw = str_count(txt, "\\S+"))

prep <- function(x, dcol, ucol) {
  x %>% filter(!is.na(.data[[dcol]]), !is.na(.data[[ucol]])) %>%
    group_by(target_category) %>%
    mutate(dz = scale(.data[[dcol]])[, 1], uz = scale(.data[[ucol]])[, 1]) %>% ungroup() %>%
    group_by(participant_id) %>%
    mutate(ub = mean(uz), uw = uz - ub, dw = dz - mean(dz)) %>% ungroup()
}
g <- function(m, term = "uw") {
  cs <- coef(summary(m))
  sprintf("b=%.3f se=%.3f p=%.3f", cs[term, 1], cs[term, 2], cs[term, 5])
}

PAIRS <- list(c("c_draw", "c_utt", "CLIP draw ~ CLIP desc"),
              c("g_draw", "g_utt", "Gemini draw ~ Gemini desc"),
              c("c_draw", "g_utt", "CLIP draw ~ Gemini desc"),
              c("g_draw", "c_utt", "Gemini draw ~ CLIP desc"))
CATS <- c("crab", "octopus", "seahorse", "shark", "turtle", "whale")

cat("== A. Within-child coupling, children ==\n")
for (pr in PAIRS) {
  x <- prep(b, pr[1], pr[2])
  f1 <- lmer(dz ~ uw + ub + scale(age_months) + (1 | participant_id) + (1 | target_category), x)
  f2 <- lmer(dz ~ uw + ub + scale(age_months) + factor(trial_num_animal) + scale(log(num_strokes + 1)) +
               scale(log(draw_duration)) + scale(log(nw + 1)) + (1 | participant_id) + (1 | target_category), x)
  f3 <- lmer(dz ~ uw + ub + scale(age_months) + (1 + uw | participant_id) + (1 | target_category), x)
  f4 <- lmer(dz ~ uw * scale(age_months) + ub + (1 | participant_id) + (1 | target_category), x)
  cat(sprintf("%-26s n=%d kids=%d\n   base:           %s   (between-child ub: %s)\n   +order+effort:  %s\n   random slope:   %s\n   uw x age:       %s\n",
              pr[3], nrow(x), n_distinct(x$participant_id), g(f1), g(f1, "ub"), g(f2), g(f3),
              g(f4, "uw:scale(age_months)")))
}

cat("\n== B. Per animal ==\n")
for (pr in PAIRS) {
  x <- prep(b, pr[1], pr[2])
  per <- sapply(CATS, function(k) { s <- filter(x, target_category == k); cor(s$dw, s$uw) })
  loo <- sapply(CATS, function(k) {
    y <- prep(filter(b, target_category != k), pr[1], pr[2])
    m <- lmer(dz ~ uw + ub + scale(age_months) + (1 | participant_id) + (1 | target_category), y)
    cs <- coef(summary(m))["uw", ]
    sprintf("%.3f (p=%.3f)", cs[1], cs[5])
  })
  cat(pr[3], "\n   within-child r:      ", paste(sprintf("%s %.2f", CATS, per), collapse = " | "),
      "\n   leave-one-animal-out:", paste(sprintf("-%s %s", CATS, loo), collapse = " | "), "\n")
}

cat("\n== C. Adults vs children (Gemini, leave-one-out adult prototype; from 17a) ==\n")
loo_path <- "data/derived/loo_adult_sims.csv"
if (!file.exists(loo_path)) stop("run analysis/17a_loo_adult_similarity.py first")
x <- read_csv(loo_path, show_col_types = FALSE)
fit_group <- function(s, lab) {
  s <- s %>% group_by(cat) %>% mutate(dz = scale(d)[, 1], uz = scale(u)[, 1]) %>% ungroup() %>%
    group_by(pid) %>% mutate(ub = mean(uz), uw = uz - ub, dw = dz - mean(dz)) %>% ungroup()
  m <- lmer(dz ~ uw + ub + (1 | pid) + (1 | cat), s)
  per <- s %>% group_by(cat) %>% summarise(r = cor(dw, uw), .groups = "drop")
  cat(sprintf("%-9s n=%d people=%d  within %s | between %s\n   within r by animal: %s\n",
              lab, nrow(s), n_distinct(s$pid), g(m), g(m, "ub"),
              paste(sprintf("%s %.2f", per$cat, per$r), collapse = " | ")))
}
fit_group(filter(x, !adult), "children")
fit_group(filter(x, adult), "adults")
print(x %>% group_by(adult) %>% summarise(sd_draw = sd(d), sd_desc = sd(u)))
