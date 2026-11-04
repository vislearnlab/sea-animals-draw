"""
Flag masked descriptions in which the target animal's name still appears.

`utterance_masked` / `utterance_full_masked_filtered` / `utterance_half_masked_filtered`
are hand-edited versions of the transcript with the animal's name rewritten (usually
as a pronoun). Every embedding (CLIP, Gemini) and the Gemini alignment judge reads
these columns, so a leftover name gives the model the category label outright. That
inflates description-to-adult similarity, label RDMs and judge accuracy for the
affected trials.

The patterns cover plurals and spelling variants ("sharks", "octopuses", "octopi",
"sea horse"). They deliberately do NOT flag other animals' names, or near-synonyms such
as shark species ("great white", "hammerhead"); those are listed separately as
"possible cues" for a human to decide on.

Exits with status 1 if any leak is found, so it can gate the preprocessing pipeline.

    python preprocessing/check_masking.py
"""

import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
UTTERANCES_CSV = REPO_ROOT / "data/tidy/utterances.csv"
MASKED_COLS = ["utterance_masked", "utterance_full_masked_filtered",
               "utterance_half_masked_filtered"]

NAME_PATTERNS = {
    "crab": r"\bcrabs?\b",
    "octopus": r"\boctop(?:us|uses|i|odes)\b",
    "seahorse": r"\bsea[\s-]?horses?\b",
    "shark": r"\bsharks?\b",
    "turtle": r"\b(?:sea[\s-]?)?turtles?\b",
    "whale": r"\bwhales?\b",
}

# Strong category cues that are not the name itself. Reported, not counted as leaks.
CUE_PATTERNS = {
    "shark": r"\b(?:great white|hammerhead|tiger shark|bull shark|mako)\b",
    "whale": r"\b(?:orca|humpback|hump[\s-]?backed|blue whale|beluga|narwhal)\b",
    "turtle": r"\b(?:tortoise|loggerhead|leatherback)\b",
}


def main() -> int:
    u = pd.read_csv(UTTERANCES_CSV)
    u = u[(u["utterance_exists"] == 1) & (u["utterance_interference"] == 0)]

    leaks, cues = [], []
    for _, r in u.iterrows():
        cat = str(r["target_category"]).lower().strip()
        for col in MASKED_COLS:
            text = str(r.get(col, "") or "")
            m = re.search(NAME_PATTERNS[cat], text, flags=re.I)
            if m:
                leaks.append((r["participant_id"], cat, col, m.group(0), text))
            if cat in CUE_PATTERNS and col == "utterance_masked":
                c = re.search(CUE_PATTERNS[cat], text, flags=re.I)
                if c:
                    cues.append((r["participant_id"], cat, c.group(0)))

    trials = sorted({(p, c) for p, c, *_ in leaks})
    print(f"{len(trials)} trials with the target name still in a masked column "
          f"(of {len(u)} valid utterance trials)\n")
    for p, c in trials:
        rows = [x for x in leaks if x[0] == p and x[1] == c]
        cols = ", ".join(sorted({x[2] for x in rows}))
        print(f"  {p:<6} {c:<9} '{rows[0][3]}'  in: {cols}")
        print(f"         \"{rows[0][4][:200]}\"")

    if cues:
        print(f"\n{len(cues)} possible category cues (not counted as leaks; review by hand):")
        for p, c, w in cues:
            print(f"  {p:<6} {c:<9} '{w}'")

    return 1 if trials else 0


if __name__ == "__main__":
    sys.exit(main())
