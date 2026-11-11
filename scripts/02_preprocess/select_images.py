"""
select_images.py - decide which images go into preprocessing.

keep                         -> used
borderline + review 'keep'   -> used
everything else              -> skipped
Nothing is copied or deleted. Writes reports/preprocess_list.csv (path, category).

Run from the repo root:
  python scripts/02_preprocess/select_images.py
"""
import pandas as pd
from pathlib import Path

REPORTS = Path("reports")


def main():
    flags = pd.read_csv(REPORTS / "filter_flags.csv", dtype=str, keep_default_na=False)
    dec = REPORTS / "review_decisions.csv"
    if dec.exists():
        decisions = pd.read_csv(dec, dtype=str, keep_default_na=False)
    else:  # merge not run yet: read the three personal files directly
        parts = [pd.read_csv(f, dtype=str, keep_default_na=False)
                 for f in sorted(REPORTS.glob("review_[a-z]*.csv")) if f.name != "review_decisions.csv"]
        decisions = pd.concat(parts) if parts else pd.DataFrame(columns=["path", "decision"])
    reviewed_keep = set(decisions.loc[decisions.decision == "keep", "path"])

    use = flags[(flags.tag == "keep") | ((flags.tag == "borderline") & flags.path.isin(reviewed_keep))]
    use[["path", "category"]].to_csv(REPORTS / "preprocess_list.csv", index=False)

    pending = flags[(flags.tag == "borderline") & ~flags.path.isin(set(decisions.path))]
    print(f"selected {len(use)} of {len(flags)} images -> reports/preprocess_list.csv")
    print(use.category.value_counts().to_string())
    if len(pending):
        print(f"\nWARNING: {len(pending)} borderline images have no review decision yet (they are excluded).")


if __name__ == "__main__":
    main()