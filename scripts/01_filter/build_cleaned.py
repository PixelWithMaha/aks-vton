#!/usr/bin/env python3
"""
build_cleaned.py - build the cleaned dataset from the filter tags and the human review.

  used = filter 'keep'  +  borderline images the reviewers marked 'keep'

Writes data/cleaned/  (one folder; the version is the git tag, e.g. v2.0-cleaned)
  images/<Category>/<file>.jpg    the selected photos (copied; data/raw is never touched)
  metadata.csv                    one row per image: path, raw_path, category, garment_type,
                                  brand, title, product_url, source, group, split

Split: test_fraction of every category goes to test (stratified). Images that look alike
(DINOv2 similarity >= group_similarity, e.g. colour variants of one shoot) form a group and
always land on the same side, so the test set never contains a near-copy of a training image.
Settings: params.yaml -> clean.

  python scripts/01_filter/build_cleaned.py
"""
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))
import smart_filter as sf  # noqa: E402  (reuses its image loading and DINOv2 embedding)

RAW = Path("data/raw")
REPORTS = Path("reports")


def select(flags, decisions):
    reviewed_keep = set(decisions.loc[decisions.decision == "keep", "path"])
    borderline = set(flags.loc[flags.tag == "borderline", "path"])
    missing = borderline - set(decisions.path)
    if missing:
        sys.exit(f"{len(missing)} borderline images have no review decision, e.g. {sorted(missing)[:3]}")
    use = flags[(flags.tag == "keep") | flags.path.isin(reviewed_keep)].copy()
    use["source"] = np.where(use.tag == "keep", "filter_keep", "review_keep")
    return use


def similarity_groups(paths, threshold, device):
    """Union-find over DINOv2 similarity; returns a group number per path."""
    rows = [{"path": p, "rm": []} for p in paths]
    idx, E, _ = sf.embed_all(rows, "facebook/dinov2-small", device)
    parent = list(range(len(paths)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for s in range(0, len(idx), 1024):
        for a, b in np.argwhere(E[s:s + 1024] @ E.T >= threshold):
            a += s
            if b > a:
                parent[find(idx[b])] = find(idx[a])
    roots = [find(i) for i in range(len(paths))]
    number = {r: n for n, r in enumerate(dict.fromkeys(roots))}
    return [number[r] for r in roots]


def split(meta, test_fraction, seed):
    """Stratified by category, whole groups at a time."""
    rng = random.Random(seed)
    meta["split"] = "train"
    for cat, g in meta.groupby("category"):
        target = round(len(g) * test_fraction)
        groups = sorted(g.group.unique())
        rng.shuffle(groups)
        sizes = g.group.value_counts()
        n = 0
        for grp in groups:
            if n >= target:
                break
            if n + sizes[grp] > target and n > 0:
                continue  # a big group would overshoot; try a smaller one
            meta.loc[meta.group == grp, "split"] = "test"
            n += sizes[grp]
    return meta


def copy_images(meta, out):
    """APFS clone copies on macOS (no extra disk space), normal copies elsewhere."""
    clone = sys.platform == "darwin"
    for r in meta.itertuples():
        dst = out / r.path
        dst.parent.mkdir(parents=True, exist_ok=True)
        if clone:
            subprocess.run(["cp", "-c", str(RAW / r.raw_path), str(dst)], check=True)
        else:
            shutil.copy2(RAW / r.raw_path, dst)


def main():
    P = yaml.safe_load(open("params.yaml"))
    C, F = P["clean"], P["filter"]
    flags = pd.read_csv(REPORTS / "filter_flags.csv", dtype=str, keep_default_na=False)
    decisions = pd.read_csv(REPORTS / "review_decisions.csv", dtype=str, keep_default_na=False)
    use = select(flags, decisions).sort_values("path").reset_index(drop=True)

    meta = pd.DataFrame({
        "path": "images/" + use.category + "/" + use.filename,
        "raw_path": use.path,
        "category": use.category,
        "garment_type": np.where(use.category.isin(F["upper_body_categories"]), "upper_body", "full_outfit"),
        "brand": use.brand,
        "title": use.title,
        "product_url": use.product_url,
        "source": use.source,
    })
    import torch
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    meta["group"] = similarity_groups(meta.raw_path.tolist(), C["group_similarity"], device)
    meta = split(meta, C["test_fraction"], C["seed"])

    out = Path("data/cleaned")
    if out.exists():
        shutil.rmtree(out)
    copy_images(meta, out)
    meta.to_csv(out / "metadata.csv", index=False)

    summary = meta.pivot_table(index="category", columns="split", values="path", aggfunc="count", fill_value=0)
    summary["test_%"] = (100 * summary.test / summary.sum(axis=1)).round(1)
    print(summary.to_string())
    print(f"\n{len(meta)} images -> {out}  | sources {meta.source.value_counts().to_dict()}"
          f" | groups with >1 image: {(meta.group.value_counts() > 1).sum()}")


if __name__ == "__main__":
    main()
