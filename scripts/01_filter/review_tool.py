#!/usr/bin/env python3
"""
review_tool.py - review your borderline images in FiftyOne; tags are saved to a CSV.

  python scripts/01_filter/review_tool.py maaham     # or maheen / izma
  python scripts/01_filter/review_tool.py --merge    # combine the 3 files

Tag each image keep, remove or unsure in FiftyOne. Your own dataset
(aks_review_<name>) is kept between runs, and the tags are copied to
reports/review_<name>.csv every 15 s and when you stop with Ctrl+C.
--merge writes reports/review_decisions.csv and lists anything missing or unsure.
"""
import csv
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

RAW = Path("data/raw").resolve()
FLAGS = Path("reports/filter_flags.csv")
OUT = Path("reports")
TEAM = {
    "maaham": ["Saree", "Sharara", "Gharara", "Anarkali", "Lehenga", "Pishwas", "Gown"],
    "maheen": ["Women_Kurta_Set", "Men_Kurta_Set", "Kaftan", "Maxi", "Angrakha", "Co-Ord-Sets"],
    "izma": ["Men_Waistcoat", "Men_Sherwani", "Men_Shalwar_Kameez", "Jacket", "Peplum", "Wrap"],
}
FIELDS = ["path", "decision", "reviewer", "time"]


def borderline():
    df = pd.read_csv(FLAGS, dtype=str, keep_default_na=False)
    return df[df.tag == "borderline"]


def load_decisions(f):
    if not f.exists():
        return {}
    with open(f) as fh:
        return {r["path"]: r for r in csv.DictReader(fh)}


def save_decisions(f, d):
    tmp = f.with_suffix(".tmp")
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS)
        w.writeheader()
        w.writerows(sorted(d.values(), key=lambda r: r["path"]))
    tmp.replace(f)  # never leaves a half-written file


def merge():
    todo = borderline()
    got = {}
    for name in TEAM:
        got.update(load_decisions(OUT / f"review_{name}.csv"))
    rows = [{"path": p, "decision": got[p]["decision"], "reviewer": got[p]["reviewer"]}
            for p in todo.path if p in got]
    pd.DataFrame(rows, columns=["path", "decision", "reviewer"]).to_csv(OUT / "review_decisions.csv", index=False)
    missing = [p for p in todo.path if p not in got]
    unsure = [r["path"] for r in rows if r["decision"] == "unsure"]
    print(f"{len(rows)} / {len(todo)} reviewed -> reports/review_decisions.csv")
    print(pd.Series([r["decision"] for r in rows]).value_counts().to_string() if rows else "")
    for cat, g in todo[todo.path.isin(missing)].groupby("category"):
        owner = next(n for n, cs in TEAM.items() if cat in cs)
        print(f"  not reviewed: {cat} ({owner}) {len(g)}")
    if unsure:
        print("unsure (decide together):")
        print("\n".join("  " + p for p in unsure))


TAGS = ("keep", "remove", "unsure")


def build_dataset(name, mine, decisions):
    """Persistent per-person FiftyOne dataset; the CSV is the source of truth for tags."""
    import fiftyone as fo
    dname = f"aks_review_{name}"
    if dname in fo.list_datasets():
        return fo.load_dataset(dname)
    samples = []
    for r in mine.itertuples():
        s = fo.Sample(filepath=str(RAW / r.path))
        s["path"] = r.path
        s["category"] = r.category
        s["reasons"] = [x for x in r.reasons.split(";") if x]
        s["brand"], s["title"], s["product_url"] = r.brand, r.title, r.product_url
        if r.path in decisions:  # e.g. new laptop: restore earlier answers
            s.tags = [decisions[r.path]["decision"]]
        samples.append(s)
    ds = fo.Dataset(dname, persistent=True)
    ds.add_sample_field("reasons", fo.ListField, subfield=fo.StringField)
    ds.add_samples(samples)
    return ds


def export(ds, out, name, decisions, quiet=False):
    """Copy FiftyOne tags into review_<name>.csv. Returns paths with more than one decision tag."""
    clash = []
    new = {}
    for path, tags in zip(ds.values("path"), ds.values("tags")):
        t = [x for x in tags if x in TAGS]
        if len(t) > 1:
            clash.append(path)
            if path in decisions:  # keep the earlier answer until the clash is fixed
                new[path] = decisions[path]
            continue
        if not t:
            continue
        old = decisions.get(path)
        time = old["time"] if old and old["decision"] == t[0] else datetime.now().isoformat(timespec="seconds")
        new[path] = {"path": path, "decision": t[0], "reviewer": name, "time": time}
    if new != decisions:
        decisions.clear()
        decisions.update(new)
        save_decisions(out, decisions)
        if not quiet:
            print(f"  saved {len(decisions)} / {len(ds)}")
    return clash


def serve(name):
    import threading
    import time
    import fiftyone as fo
    mine = borderline()
    mine = mine[mine.category.isin(TEAM[name])].sort_values("path")
    out = OUT / f"review_{name}.csv"
    decisions = load_decisions(out)
    ds = build_dataset(name, mine, decisions)
    print(f"{name}: {len(ds)} images, {len(decisions)} already done. Saving to {out} every 15 s")
    print("Tag each image keep / remove / unsure. Ctrl+C here when finished.")
    stop = threading.Event()

    def autosave():
        while not stop.wait(15):
            try:
                export(ds, out, name, decisions)
            except Exception as e:  # keep going; final save happens on exit
                print("  autosave failed:", e)

    threading.Thread(target=autosave, daemon=True).start()
    session = fo.launch_app(ds)
    try:
        session.wait()
    except KeyboardInterrupt:
        pass
    stop.set()
    clash = export(ds, out, name, decisions, quiet=True)
    print(f"\nsaved {len(decisions)} / {len(ds)} in {out}")
    if clash:
        print("these have two decision tags, remove one and run again:")
        print("\n".join("  " + p for p in clash))


if __name__ == "__main__":
    arg = sys.argv[1].lower() if len(sys.argv) > 1 else ""
    if arg == "--merge":
        merge()
    elif arg in TEAM:
        serve(arg)
    else:
        sys.exit(f"usage: review_tool.py <{'|'.join(TEAM)}>   or   review_tool.py --merge")
