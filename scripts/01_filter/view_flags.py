#!/usr/bin/env python3
"""
view_flags.py - open the filter results in FiftyOne as an image grid.

  python scripts/01_filter/view_flags.py                      # reports/filter_flags.csv
  python scripts/01_filter/view_flags.py reports/sample_flags.csv
"""
import sys
from pathlib import Path

import fiftyone as fo
import pandas as pd

RAW = Path("data/raw").resolve()
csv = Path(sys.argv[1] if len(sys.argv) > 1 else "reports/filter_flags.csv")
df = pd.read_csv(csv).fillna("")

samples = []
for r in df.itertuples():
    s = fo.Sample(filepath=str(RAW / r.path))
    s["auto_tag"] = r.tag
    s["reasons"] = [x for x in str(r.reasons).split(";") if x]
    s["category"] = r.category
    s["brand"] = r.brand
    s["title"] = r.title
    for col in ["blur", "brisque", "yaw_deg", "limbs_visible", "n_people", "n_background", "ankle_y", "leg_vertical"]:
        v = getattr(r, col, "")
        s[col] = float(v) if v != "" else None
    s["dup_group"] = int(r.dup_group) if getattr(r, "dup_group", "") != "" else None
    samples.append(s)

ds = fo.Dataset("aks_filter_review", overwrite=True)
ds.add_sample_field("reasons", fo.ListField, subfield=fo.StringField)
ds.add_samples(samples)
print(ds.count_values("auto_tag"))
session = fo.launch_app(ds)
session.wait()