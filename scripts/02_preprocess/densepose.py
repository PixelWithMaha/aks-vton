#!/usr/bin/env python3
"""
densepose.py - DensePose maps for IDM-VTON (data/processed/densepose/<Cat>/<name>.png).

Needs Linux + CUDA + detectron2 and a clone of IDM-VTON (it ships apply_net.py and the config).
Written after IDM-VTON's gradio_demo/app.py; if it errors, compare with that file in your clone.

One-time setup (separate environment recommended):
  git clone https://github.com/yisol/IDM-VTON.git
  # download model_final_162be9.pkl into IDM-VTON/ckpt/densepose/  (from the IDM-VTON HF space / repo)
  pip install 'git+https://github.com/facebookresearch/detectron2.git'

Run from the repo root:
  python scripts/02_preprocess/densepose.py --idm-dir ../IDM-VTON --limit 5
  python scripts/02_preprocess/densepose.py --idm-dir ../IDM-VTON
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idm-dir", required=True, help="path to the IDM-VTON clone")
    ap.add_argument("--limit", type=int, default=0, help="images per category")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    idm = Path(a.idm_dir).resolve()
    sys.path.insert(0, str(idm / "gradio_demo"))
    sys.path.insert(0, str(idm))
    os.chdir(idm)                      # config / checkpoint paths below are relative to the clone
    import apply_net
    from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation

    args = apply_net.create_argument_parser().parse_args((
        "show", "./configs/densepose_rcnn_R_50_FPN_s1x.yaml",
        "./ckpt/densepose/model_final_162be9.pkl", "dp_segm", "-v",
        "--opts", "MODEL.DEVICE", "cuda"))

    files = sorted((PROC / "image").rglob("*.jpg"))
    if a.limit:
        per = {}
        files = [f for f in files if per.setdefault(f.parent.name, []).append(1) is None
                 and len(per[f.parent.name]) <= a.limit]
    fails = 0
    for f in tqdm(files):
        out = PROC / "densepose" / f.parent.name / (f.stem + ".png")
        if out.exists() and not a.force:
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            img = Image.open(f).convert("RGB").resize((384, 512))
            arr = convert_PIL_to_numpy(_apply_exif_orientation(img), format="BGR")
            dp = args.func(args, arr)[:, :, ::-1]
            Image.fromarray(dp).resize((768, 1024)).save(out)
        except Exception as e:
            fails += 1
            print(f"failed {f.name}: {e}")
    print(f"done, {fails} failures")


if __name__ == "__main__":
    main()