#!/usr/bin/env python3
"""
smart_filter.py - label every raw image as keep / borderline / remove.

Nothing is deleted or moved. The script only writes:
  reports/filter_flags.csv      one row per image (scores, tag, reasons, metadata)
  reports/filter_metrics.json   counts, tracked by DVC (dvc exp show)

Settings come from params.yaml -> filter, so `dvc exp run -S filter.x=...` works.

Test on a small sample first (writes reports/sample_*.csv/json, not the real ones):
  python scripts/01_filter/smart_filter.py --limit 10
"""
import argparse
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageOps
from tqdm import tqdm

RAW = Path("data/raw")
REPORTS = Path("reports")
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}
# MediaPipe pose indices: nose, shoulders, hips, ankles
JOINTS = [11, 12, 23, 24]
L_SH, R_SH = 11, 12


# ----------------------------------------------------------------- loading
def collect(limit, seed=0):
    """Walk data/raw/<category>/ and merge each category's metadata.json."""
    rows = []
    for cat_dir in sorted(p for p in RAW.iterdir() if p.is_dir()):
        meta_file = cat_dir / "metadata.json"
        meta = json.load(open(meta_file)) if meta_file.exists() else {}
        by_name = {v.get("filename"): v for v in meta.values()}
        files = sorted(f for f in cat_dir.iterdir() if f.suffix.lower() in IMG_EXT)
        if limit:
            files = random.Random(seed).sample(files, min(limit, len(files)))
        for f in files:
            m = by_name.get(f.name, {})
            rows.append({
                "path": f.relative_to(RAW).as_posix(),
                "category": cat_dir.name[:1].upper() + cat_dir.name[1:],
                "filename": f.name,
                "brand": m.get("brand", ""),
                "title": m.get("title", ""),
                "product_url": m.get("product_url", ""),
                "image_url": m.get("image_url", ""),
                "rm": [], "bd": [],  # remove / borderline reasons
            })
    return rows


def load_rgb(path):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    return np.array(img)


def downscale(img, long_side):
    h, w = img.shape[:2]
    s = long_side / max(h, w)
    if s >= 1:
        return img
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


# ----------------------------------------------------------------- checks
def brisque_score(img):
    import torch
    import piq
    t = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    return float(piq.brisque(t, data_range=1.0).item())

def count_people(yolo, img, min_height):
    """Return (n_similar, n_background, (top, bottom) of main box, phone position)."""
    H = img.shape[0]
    res = yolo(img[..., ::-1], classes=[0, 67], conf=0.25, verbose=False)[0]  # 0=person, 67=phone
    xyxy = res.boxes.xyxy.cpu().numpy()
    cls = res.boxes.cls.cpu().numpy()
    conf = res.boxes.conf.cpu().numpy()
    people = [(b, c) for b, k, c in zip(xyxy, cls, conf)
              if k == 0 and (b[3] - b[1]) / H >= min_height]
    phones = [b for b, k, c in zip(xyxy, cls, conf) if k == 67 and c >= 0.3]
    if not people:
        return 0, 0, None, None
    people.sort(key=lambda p: p[0][3] - p[0][1], reverse=True)
    main = people[0][0]
    main_h = main[3] - main[1]

    def overlap(a, b):  # share of the smaller box that lies inside the other
        w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
        return w * h / small if small > 0 else 1.0

    # other people must overlap little with the main person AND be confident
    others = [b for b, c in people[1:] if overlap(main, b) < 0.5 and c >= 0.5]
    n_similar = sum(1 for b in others if (b[3] - b[1]) >= 0.5 * main_h)

    phone = None
    for p in phones:
        cx, cy = (p[0] + p[2]) / 2, (p[1] + p[3]) / 2
        if main[0] <= cx <= main[2] and main[1] <= cy <= main[3]:
            if (cy - main[1]) / main_h < 0.4:
                phone = "top"
                break
            phone = "body"
    return 1 + n_similar, len(others) - n_similar, (main[1] / H, main[3] / H), phone

def make_pose_model(min_conf):
    """MediaPipe Tasks API (the old mp.solutions API was removed)."""
    import urllib.request
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    model = Path("models/pose_landmarker_heavy.task")
    if not model.exists():
        model.parent.mkdir(exist_ok=True)
        url = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
               "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task")
        print("downloading pose model ...")
        urllib.request.urlretrieve(url, model)
    opts = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=min_conf,
    )
    return vision.PoseLandmarker.create_from_options(opts)

def pose_checks(pose, img):
    """Return (joints visible, yaw, ankle_y, leg_vertical) or four Nones."""
    import mediapipe as mp
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img))
    r = pose.detect(mp_img)
    if not r.pose_landmarks:
        return None, None, None, None
    lm = r.pose_landmarks[0]
    vis = [
        (lm[i].visibility or 0) >= 0.5 and 0 <= lm[i].x <= 1 and 0 <= lm[i].y <= 1
        for i in JOINTS
    ]
    frac = sum(vis) / len(JOINTS)
    ankle_y = max(lm[27].y, lm[28].y)  # 1.0 = bottom edge of the image
    knees_ok = all((lm[i].visibility or 0) >= 0.5 and 0 <= lm[i].y <= 1 for i in (25, 26))
    yaw, leg = None, None
    if r.pose_world_landmarks:
        w = r.pose_world_landmarks[0]
        dx = abs(w[L_SH].x - w[R_SH].x)
        dz = abs(w[L_SH].z - w[R_SH].z)
        yaw = math.degrees(math.atan2(dz, dx)) if (dx + dz) > 1e-6 else None
        if knees_ok:
            ratios = []
            for hip, knee in ((23, 25), (24, 26)):
                ddx, ddy, ddz = (w[knee].x - w[hip].x, w[knee].y - w[hip].y, w[knee].z - w[hip].z)
                length = math.sqrt(ddx * ddx + ddy * ddy + ddz * ddz)
                if length > 1e-6:
                    ratios.append(abs(ddy) / length)  # 1 = thigh vertical, 0 = horizontal
            leg = sum(ratios) / len(ratios) if ratios else None
    return frac, yaw, ankle_y, leg


def sharpness(gray, grid=4, pct=80):
    """Sharpness of the sharper tiles, so flat fabric/backgrounds don't dominate."""
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    h, w = lap.shape
    vals = [
        lap[i * h // grid:(i + 1) * h // grid, j * w // grid:(j + 1) * w // grid].var()
        for i in range(grid) for j in range(grid)
    ]
    return float(np.percentile(vals, pct))

def process_image(row, P, yolo, pose):
    try:
        img = load_rgb(RAW / row["path"])
    except Exception:
        row["rm"].append("unreadable")
        return
    h, w = img.shape[:2]
    row["width"], row["height"] = w, h
    if min(h, w) < P["min_image_size"]:
        row["rm"].append("too_small")

    m = P["borderline_margin"]
    upper_only = row["category"] in P["upper_body_categories"]

    # sharpness (sharpest tiles, fixed size)
    small = downscale(img, P["resize_to"])
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    row["blur"] = round(sharpness(gray), 1)
    if row["blur"] < P["blur_threshold"] * (1 - m):
        row["rm"].append("blurry")
    elif row["blur"] < P["blur_threshold"] * (1 + m):
        row["bd"].append("soft")

    # BRISQUE: always recorded, only flags if brisque_flags is true
    if P["use_brisque"]:
        try:
            row["brisque"] = round(brisque_score(small), 1)
            if P["brisque_flags"]:
                if row["brisque"] > P["brisque_threshold"] * (1 + m):
                    row["rm"].append("bad_quality")
                elif row["brisque"] > P["brisque_threshold"] * (1 - m):
                    row["bd"].append("mediocre_quality")
        except Exception:
            row["brisque"] = np.nan

    # people and phones
    det = downscale(img, 1024)
    n_main, n_bg, box, phone = count_people(yolo, det, P["person_min_height"])
    row["n_people"], row["n_background"] = n_main, n_bg
    pose_result = None
    if n_main == 0:
        # YOLO found nobody; let MediaPipe try before giving up
        pose_result = pose_checks(pose, det)
        if pose_result[0] is None:
            row["rm"].append("no_person")
            return
        n_main = 1
        row["n_people"] = 1
    if n_main > 1:
        row["rm"].append("multiple_people")
    elif n_bg > 0:
        row["bd"].append("background_person")
    if phone == "top":
        row["rm"].append("phone_covering_face")
    elif phone == "body":
        row["bd"].append("phone_in_hand")

    if box:
        row["box_top"], row["box_bottom"] = round(box[0], 3), round(box[1], 3)
        if box[0] < P["edge_margin"] and P["flag_head_crop"]:
            row["bd"].append("head_cropped")

    # pose: full body, facing the camera, standing
    frac, yaw, ankle_y, leg = pose_result or pose_checks(pose, det)
    row["limbs_visible"] = None if frac is None else round(frac, 2)
    row["yaw_deg"] = None if yaw is None else round(yaw, 1)
    row["ankle_y"] = None if ankle_y is None else round(ankle_y, 2)
    row["leg_vertical"] = None if leg is None else round(leg, 2)
    if frac is None:
        row["bd"].append("pose_not_found")
        return
    if ankle_y > P["ankle_below_frame"] and not upper_only:
        row["rm"].append("cropped_body")
    if frac < P["min_limbs_remove"]:
        row["rm"].append("not_full_body")
    elif frac < P["min_limbs_visible"]:
        row["bd"].append("partial_body")
    if yaw is not None:
        if yaw > P["pose_angle_threshold"]:
            row["rm"].append("not_frontal")
        elif yaw > P["angle_borderline_start"]:
            row["bd"].append("slightly_turned")
    if leg is not None:
        if leg < P["seated_remove"]:
            row["rm"].append("seated")
        elif leg < P["seated_borderline"]:
            row["bd"].append("maybe_seated")

# ----------------------------------------------------------------- duplicates
def embed_all(rows, model_name, device):
    import torch
    from transformers import AutoImageProcessor, AutoModel

    proc = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    idx, feats, batch, batch_idx = [], [], [], []

    def flush():
        if not batch:
            return
        x = proc(images=batch, return_tensors="pt")["pixel_values"].to(device)
        with torch.no_grad():
            f = model(pixel_values=x).last_hidden_state[:, 0]
        f = torch.nn.functional.normalize(f, dim=-1).cpu().numpy()
        feats.append(f)
        idx.extend(batch_idx)
        batch.clear()
        batch_idx.clear()

    for i, row in enumerate(tqdm(rows, desc="embeddings")):
        if "unreadable" in row["rm"]:
            continue
        try:
            batch.append(Image.fromarray(downscale(load_rgb(RAW / row["path"]), 512)))
            batch_idx.append(i)
        except Exception:
            continue
        if len(batch) == 32:
            flush()
    flush()
    return np.array(idx), (np.concatenate(feats) if feats else np.zeros((0, 1)))


def find_duplicates(rows, P, device):
    idx, E = embed_all(rows, P["dedup_model"], device)
    parent = list(range(len(idx)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    chunk = 1024
    for s in range(0, len(idx), chunk):
        sims = E[s:s + chunk] @ E.T
        for a, b in np.argwhere(sims >= P["dedup_similarity"]):
            a += s
            if b > a:
                parent[find(b)] = find(a)

    groups = {}
    for k in range(len(idx)):
        groups.setdefault(find(k), []).append(idx[k])
    gid = 0
    for members in groups.values():
        if len(members) < 2:
            continue
        gid += 1
        # keep the best image: not already removed, then sharpest
        best = max(members, key=lambda i: (not rows[i]["rm"], rows[i].get("blur", 0)))
        for i in members:
            rows[i]["dup_group"] = gid
            if i != best:
                rows[i]["rm"].append("duplicate")


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="random images per category (test run)")
    args = ap.parse_args()

    P = yaml.safe_load(open("params.yaml"))["filter"]
    REPORTS.mkdir(exist_ok=True)

    import torch
    from ultralytics import YOLO

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", device)

    yolo = YOLO("yolov8n.pt")
    pose = make_pose_model(P["pose_confidence"])

    rows = collect(args.limit)
    print(f"{len(rows)} images")
    for row in tqdm(rows, desc="checks"):
        process_image(row, P, yolo, pose)
    find_duplicates(rows, P, device)

    for row in rows:
        row["tag"] = "remove" if row["rm"] else ("borderline" if row["bd"] else "keep")
        row["reasons"] = ";".join(row["rm"] + row["bd"])
        del row["rm"], row["bd"]

    df = pd.DataFrame(rows)
    reasons = {}
    for r in df["reasons"]:
        for name in filter(None, r.split(";")):
            reasons[name] = reasons.get(name, 0) + 1
    counts = df["tag"].value_counts()
    metrics = {
        "total": len(df),
        "keep": int(counts.get("keep", 0)),
        "borderline": int(counts.get("borderline", 0)),
        "remove": int(counts.get("remove", 0)),
        "reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
        "by_category": {
            c: {t: int((g["tag"] == t).sum()) for t in ["keep", "borderline", "remove"]}
            for c, g in df.groupby("category")
        },
    }

    prefix = "sample_" if args.limit else "filter_"
    df.to_csv(REPORTS / f"{prefix}flags.csv", index=False)
    suffix = "metrics" if not args.limit else "metrics"
    json.dump(metrics, open(REPORTS / f"{prefix}{suffix}.json", "w"), indent=2)
    print(json.dumps({k: metrics[k] for k in ["total", "keep", "borderline", "remove"]}, indent=2))


if __name__ == "__main__":
    main()