#!/usr/bin/env python3
"""
preprocess.py - build the try-on inputs for every selected image.

For each image in reports/preprocess_list.csv it writes (all 768x1024, same canvas):
  data/processed/image/<Cat>/<name>.jpg          letterboxed person photo
  data/processed/pose/<Cat>/<name>.json          MediaPipe landmarks (pixels) + visibility
  data/processed/pose_vis/<Cat>/<name>.png       skeleton drawing (for checking)
  data/processed/parse/<Cat>/<name>.png          human-parsing label map (mode L)
  data/processed/agnostic_mask/<Cat>/<name>.png  white = area the model must repaint
  data/processed/agnostic/<Cat>/<name>.jpg       photo with that area grey
  data/processed/cloth/<Cat>/<name>.jpg          garment cut out on white
  data/processed/cloth_mask/<Cat>/<name>.png     garment mask
DensePose is a separate step (densepose.py) because it needs detectron2.

Settings: params.yaml -> preprocess. Already finished images are skipped (use --force to redo).

  python scripts/02_preprocess/preprocess.py --limit 5        # 5 per category, quick test
  python scripts/02_preprocess/preprocess.py                  # everything
  python scripts/02_preprocess/preprocess.py --category Saree --sam2
"""
import argparse
import json
import traceback
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageOps
from tqdm import tqdm

RAW = Path("data/raw")
OUT = Path("data/processed")
W, H = 768, 1024
# segformer_b2_clothes labels
GARMENT = [4, 5, 6, 7, 8, 17]      # upper-clothes, skirt, pants, dress, belt, scarf (dupatta)
ARMS = [14, 15]
SUBDIRS = {"image": ".jpg", "pose": ".json", "pose_vis": ".png", "parse": ".png",
           "agnostic_mask": ".png", "agnostic": ".jpg", "cloth": ".jpg", "cloth_mask": ".png"}
BONES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
         (23, 25), (25, 27), (24, 26), (26, 28), (0, 11), (0, 12)]


# ---------------------------------------------------------------- helpers
def letterbox(img, fill=255):
    """Fit into WxH without stretching; pad with white."""
    h, w = img.shape[:2]
    s = min(W / w, H / h)
    nw, nh = round(w * s), round(h * s)
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    canvas = np.full((H, W, 3), fill, np.uint8)
    x, y = (W - nw) // 2, (H - nh) // 2
    canvas[y:y + nh, x:x + nw] = img
    return canvas


def out_path(sub, row):
    p = OUT / sub / row["category"] / (Path(row["path"]).stem + SUBDIRS[sub])
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def done(row):
    return all(out_path(s, row).exists() for s in SUBDIRS)


def save_img(path, arr_rgb):
    Image.fromarray(arr_rgb).save(path, quality=95)


# ---------------------------------------------------------------- models
def make_pose(conf):
    import urllib.request
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    model = Path("models/pose_landmarker_heavy.task")
    if not model.exists():
        model.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task", model)
    opts = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.IMAGE, num_poses=1,
        min_pose_detection_confidence=conf)
    return vision.PoseLandmarker.create_from_options(opts)


def make_parser(name, device):
    import torch
    from transformers import AutoModelForSemanticSegmentation, SegformerImageProcessor
    proc = SegformerImageProcessor.from_pretrained(name)
    model = AutoModelForSemanticSegmentation.from_pretrained(name).to(device).eval()

    @torch.no_grad()
    def parse(img):
        x = proc(images=img, return_tensors="pt").to(device)
        logits = model(**x).logits
        logits = torch.nn.functional.interpolate(logits, size=img.shape[:2], mode="bilinear")
        return logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
    return parse


def make_sam2(cfg, ckpt, device):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    pred = SAM2ImagePredictor(build_sam2(cfg, ckpt, device=device))

    def refine(img, mask):
        ys, xs = np.where(mask > 0)
        if len(xs) < 50:
            return mask
        pred.set_image(img)
        box = np.array([xs.min(), ys.min(), xs.max(), ys.max()])
        m, _, _ = pred.predict(box=box, multimask_output=False)
        m = (m[0] > 0).astype(np.uint8)
        # SAM must agree with the parser: reject if it drifts far from the parsed region
        inter = (m & mask).sum()
        union = (m | mask).sum()
        return m if union and inter / union > 0.6 else mask
    return refine


# ---------------------------------------------------------------- per image
def pose_json(pose, img):
    import mediapipe as mp
    r = pose.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img)))
    if not r.pose_landmarks:
        return None
    pts = [{"x": l.x * W, "y": l.y * H, "visibility": float(l.visibility or 0)}
           for l in r.pose_landmarks[0]]
    return {"width": W, "height": H, "landmarks": pts}


def draw_pose(pj):
    canvas = np.zeros((H, W, 3), np.uint8)
    if pj is None:
        return canvas
    p = pj["landmarks"]
    for a, b in BONES:
        if p[a]["visibility"] > 0.3 and p[b]["visibility"] > 0.3:
            cv2.line(canvas, (int(p[a]["x"]), int(p[a]["y"])), (int(p[b]["x"]), int(p[b]["y"])), (255, 255, 255), 4)
    for q in p[:29]:
        if q["visibility"] > 0.3:
            cv2.circle(canvas, (int(q["x"]), int(q["y"])), 5, (0, 255, 0), -1)
    return canvas


def agnostic_mask(parse, P):
    """Garment + arms, grown a little, never covering face/hair/shoes."""
    m = np.isin(parse, GARMENT + ARMS).astype(np.uint8)
    k = P["mask_dilate_px"]
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    protect = np.isin(parse, [1, 2, 3, 9, 10, 11])   # hat, hair, sunglasses, shoes, face
    m[protect] = 0
    return m


def garment_mask(parse, img, refine):
    m = np.isin(parse, GARMENT).astype(np.uint8)
    # drop tiny specks, keep regions >0.1% of the frame
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] > 0.001 * W * H]
    m = np.isin(lab, keep).astype(np.uint8)
    if refine is not None and m.any():
        m = refine(img, m)
    return m


def process(row, models, P):
    raw = np.array(ImageOps.exif_transpose(Image.open(RAW / row["path"])).convert("RGB"))
    img = letterbox(raw)
    parse = models["parse"](img)
    pj = pose_json(models["pose"], img)
    amask = agnostic_mask(parse, P)
    gmask = garment_mask(parse, img, models.get("sam2"))

    agn = img.copy()
    agn[amask > 0] = 128
    cloth = np.full_like(img, 255)
    cloth[gmask > 0] = img[gmask > 0]

    save_img(out_path("image", row), img)
    json.dump(pj, open(out_path("pose", row), "w"))
    cv2.imwrite(str(out_path("pose_vis", row)), draw_pose(pj))
    Image.fromarray(parse, "L").save(out_path("parse", row))
    Image.fromarray(amask * 255, "L").save(out_path("agnostic_mask", row))
    save_img(out_path("agnostic", row), agn)
    save_img(out_path("cloth", row), cloth)
    Image.fromarray(gmask * 255, "L").save(out_path("cloth_mask", row))
    return pj is not None, float(gmask.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="images per category (test run)")
    ap.add_argument("--category", help="only this category")
    ap.add_argument("--sam2", action="store_true", help="refine garment mask with SAM 2")
    ap.add_argument("--force", action="store_true", help="redo finished images")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    import torch
    P = yaml.safe_load(open("params.yaml"))["preprocess"]
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")

    lst = Path("reports/preprocess_list.csv")
    if not lst.exists():
        raise SystemExit("run scripts/02_preprocess/select_images.py first")
    df = pd.read_csv(lst, dtype=str)
    if a.category:
        df = df[df.category == a.category]
    if a.limit:
        df = df.groupby("category", group_keys=False).head(a.limit)
    rows = df.to_dict("records")
    if not a.force:
        rows = [r for r in rows if not done(r)]
    print(f"{len(rows)} images to process on {device}")

    models = {"pose": make_pose(P["mediapipe_confidence"]),
              "parse": make_parser(P["parsing_model"], device)}
    if a.sam2:
        models["sam2"] = make_sam2(P["sam2_config"], P["sam2_checkpoint"], device)

    log = []
    for r in tqdm(rows):
        try:
            has_pose, garment_frac = process(r, models, P)
            note = "" if has_pose else "pose_not_found"
            if garment_frac < 0.03:
                note = (note + " garment_tiny").strip()
            log.append({"path": r["path"], "status": "ok", "note": note, "garment_frac": round(garment_frac, 3)})
        except Exception:
            log.append({"path": r["path"], "status": "error", "note": traceback.format_exc().splitlines()[-1]})
    Path("reports").mkdir(exist_ok=True)
    name = "preprocess_log.csv" if not a.limit else "sample_preprocess_log.csv"
    pd.DataFrame(log).to_csv(Path("reports") / name, index=False)
    s = pd.DataFrame(log)
    if len(s):
        print(s.status.value_counts().to_string())
        print(f"flagged notes: {(s.note != '').sum()} (see reports/{name})")


if __name__ == "__main__":
    main()