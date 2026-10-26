# AKS by LAAM: Project Status (Dataset Cleaning Phase)

Last updated: 2026-10-26 | Maintainer: Maheen
Team: Maheen Kamran Zeb, Maaham Rizwan, Izma Asif | Supervisor: Ms Umm-e-Ammarah
University: FAST-NUCES, in collaboration with LAAM

## 1. Project in one paragraph
Virtual try-on for South Asian multi-piece garments (kameez + trousers/sharara + dupatta, long hems, heavy embroidery).
Pipeline: user photo + LAAM catalog photo -> quality gate -> preprocessing (MediaPipe, DensePose, SAM 2, SCHP)
-> fine-tuned IDM-VTON (LoRA) -> FastAPI (<10 s). Evaluation: SSIM, LPIPS, FID + user study.
IDM-VTON's license is non-commercial (research use only).

## 2. Where we are
Phase 1 of 3 (dataset + pipeline). Current task: cleaning the raw dataset.

| Item | Status |
|---|---|
| Literature review (21 works) | Done |
| Scraper and raw dataset (9,456 images, 19 categories, 6.2 GB) | Done |
| UI prototype | Done |
| DVC + DagsHub remote set up; data pushed | Done |
| Git remotes: `origin` (GitHub) and `dagshub` | Done |
| `smart_filter.py` (auto-labelling) | Written; tuning on a 190-image sample |
| `view_flags.py` (FiftyOne viewer) | Written, working |
| Full-dataset run (`dvc exp run`) | NOT done yet |
| Manual review of borderline pile | NOT started |
| `build_cleaned.py` (builds data/cleaned/v2.0) | NOT written yet |
| Preprocessing, training, API | Later phases |

Important: the target of "7,500+ clean images" is a goal, not a result. Report real numbers only after the full run.

## 3. Repo and data layout
```
data/raw/<Category>/        images + metadata.json (one per category folder)
data/raw.dvc                DVC pointer to raw data (git-tracked)
scripts/01_filter/          smart_filter.py, view_flags.py
reports/                    flags CSV and metrics JSON (outputs)
params.yaml, dvc.yaml       settings and pipeline definition
.dvc/config                 remote URL (committed)
.dvc/config.local           PERSONAL credentials (never commit)
```
metadata.json entries (dict keyed by image id): filename, category, brand, title, product_url, image_url.
Folder name lowercase `lehenga` is normalised to `Lehenga` in the script.

## 4. How the cleaning works (plain version)
The script never deletes anything. It labels every image **keep**, **borderline** or **remove**, and records why.
Humans then review only the borderline pile in FiftyOne.

Checks per image:
- Size (min 512 px) and sharpness (Laplacian on the sharpest image tiles, so plain fabric isn't marked blurry)
- BRISQUE quality score: recorded only; it does not flag (it was unreliable on studio photos)
- People: YOLO counts people; two comparable people = remove; a smaller one = borderline (reflections and paintings land here)
- Phone detected over the face (mirror selfies) = remove
- Pose (MediaPipe): shoulders and hips visible, facing the camera (shoulder angle), feet cropped (predicted ankle below the frame), seated (thigh verticality)
- Duplicates: DINOv2 embeddings, cosine similarity >= 0.95, keep the sharpest of each group
- Upper-body categories (Men_Waistcoat, Jacket, Peplum, Wrap) are allowed to be knee-up crops
- Missing face is NOT penalised (the garment is the subject); `flag_head_crop: false`

Outputs: `reports/filter_flags.csv` (one row per image), `reports/filter_metrics.json` (counts, tracked by DVC).
Sample runs (`--limit 10`) write `reports/sample_*` instead and are not recorded by DVC.

## 5. Current settings (params.yaml -> filter)
```yaml
min_image_size: 512
resize_to: 512
blur_threshold: 100         # remove <80, borderline 80-120 (margin 0.2)
brisque_threshold: 50
use_brisque: true
brisque_flags: false
borderline_margin: 0.2
pose_confidence: 0.5
min_limbs_visible: 0.75
min_limbs_remove: 0.6
edge_margin: 0.005
ankle_below_frame: 1.05
pose_angle_threshold: 60
angle_borderline_start: 35
person_min_height: 0.15
dedup_similarity: 0.95
dedup_model: facebook/dinov2-small
upper_body_categories: [Men_Waistcoat, Jacket, Peplum, Wrap]
flag_head_crop: false
seated_remove: 0.5
seated_borderline: 0.7
```
These are starting values being tuned, not final. Final values will come from the full-run experiments.

## 6. Workflow
1. Tune on the sample: `python scripts/01_filter/smart_filter.py --limit 10`, then
   `python scripts/01_filter/view_flags.py reports/sample_flags.csv`. Click a reason in the sidebar, look at ~20-30 images, and adjust settings. Aim for borderline of about 5-10%.
2. Full run through DVC (about 40 min): `dvc exp run smart_filter`. Try 2-4 threshold variations with
   `dvc exp run smart_filter -S filter.blur_threshold=120`, compare with `dvc exp show`.
3. Pick the best run: `dvc exp apply <name>`, commit, `dvc exp push dagshub <name>`.
4. Manual review of the borderline pile in FiftyOne, split three ways; export decisions to `reports/review_decisions.csv` (keyed by image path, with reviewer name).
5. `build_cleaned.py` builds `data/cleaned/v2.0` (reviewed decision overrides the automatic tag). Then `dvc push -r dagshub_remote`, `git push` to both remotes, tag `v2.0-cleaned`.

## 7. Decisions made (with reasons)
- Flag first, delete never: all decisions are reversible and reviewable.
- Manual review only for borderline images, not all 9,456.
- Face visibility not required: try-on keeps the person's face unchanged; shoulders and neckline matter.
- Feet cropped detected via predicted ankle position, not ankle visibility: long hems hide ankles in good photos.
- Seated, phone-over-face and reflections handled as above; reflections go to human review.
- Waistcoat/jacket/peplum/wrap may be knee-up; multi-piece outfits must be full-length.
- Experiments (`dvc exp run`) only on full-data runs; the quick tuning loop uses the sample.

## 8. Lessons and known problems (useful for the paper)
- MediaPipe's old `mp.solutions` API was removed; the script uses the Tasks API (downloads a pose model to `models/`).
- MediaPipe ankle visibility fails on flared garments; this flagged most good photos at first (120 of 190 borderline).
- YOLO sometimes draws two boxes on one person (dupatta); handled by an overlap filter.
- YOLO-nano missed loose abayas; use a larger YOLO model and a MediaPipe fallback.
- BRISQUE is unreliable on smooth studio backgrounds.
- Heavy outfits may give `pose_not_found`; these go to human review.
- Back-view photos can look "frontal" to the shoulder-angle check; catch in manual review.
- The latest set of fixes (seated, phone, sharpness tiles) is applied but must be re-verified on a fresh sample run.

## 9. Open tasks
| Task | Owner |
|---|---|
| Re-run the sample, check the `seated`, `phone_covering_face`, `soft` reasons | Maheen |
| Full `dvc exp run smart_filter` and 2-4 comparison experiments | Maheen |
| Write `build_cleaned.py` and the review export script | Maheen |
| Clean up `dvc.yaml` (only `smart_filter` and `build_cleaned` for now); move `manual_review` ranges into `filter` | Maheen |
| Everyone: clone, `dvc pull`, own DagsHub token in `config.local` | Maaham, Izma |
| Commit scripts and docs; add `models/`, `*.pt` to `.gitignore` | Maheen |
| Check DagsHub free storage quota before preprocessing outputs | Maheen |
| Get written permission from LAAM for dataset use before publishing anywhere | Maheen + supervisor |
| Tag raw data: `git tag v1.0-raw a307395`, push to both remotes | Maheen |

## 10. Team rules
- Never commit `.dvc/config.local` or paste tokens in chat. A token pasted in plain text must be revoked.
- Never run `dvc gc -c` without asking the team (it deletes remote data).
- `dvc push -r dagshub_remote` whenever data changes, then `git push` for both remotes.
- Do not edit `data/raw`. All cleaning goes through the scripts.
- Report only measured numbers (use `dvc exp show`).
- Keep the Hugging Face and DagsHub repos private until LAAM permission is confirmed.

## 11. Paper ownership
Section 3 (Maheen): literature review + dataset collection and cleaning. Section 4 (Maaham): preprocessing.
Section 5 (Izma): training, fine-tuning, evaluation. All: abstract, intro, conclusion, related work.

## 12. Setup for teammates
```bash
git clone https://github.com/PixelWithMaha/aks-VTON.git && cd aks-VTON
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
dvc remote modify --local dagshub_remote auth basic
dvc remote modify --local dagshub_remote user <your DagsHub username>
dvc remote modify --local dagshub_remote password <your OWN DagsHub token>
dvc pull -r dagshub_remote
```
Add the `dagshub` git remote (`https://dagshub.com/PixelWithMaha/aks-dataset.git`) to push there.