# Preprocessing Guide (Phase 3)

Goal: turn the cleaned images into the inputs IDM-VTON needs (pose, parsing, masks, cloth cutouts, DensePose).
Scripts live in `scripts/02_preprocess/`. Results go to `data/processed/`.

| Person | Job |
|---|---|
| **Maheen** | Sets up the GPU machine, runs the main preprocessing, saves results with DVC and Git |
| **Izma** | Sets up and runs DensePose (the hardest part), helps check quality |

Both of you do Part 0 and Part 1 first. Both of you do the quality check in Part 4.

---

## Part 0. Rules for both of us

1. **Never paste your DagsHub token in chat or commit it.** It goes only in `.dvc/config.local`.
2. **Do not edit `data/raw`.**
3. **Only one person uses the GPU machine at a time.** Say in our chat before you start and when you finish.
4. **Always test with `--limit 5` before a full run.**
5. **Never run `dvc gc -c`.** Never run `dvc repro` or `dvc exp run` on a `preprocess` stage (it deletes `data/processed` before rerunning).
6. Report only numbers that come from `reports/preprocess_log.csv`.

---

## Part 1. Using the remote GPU machine over AnyDesk

AnyDesk shows the university computer's screen on yours. Anything you type runs on THAT computer.

**First time (both of us):**
1. Connect with the AnyDesk ID and password from the university.
2. Open a terminal on the remote machine:
   - Linux: Ctrl+Alt+T, or search "Terminal".
   - Windows: search "PowerShell", or "Ubuntu" if WSL is installed.
3. Check the GPU works:
   ```bash
   nvidia-smi
   ```
   You should see a table with a GPU name and memory. If you get "command not found", tell our supervisor (the NVIDIA driver is missing).
4. Check the operating system. DensePose (Part 5) is much easier on **Linux** or **WSL2 Ubuntu**:
   ```bash
   uname -a          # Linux
   ```
   On plain Windows, do Parts 2 to 4 and 6 only, and ask our supervisor about Linux or WSL for Part 5.

**Tips for working over AnyDesk**
- Use AnyDesk's **file transfer** (the file icon in the top bar) or Git/DVC to move files. Do not copy big folders by hand.
- Long jobs: do **not** lock or log out of the remote Windows session. On Linux, run inside `tmux` so the job survives a disconnect:
  ```bash
  tmux new -s prep        # start
  # ... run your command ...
  # press Ctrl+B then D to leave it running
  tmux attach -t prep     # come back later
  ```
- The scripts skip finished images, so if a run is interrupted, run the same command again and it continues.
- Close AnyDesk when done, but leave the machine on if a job is running.

---

## Part 2. One-time setup on the GPU machine (Maheen)

```bash
git clone https://github.com/PixelWithMaha/aks-VTON.git
cd aks-VTON
python -m venv venv
source venv/bin/activate              # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Check PyTorch sees the GPU:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
It must print `True`. If it prints `False`, install the CUDA build of PyTorch from https://pytorch.org/get-started/locally/ (pick your CUDA version from `nvidia-smi`), then check again.

Log in to DagsHub with **your own** token (DagsHub, then profile, Settings, Tokens):
```bash
dvc remote modify --local dagshub_remote auth basic
dvc remote modify --local dagshub_remote user <your DagsHub username>
dvc remote modify --local dagshub_remote password <your token>
dvc pull -r dagshub_remote            # about 6 GB, takes a while
```
Izma needs to be added as a collaborator on the DagsHub repo (Settings, Collaborators) before she can pull. When she works on the same machine, she does the same login with her own token.

Check that `data/raw/` now has the category folders.

---

## Part 3. Selecting images and the quick test (Maheen)

```bash
git pull
python scripts/02_preprocess/select_images.py
```
Expected: about 8,389 images selected. If it warns that some borderline images have no review decision, ask Maaham to push her final `reports/review_*.csv` files, `git pull`, and run it again.

Now the test (about 95 images):
```bash
python scripts/02_preprocess/preprocess.py --limit 5
```
The first run downloads two models (MediaPipe and SegFormer). This is normal and needs internet.

Then do Part 4 before the full run. If it crashes, read the last lines and see Part 7.

---

## Part 4. Quality check (Maheen and Izma together)

If you are not on the same machine, Maheen zips a small piece and sends it with AnyDesk file transfer:
```bash
zip -r sample.zip data/processed/agnostic data/processed/pose_vis data/processed/cloth data/processed/parse
```
(Windows: right-click the folders, then Send to, then Compressed folder.)

Open these folders and look at about 30 images across different categories (split the categories between you).

| Folder | What good looks like | What bad looks like |
|---|---|---|
| `agnostic/` | Grey area covers the whole outfit and arms; face, hair and shoes untouched | Grey misses the dupatta or hem; grey covers the face; a leftover patch of the old garment |
| `pose_vis/` | Skeleton sits on the body (shoulders, hips) | Black image (pose not found) or skeleton far off the body |
| `cloth/` | Whole garment on white, no skin or background | Missing dupatta; arms included; big holes |
| `parse/` | Open in an image viewer; shapes follow body and clothes | Garment labelled as the wrong thing |

Write down findings by category (for example: "Lehenga: grey stops above the hem in 4 of 5").

What to change, in `params.yaml` under `preprocess`:
- Grey area too small, outfit visible around it: raise `mask_dilate_px` (25 to 35).
- Grey area too big: lower it (25 to 15).
- Cloth cutouts rough or dupatta missing: try SAM 2 (`--sam2` flag, see Part 6) and compare.

After each change, redo the test on one bad category only:
```bash
python scripts/02_preprocess/preprocess.py --category Lehenga --limit 5 --force
```
Do not start the full run until both of you agree the sample looks acceptable. Some failures are expected, so decide a threshold together, for example "at least 90 of 100 look fine".

---

## Part 5. DensePose (Izma)

Needs Linux (or WSL2), a CUDA GPU, and a clone of IDM-VTON. Use a **separate** virtual environment so it does not break the main one.

```bash
cd ~
git clone https://github.com/yisol/IDM-VTON.git
python -m venv dp-venv && source dp-venv/bin/activate
pip install torch torchvision          # CUDA build, same as in Part 2
pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install pillow tqdm numpy opencv-python
```
Then download the DensePose checkpoint `model_final_162be9.pkl` and put it at:
```
IDM-VTON/ckpt/densepose/model_final_162be9.pkl
```
(it is listed in IDM-VTON's repository and Hugging Face space; follow the download instructions in its README.)

Installing detectron2 often fails. If so, copy the full error and ask Claude or our supervisor. Do not spend more than a couple of hours on it without asking.

Izma also needs the repo and data on the machine (same steps as Part 2, with her own token), and Maheen must have run `preprocess.py` first, because DensePose reads `data/processed/image/`.

Test:
```bash
cd aks-VTON
python scripts/02_preprocess/densepose.py --idm-dir ../IDM-VTON --limit 5
```
Check `data/processed/densepose/`: each image should be a coloured body-part map matching the person. If an error says `apply_net` or a config is not found, open `IDM-VTON/gradio_demo/app.py` and compare how it calls DensePose. The script was written from that file and may need a small fix.

Full run once the test is fine:
```bash
python scripts/02_preprocess/densepose.py --idm-dir ../IDM-VTON
```

---

## Part 6. Full run and saving the results

**Maheen**, after you both approve the sample:
```bash
tmux new -s prep                    # Linux only
source venv/bin/activate
python scripts/02_preprocess/preprocess.py
```
Optional SAM 2 garment refinement (only if the cloth cutouts looked rough in Part 4):
```bash
pip install "git+https://github.com/facebookresearch/sam2.git"
# download sam2.1_hiera_large.pt into models/
python scripts/02_preprocess/preprocess.py --sam2 --force
```
Tell Izma when it finishes. Then check:
```bash
python - <<'EOF'
import pandas as pd
d = pd.read_csv("reports/preprocess_log.csv")
print(d.status.value_counts())
print(d.note.value_counts())
EOF
```
Write down how many are `ok`, `error`, `pose_not_found` and `garment_tiny`. Those are your real numbers for the paper.

**Save to DVC and Git** (after both `preprocess.py` and `densepose.py` are done, on the machine that has `data/processed`; if DensePose ran on a different copy, copy `data/processed/densepose` into Maheen's `data/processed` first):
```bash
dvc add data/processed
git add data/processed.dvc reports/preprocess_list.csv reports/preprocess_log.csv params.yaml
git commit -m "preprocessing: v1 outputs"
dvc push -r dagshub_remote
git push
```
Check your DagsHub free storage first; the outputs will be several GB.

Anyone else gets the results with:
```bash
git pull
dvc pull -r dagshub_remote
```

---

## Part 7. Troubleshooting

| Problem | What to do |
|---|---|
| `torch.cuda.is_available()` is `False` | Install the CUDA build of PyTorch (Part 2) |
| `CUDA out of memory` | Close other programs using the GPU (`nvidia-smi`), or check whether someone else is using it |
| `run scripts/02_preprocess/select_images.py first` | Run Part 3 in order |
| `FileNotFoundError` for `reports/filter_flags.csv` | `git pull`, then `dvc pull` |
| Model download fails | The machine has no internet; ask our supervisor, or download the file on a laptop and transfer it |
| `No module named sam2` | Only needed with `--sam2`: see Part 6 |
| `ModuleNotFoundError: apply_net` or detectron2 | Check Part 5 setup and that `--idm-dir` points at the IDM-VTON clone |
| AnyDesk disconnected during a run | Reconnect and run the same command again; finished images are skipped |
| Many `pose_not_found` | Normal for heavy outfits; list them, do not delete them |
| `git pull` conflicts | Stop and ask each other; do not force anything |

---

## Part 8. Checklist

- [ ] Both: DagsHub tokens set up, `dvc pull` works, AnyDesk connects, `nvidia-smi` works
- [ ] Maheen: venv works, `torch.cuda.is_available()` is `True`
- [ ] Maheen: `select_images.py` run, `--limit 5` test done
- [ ] Both: sample checked, settings agreed
- [ ] Izma: DensePose environment works on the sample
- [ ] Maheen: full `preprocess.py` run, log numbers written down
- [ ] Izma: full `densepose.py` run
- [ ] Maheen: `dvc add data/processed`, `dvc push`, `git push`
- [ ] Izma: `git pull` and `dvc pull` work