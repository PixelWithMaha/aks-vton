# Reviewing the borderline images

The filter could not decide on 144 images. Each of us checks our own categories in FiftyOne. Tags are saved to a file automatically.

| Person | Categories | Images |
|---|---|---|
| Maaham | Saree, Sharara, Gharara, Anarkali, Lehenga, Pishwas, Gown | 53 |
| Maheen | Women_Kurta_Set, Men_Kurta_Set, Kaftan, Maxi, Angrakha, Co-Ord-Sets | 41 |
| Izma | Men_Waistcoat, Men_Sherwani, Men_Shalwar_Kameez, Jacket, Peplum, Wrap | 50 |

## 1. Setup (once)

```bash
cd aks-vton
git pull
source venv/bin/activate        # Windows: venv\Scripts\activate
dvc pull -r dagshub_remote      # downloads the images into data/raw (skip if you already have them)
```

## 2. Review in FiftyOne

Use your own name: `maaham`, `maheen` or `izma`.

```bash
python scripts/01_filter/review_tool.py maheen
```

FiftyOne opens with only your images. Field `reasons` (left sidebar) shows why each was flagged.

1. Click an image to open it large. The arrow keys move to the next image.
2. Click the **tag icon** (top bar) → **Sample** tab → type `keep`, `remove` or `unsure` → **Apply**.
   Use exactly these three words. Each image gets one.
3. Faster for obvious ones: in the grid, tick several images (top-left checkbox on each), then tag icon → Apply. This tags them all at once.
4. To see what is left: sidebar → **sample tags** → shows counts; images with no tag still need doing.

Your tags are copied to `reports/review_<yourname>.csv` every 15 seconds. Stop with Ctrl+C in the terminal (or close the browser tab); the last save happens then.
Next time you run it, your earlier tags are still there.

If the terminal says an image "has two decision tags", open it, remove the wrong tag and run again.

## 3. What counts as keep

**Keep:** one real person, standing, full outfit visible from shoulders to feet, roughly facing the camera (a slight turn is fine).

**Remove:** a mannequin or dress form, sitting, a back view, a mirror selfie, the outfit cut off, a second person in front of the outfit, or a close-up or detail shot.

For Men_Waistcoat, Jacket, Peplum and Wrap, an upper-body photo is fine.

## 4. Send your file

When every image has a tag, send `reports/review_<yourname>.csv` to Maaham, or commit and push it:

```bash
git add reports/review_maheen.csv
git commit -m "review: maheen"
git push
```

Maaham then runs `python scripts/01_filter/review_tool.py --merge`.
