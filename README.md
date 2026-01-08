# Flower Classification (ResNet50 fine-tune)

This repository contains code to fine-tune a ResNet50 model on a flower classification dataset.

Contents
- `train_finetune.py` — training script (head training + finetuning).
- `finetune_predict.py` — (inference / prediction helper).
- `cat_to_name.json` — mapping from class id to flower name used for visualization.
- `flower_data/` — dataset (train/valid/test folders). NOTE: dataset is large and is ignored by Git in this repo.
- `checkpoint_resnet_head.pth` — saved checkpoint containing the trained head (ignored in repo history).
- `checkpoint_resnet_finetune.pth` — saved final finetuned model checkpoint (ignored in repo history).
- `output/` — training plots and example predictions (ignored in repo history).

Quickstart
1. Create a virtual environment and install required packages (example):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision matplotlib numpy pillow
```

2. Run training (non-interactive plotting):

```bash
# from project root
cd "/Users/monaverma/Documents/image classifiers/finetune_resnet"
MPLBACKEND=Agg python3 train_finetune.py
```

- The script will detect `mps` when available (macOS Apple Silicon). To force CPU, you can edit the device selection in the script.
- Training saves two checkpoints (`checkpoint_resnet_head.pth` and `checkpoint_resnet_finetune.pth`) and plots into `output/`.

Inference
- Use `finetune_predict.py` (script included) to load a checkpoint and predict on an image. Typical usage:

```bash
python3 finetune_predict.py --image path/to/image.jpg --checkpoint checkpoint_resnet_finetune.pth --top_k 5 --category_names cat_to_name.json
```

(Adjust flags according to the script's argument parsing — open `finetune_predict.py` for exact options.)

Results display (view-only) — show outputs for others to see
----------------------------------------------------------

Overview
- If your goal is to *show* results (visualizations, sample predictions, plots) to others without providing an upload/prediction interface — for example because the dataset is small or private — you can provide a view-only results section. This makes it easy for anyone to inspect the model outputs without running the model or uploading images.

What a view-only results display can include
- Example input images from the dataset with the model's predicted top-K labels and probabilities rendered next to each image.
- Training/validation loss & accuracy plots.
- A small table or JSON file with the raw predictions for each example (class ids, readable names via `cat_to_name.json`, probability).
- A single static HTML page or a README gallery showing thumbnails and predictions.

Where to put results
- `output/` — add sample figures and example predictions here (note: large files are ignored by git in this repo by default).
- `docs/` or a `results/` folder — useful if you want to publish via GitHub Pages.
- If files are too large, host images externally (Google Drive, S3) and link to them from the README.

Simple approaches to create a view-only gallery
1. Save prediction images and a JSON with labels
- Use `finetune_predict.py` to run predictions locally on a handful of representative images and save:
  - Annotated image (image + overlayed prediction text) or a thumbnail.
  - A JSON file with top-K predictions per image: { "image1.jpg": [{"class":"3","name":"rose","prob":0.72}, ...], ... }.

2. Add a static results HTML page
- Create `results/index.html` that shows thumbnails and predictions — this can be served via GitHub Pages (no server code required).

3. Show results directly in the README
- Add a small gallery section with images and captions. Example Markdown for one image:

```markdown
### Example predictions

![test1](output/test1_pred.jpg)

test1.jpg — top predictions:
- rose: 72.0%
- daisy: 15.3%
- sunflower: 4.7%
```

4. Publish via GitHub Pages
- If you want anyone on the web to see the gallery, enable GitHub Pages for the repo and put the static HTML in `docs/` or the `gh-pages` branch.

Minimal example: save predictions as JSON (pseudo-commands)
- Run predictions locally and save outputs:

```bash
python3 finetune_predict.py --image samples/* --checkpoint checkpoint_resnet_finetune.pth --top_k 5 --category_names cat_to_name.json --save_results output/predictions.json
```

- (Add a small wrapper in your repo to batch-predict a chosen set of images and save annotated thumbnails to `output/gallery/`.)

Security & privacy
- Because this is view-only, you avoid handling arbitrary uploads.
- If you include dataset images in the repo or on Pages, ensure you have the right to share them.

Why choose view-only
- Keeps the repo simple and avoids running a hosted service.
- Good when dataset is small, proprietary, or you only want to demonstrate results.
- Works well for academic reports, portfolio pages, or README demos.

What to add to this repo (optional suggestions)
- `output/gallery/` — sample annotated images and README gallery entries.
- `output/predictions.json` — JSON of example predictions.
- `results/index.html` — static page for GitHub Pages.
- A tiny helper script `make_gallery.py` to run predictions for a chosen set of images and save annotated thumbnails (I can generate this for you if you want).

About Git & large files
- The dataset (`flower_data/`), `output/` images, and model checkpoint files (`*.pth`) are large and have been removed from Git history and are excluded by `.gitignore` to keep the repository small and fast to clone.
- If you want to version checkpoints in the repo, enable Git LFS and track `*.pth`:

```bash
# install git-lfs (macOS example via Homebrew)
brew install git-lfs
# in repo
git lfs install
git lfs track "*.pth"
git add .gitattributes
# re-add and commit large files so they become LFS objects
git add path/to/*.pth
git commit -m "chore: track checkpoints with git-lfs"
git push origin main
```

Notes & suggestions
- Recommended workflow: keep code and small scripts in Git, store datasets and heavy artifacts in remote storage (Google Drive, S3) or use Git LFS for model checkpoints.
- Consider adding a `requirements.txt` or `pyproject.toml` for reproducibility.

Contact / next steps
- If you want, I can:
  - generate `requirements.txt` with your environment's versions,
  - create a small `make_gallery.py` script that produces annotated thumbnails and a `predictions.json`,
  - or create a static `results/index.html` you can publish on GitHub Pages.

---
Generated on 2026-01-08
