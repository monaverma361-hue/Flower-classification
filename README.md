# Flower Classification (ResNet50 fine-tune)

This repository contains code to fine-tune a ResNet50 model on a flower classification dataset.

Contents
- `train_finetune.py` — training script (head training + finetuning).
- `finetune_predict.py` — (inference / prediction helper).
- `cat_to_name.json` — mapping from class id to flower name used for visualization.
- `flower_data/` — dataset (train/valid/test folders). NOTE: dataset is large and is ignored by Git in this repo.
- `checkpoint_resnet_head.pth` — saved checkpoint containing the trained head (ignored in repo history).
- `checkpoint_resnet_finetune.pth` — saved final finetuned model checkpoint (ignored in repo history).
- `output/` — training plots (ignored in repo history).

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
- Training saves two checkpoints (`checkpoint_resnet_head.pth` and `checkpoint_resnet_finetune.pth`) and two plots into `output/`.

Inference
- Use `finetune_predict.py` (script included) to load a checkpoint and predict on an image. Typical usage:

```bash
python3 finetune_predict.py --image path/to/image.jpg --checkpoint checkpoint_resnet_finetune.pth --top_k 5 --category_names cat_to_name.json
```

(Adjust flags according to the script's argument parsing — open `finetune_predict.py` for exact options.)

About Git & large files
- The dataset (`flower_data/`), `output/` images, and model checkpoint files (`*.pth`) are large and have been removed from Git history and are excluded by `.gitignore` to keep the repository small and GitHub-friendly.
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
  - enable Git LFS here if you install it locally and I re-run the migration,
  - add CLI arguments to `train_finetune.py` for epochs/batch size/device paths.

---
Generated on 2026-01-07
