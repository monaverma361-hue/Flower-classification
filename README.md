# Flower Classification (ResNet50 fine-tune)


This repository contains code to fine-tune a ResNet50 model for classifying flowers into 50 categories. It demonstrates transfer learning, model inference, and visualizations through a Jupyter notebook interface.

⸻

Project Overview

The goal of this project is to:
- Fine-tune a pretrained ResNet50 on a flower dataset.
- Predict the class of any flower image.
- Provide a visual interface to view predictions and top probabilities.
- Showcase results for portfolio or client review (e.g., Upwork link).

⸻

Repository Contents

| File / Folder                             | Description                                                        |
|-------------------------------------------|--------------------------------------------------------------------|
| train_finetune.py                         | Training script (head training + fine-tuning)                      |
| finetune_predict.py                       | Prediction script for single images                                |
| Flower Classification Interface.ipynb     | Jupyter notebook showing example predictions with images & probs   |
| cat_to_name.json                          | Mapping from class id to flower name (used for visualization)      |
| flower_data/                              | Original dataset (train/valid/test). Not included in this repo     |
| checkpoint_resnet_head.pth                | Head-only trained model checkpoint. Not included                   |
| checkpoint_resnet_finetune.pth            | Fully fine-tuned model checkpoint. Not included                    |
| output/                                   | Contains training plots and example prediction images (ignored)    |

⸻

Environment & Dependencies

Tested environment:
- Python 3.8–3.11
- PyTorch 1.12+
- Torchvision 0.13+
- Matplotlib
- NumPy
- Pillow

Virtual environment (recommended)

Linux / macOS (bash/zsh):
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision matplotlib numpy pillow
```

Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install torch torchvision matplotlib numpy pillow
```

Alternatively, use conda:
```bash
conda create -n flower-env python=3.10
conda activate flower-env
pip install torch torchvision matplotlib numpy pillow
```

You can pin exact versions in a `requirements.txt` (example):
```
torch>=1.12
torchvision>=0.13
matplotlib
numpy
Pillow
tqdm
```
Generate from your env with:
```bash
pip freeze > requirements.txt
```

⸻

Quickstart

1. Prepare environment (see above).
2. From project root, run training (non-interactive plotting recommended on headless machines):
```bash
# example path used during development — run from your repo root instead
MPLBACKEND=Agg python3 train_finetune.py
```
- The training script will detect `mps` (Apple Silicon) when available. To force CPU, edit the device selection in the script.
- Training saves two checkpoints (`checkpoint_resnet_head.pth` and `checkpoint_resnet_finetune.pth`) and outputs plots to `output/`.

Inference
- Example:
```bash
python3 finetune_predict.py --image path/to/image.jpg --checkpoint checkpoint_resnet_finetune.pth --top_k 5 --category_names cat_to_name.json
```
- For batch predictions and saving results (if script supports it):
```bash
python3 finetune_predict.py --image samples/* --checkpoint checkpoint_resnet_finetune.pth --top_k 5 --category_names cat_to_name.json --save_results output/predictions.json
```

Results display (view-only)
- Add annotated images and `output/predictions.json` to `output/gallery/` or create `results/index.html` for GitHub Pages.
- If files are large, host externally and link from the README.

About Git & large files
- Large items (dataset, outputs, checkpoints) are excluded from git in this repo. Use Git LFS to track `*.pth` if you want to version checkpoints:
```bash
# example (macOS)
brew install git-lfs
git lfs install
git lfs track "*.pth"
git add .gitattributes
git add path/to/*.pth
git commit -m "chore: track checkpoints with git-lfs"
git push origin main
```

Notes & suggestions
- Keep code and scripts in Git; store datasets and heavy artifacts in remote storage (Drive, S3) or use Git LFS.
- Consider adding `requirements.txt` or `pyproject.toml` for reproducibility.
- Consider adding `output/gallery/`, `output/predictions.json`, and `results/index.html` for a GitHub Pages demo.

Contact / next steps
If you'd like, I can:
- generate a `requirements.txt` with pinned versions,
- create a `make_gallery.py` script to produce annotated thumbnails and `predictions.json`,
- or create a static `results/index.html` for GitHub Pages.
