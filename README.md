# Flower Classification (ResNet50 fine-tune)


This repository contains code to fine-tune a ResNet50 model for classifying flowers into 50 categories. It demonstrates transfer learning, model inference, and visualizations through a Jupyter notebook interface.

⸻

Project Overview

The goal of this project is to:
	•	Fine-tune a pretrained ResNet50 on a flower dataset.
	•	Predict the class of any flower image.
	•	Provide a visual interface to view predictions and top probabilities.
	•	Showcase results for portfolio or client review (e.g., Upwork link).

⸻

Repository Contents
| File / Folder | Description |
|---------------|-------------|
| train_finetune.py | Training script (head training + fine-tuning) |
| finetune_predict.py | Prediction script for single images |
| Flower Classification Interface.ipynb | Jupyter notebook showing example predictions with images and probabilities |
| cat_to_name.json | Mapping from class id to flower name (used for visualization) |
| flower_data/ | Original dataset (train/valid/test). Not included in this repo |
| checkpoint_resnet_head.pth | Head-only trained model checkpoint. Not included |
| checkpoint_resnet_finetune.pth | Fully fine-tuned model checkpoint. Not included |
| output/ | Contains training plots and example prediction images (ignored in Git to reduce repo size) |

Environment & Dependencies

Tested environment:
	•	Python 3.8–3.11
	•	PyTorch 1.12+
	•	Torchvision 0.13+
	•	Matplotlib
	•	NumPy
	•	Pillow
