# Plant Disease Detector — Design Document

**Date:** 2026-06-03
**Type:** Full-semester ML portfolio project
**Author:** (your name)

## 1. Goal

Build an image classifier that identifies plant leaf diseases from a photo, deployed
as a live web app. The project demonstrates an end-to-end ML workflow — data,
training, evaluation, and deployment — and serves as a portfolio piece with a
publicly shareable demo.

## 2. Problem Statement

Given a photo of a plant leaf, predict whether the plant is healthy or which disease
it has. Early, accessible disease detection helps farmers and gardeners act before
crops are lost. The app makes this as simple as uploading a photo.

## 3. Dataset

- **PlantVillage** — ~54,000 labeled leaf images, 38 classes across 14 crop species
  (each crop has a "healthy" class plus one or more disease classes).
- Images are clean, well-lit, and consistently labeled — ideal for transfer learning.
- Source: publicly available (Kaggle / TensorFlow Datasets).
- Split: 80% train / 10% validation / 10% test (stratified by class).

## 4. Modeling Approach — Transfer Learning

- Start from a CNN pre-trained on ImageNet (**MobileNetV2** primary; ResNet18 as a
  fallback/comparison). These are small and train fast on a free Colab GPU.
- Freeze the convolutional backbone initially, replace the classifier head with a new
  38-way output layer, and train the head.
- Optionally unfreeze the top layers and fine-tune at a low learning rate for a few
  extra epochs.
- This counts as training your own model while being achievable for a beginner.

### Training details
- Input size: 224×224, normalized with ImageNet mean/std.
- Data augmentation: random flips, rotation, color jitter (train split only).
- Loss: cross-entropy. Optimizer: Adam. LR scheduler: step/plateau.
- Target: ≥95% validation accuracy (transfer learning on PlantVillage reaches this
  comfortably).

## 5. Evaluation

- Overall accuracy on the held-out test set.
- Per-class accuracy + confusion matrix to find weak classes.
- A handful of qualitative predictions on images outside the dataset (real photos).
- Saved as plots + a short report.

## 6. Application

- **Gradio** web app: drag-and-drop / upload a leaf image → top prediction + confidence,
  and top-3 class probabilities.
- Loads the exported trained model (`models/plant_disease_model.pt`) and the class-name
  list.

## 7. Deployment

- Hosted on **Hugging Face Spaces** (free), giving a public URL to share on a resume.
- `app/` contains everything the Space needs: `app.py`, `requirements.txt`, the model
  file, and `class_names.json`.

## 8. Tech Stack

| Layer | Choice |
|---|---|
| Language | Python |
| Training | PyTorch + torchvision |
| Environment | Google Colab (free GPU) |
| App | Gradio |
| Hosting | Hugging Face Spaces |
| Repo | GitHub |

## 9. Components & Boundaries

- **Training notebook** (`notebooks/train_plant_disease.ipynb`) — self-contained:
  download data → build loaders → train → evaluate → export model + class names.
  Output: `plant_disease_model.pt`, `class_names.json`, evaluation plots.
- **Inference app** (`app/app.py`) — loads model + class names, exposes a Gradio UI.
  Depends only on the exported artifacts; no training code.
- **Artifacts** — model weights and class-name list are the contract between the two.

## 10. Semester Timeline

| Phase | Weeks | Milestone |
|---|---|---|
| Setup & data | 1–2 | Data loaded, EDA notebook |
| Baseline model | 3–4 | Baseline accuracy to beat |
| Transfer learning | 5–7 | Trained model ≥95% |
| Evaluation | 8–9 | Confusion matrix + report |
| Web app | 10–11 | Live demo URL |
| Polish | 12–13 | Portfolio-ready repo |
| Report & buffer | 14–16 | Final submission + slides |

## 11. Deliverables

1. Live Gradio demo on Hugging Face Spaces.
2. Clean GitHub repo (training notebook, app, README with results).
3. Evaluation report (accuracy, confusion matrix, examples).
4. Academic report / slides.

## 12. Stretch Goals (only if ahead)

- Grad-CAM heatmaps showing where the model looks.
- Top-3 prediction view (basic version included by default).
- Mobile-friendly UI.

## 13. Scope Guardrails

- Stick to PlantVillage; don't mix datasets early.
- One architecture done well beats five half-trained ones.
- Get a working end-to-end pipeline first, optimize later.
