---
title: Plant Disease Detector
emoji: 🌱
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: mit
---

# Plant Disease Detector (Hugging Face Space)

Upload a photo of a plant leaf to detect the disease (or "healthy"). Trained on the
PlantVillage dataset (38 classes) using transfer learning with MobileNetV2.

## Deploying this Space

1. Create a new Space at https://huggingface.co/new-space → SDK: **Gradio**.
2. Upload the contents of this `app/` folder: `app.py`, `requirements.txt`, this
   `README.md`, plus the trained artifacts `plant_disease_model.pt` and
   `class_names.json` (produced by the training notebook).
3. The Space builds automatically and gives you a public URL to share.

> The two artifact files are not committed to git (they're large/binary). Copy them
> into `app/` after training before uploading to the Space.
