# 🌱 Plant Disease Detector

An end-to-end deep learning project that identifies **plant leaf diseases from a photo**.
Upload a picture of a leaf and the model predicts whether the plant is healthy or which
of 38 diseases it has — served as a live, shareable web app.

> **Live demo:** https://huggingface.co/spaces/Raghuram04/plant-disease
> **Tech:** PyTorch · MobileNetV2 (transfer learning) · Gradio · Hugging Face Spaces

---

## 📋 Overview

| | |
|---|---|
| **Problem** | Classify plant leaf images into healthy / diseased categories |
| **Dataset** | [PlantVillage](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) — ~54,000 images, 38 classes, 14 crops |
| **Model** | MobileNetV2 pre-trained on ImageNet, fine-tuned on PlantVillage |
| **Target accuracy** | ≥ 95% on the held-out test set |
| **Deployment** | Gradio app on Hugging Face Spaces (free) |

## 🏗️ How it works

```
Leaf photo ──► resize 224×224 + normalize ──► MobileNetV2 ──► 38-class softmax ──► top-3 predictions
```

Transfer learning: instead of training a CNN from scratch (which needs huge data and
compute), we start from a network already trained on millions of images, freeze its
feature extractor, and train a new 38-class head on plant leaves. This reaches high
accuracy quickly on a free GPU.

## 📂 Project structure

```
.
├── notebooks/
│   └── train_plant_disease.ipynb   # Colab-ready: download → train → evaluate → export
├── app/
│   ├── app.py                      # Gradio web app (upload leaf → prediction)
│   ├── requirements.txt            # deps for Hugging Face Space
│   └── README.md                   # Space config + deploy notes
├── docs/superpowers/specs/         # design document
├── models/                         # trained model lands here (gitignored)
├── requirements.txt                # local dev deps
└── README.md
```

## 🚀 Getting started

### 1. Train the model (Google Colab — free GPU)

1. Open `notebooks/train_plant_disease.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Set **Runtime → Change runtime type → T4 GPU**.
3. Run all cells. You'll need a free Kaggle API token (`kaggle.json`) to download the
   data — the notebook explains how to get it.
4. At the end, download **`plant_disease_model.pt`** and **`class_names.json`**.

Training ~5 epochs takes roughly 20–40 minutes on the free T4 GPU.

### 2. Run the app locally

```bash
# put plant_disease_model.pt and class_names.json into the app/ folder first
pip install -r requirements.txt
python app/app.py
```

Then open the local URL Gradio prints.

### 3. Deploy the live demo (Hugging Face Spaces)

1. Create a Gradio Space at https://huggingface.co/new-space.
2. Upload the `app/` folder contents **plus** the two trained artifacts.
3. Copy the public Space URL back into the **Live demo** line at the top of this README.

## 📊 Results

- **Test accuracy: 96.21%** (held-out test set)
- **Best validation accuracy: 96.41%**
- Trained in 5 epochs (~4 min/epoch) on a free Colab T4 GPU

| Epoch | Train acc | Val acc |
|---|---|---|
| 1 | 0.867 | 0.938 |
| 2 | 0.925 | 0.952 |
| 3 | 0.937 | 0.958 |
| 4 | 0.941 | 0.963 |
| 5 | 0.944 | 0.964 |

**Confusion matrix:**

![Confusion Matrix](app/confusion_matrix.png)

> _Tip: add a screenshot of the app making a prediction here too._

## 🔭 Possible extensions

- Grad-CAM heatmaps to visualize where the model "looks"
- Larger backbone (ResNet50/EfficientNet) comparison
- Mobile-friendly UI

## 📝 License

MIT

---

_Machine learning course project — full semester. See `docs/superpowers/specs/` for the
design document._
