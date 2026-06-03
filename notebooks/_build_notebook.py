"""Generates train_plant_disease.ipynb (Colab-ready). Run once: python _build_notebook.py"""
import json, os

def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": [l if l.endswith("\n") else l + "\n" for l in lines]}

def code(*lines):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": [l if l.endswith("\n") else l + "\n" for l in lines]}

cells = []

cells.append(md(
    "# 🌱 Plant Disease Detector — Training Notebook",
    "",
    "Fine-tunes a pre-trained **MobileNetV2** on the **PlantVillage** dataset (38 classes) to",
    "classify plant leaf diseases from photos.",
    "",
    "**How to use this notebook:**",
    "1. Open it in [Google Colab](https://colab.research.google.com/).",
    "2. Set the runtime to GPU: **Runtime → Change runtime type → T4 GPU**.",
    "3. Run the cells top to bottom. The data download needs your free Kaggle API token (instructions below).",
    "4. At the end you'll download `plant_disease_model.pt` and `class_names.json` — put both in the `app/` folder of your repo to deploy.",
    "",
    "Total run time on a free T4 GPU: roughly **20–40 minutes** for 5 epochs.",
))

cells.append(md("## Step 1 — Check the GPU and import libraries"))
cells.append(code(
    "import torch, torchvision",
    "print('PyTorch:', torch.__version__)",
    "print('CUDA available:', torch.cuda.is_available())",
    "if torch.cuda.is_available():",
    "    print('GPU:', torch.cuda.get_device_name(0))",
    "else:",
    "    print('WARNING: No GPU detected. Go to Runtime > Change runtime type > T4 GPU, then re-run.')",
    "",
    "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')",
))
cells.append(code(
    "import os, json, time, random",
    "import numpy as np",
    "import matplotlib.pyplot as plt",
    "import torch.nn as nn",
    "from torch.utils.data import DataLoader, random_split",
    "from torchvision import datasets, transforms, models",
    "",
    "SEED = 42",
    "random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)",
))

cells.append(md(
    "## Step 2 — Download the PlantVillage dataset (via Kaggle)",
    "",
    "**You need two things from Kaggle:**",
    "1. **Username** — shown in your profile URL: `kaggle.com/YOUR_USERNAME`.",
    "2. **API key/token** — go to [kaggle.com](https://www.kaggle.com/) → profile → **Settings**",
    "   → **API** → **Create New Token**. This is the secret key string.",
    "",
    "The cell below asks for both (the key is typed in hidden) and sets them as",
    "environment variables — no `kaggle.json` file required.",
))
cells.append(code(
    "# Enter your Kaggle credentials (key input is hidden)",
    "import getpass",
    "os.environ['KAGGLE_USERNAME'] = input('Kaggle username: ').strip()",
    "os.environ['KAGGLE_KEY'] = getpass.getpass('Kaggle API key (hidden): ').strip()",
    "!pip install -q kaggle",
    "print('Credentials set for user:', os.environ['KAGGLE_USERNAME'])",
))
cells.append(code(
    "# Download + unzip the PlantVillage dataset (~2 GB). Takes a few minutes.",
    "!kaggle datasets download -d abdallahalidev/plantvillage-dataset -q",
    "!unzip -q -o plantvillage-dataset.zip -d data",
    "",
    "# The 'color' folder holds 38 class subfolders of RGB leaf images.",
    "DATA_DIR = 'data/plantvillage dataset/color'",
    "print('Classes found:', len(os.listdir(DATA_DIR)))",
))

cells.append(md(
    "## Step 3 — Build data loaders",
    "",
    "We resize to 224×224, normalize with ImageNet statistics, and augment the training",
    "images (flips, rotation, color jitter) so the model generalizes better.",
))
cells.append(code(
    "IMG_SIZE = 224",
    "BATCH_SIZE = 32",
    "MEAN = [0.485, 0.456, 0.406]",
    "STD  = [0.229, 0.224, 0.225]",
    "",
    "train_tf = transforms.Compose([",
    "    transforms.Resize((IMG_SIZE, IMG_SIZE)),",
    "    transforms.RandomHorizontalFlip(),",
    "    transforms.RandomRotation(20),",
    "    transforms.ColorJitter(0.2, 0.2, 0.2),",
    "    transforms.ToTensor(),",
    "    transforms.Normalize(MEAN, STD),",
    "])",
    "eval_tf = transforms.Compose([",
    "    transforms.Resize((IMG_SIZE, IMG_SIZE)),",
    "    transforms.ToTensor(),",
    "    transforms.Normalize(MEAN, STD),",
    "])",
    "",
    "# Load once to grab class names + indices, then split 80/10/10.",
    "full = datasets.ImageFolder(DATA_DIR)",
    "class_names = full.classes",
    "num_classes = len(class_names)",
    "print('Num classes:', num_classes)",
    "",
    "n = len(full)",
    "n_train = int(0.8 * n); n_val = int(0.1 * n); n_test = n - n_train - n_val",
    "g = torch.Generator().manual_seed(SEED)",
    "train_ds, val_ds, test_ds = random_split(full, [n_train, n_val, n_test], generator=g)",
    "",
    "# Apply the right transform to each split (wrap so transforms differ per split).",
    "class TfSubset(torch.utils.data.Dataset):",
    "    def __init__(self, subset, tf): self.subset, self.tf = subset, tf",
    "    def __len__(self): return len(self.subset)",
    "    def __getitem__(self, i):",
    "        img, label = self.subset[i]",
    "        return self.tf(img), label",
    "",
    "train_loader = DataLoader(TfSubset(train_ds, train_tf), batch_size=BATCH_SIZE, shuffle=True, num_workers=2)",
    "val_loader   = DataLoader(TfSubset(val_ds, eval_tf), batch_size=BATCH_SIZE, shuffle=False, num_workers=2)",
    "test_loader  = DataLoader(TfSubset(test_ds, eval_tf), batch_size=BATCH_SIZE, shuffle=False, num_workers=2)",
    "print(f'Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}')",
))

cells.append(md("## Step 3b — Quick look at the data (EDA)"))
cells.append(code(
    "# Show a few sample images with their labels",
    "def denorm(t):",
    "    t = t.clone()",
    "    for c in range(3): t[c] = t[c]*STD[c] + MEAN[c]",
    "    return t.clamp(0,1).permute(1,2,0).numpy()",
    "",
    "imgs, labels = next(iter(train_loader))",
    "plt.figure(figsize=(12,6))",
    "for i in range(8):",
    "    plt.subplot(2,4,i+1)",
    "    plt.imshow(denorm(imgs[i]))",
    "    plt.title(class_names[labels[i]][:20], fontsize=8)",
    "    plt.axis('off')",
    "plt.tight_layout(); plt.show()",
))

cells.append(md(
    "## Step 4 — Build the model (transfer learning)",
    "",
    "Load MobileNetV2 pre-trained on ImageNet, freeze its feature extractor, and replace",
    "the final classifier with a fresh 38-class layer that we'll train.",
))
cells.append(code(
    "model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)",
    "for p in model.features.parameters():",
    "    p.requires_grad = False  # freeze backbone",
    "",
    "model.classifier[1] = nn.Linear(model.last_channel, num_classes)  # new head",
    "model = model.to(device)",
    "",
    "criterion = nn.CrossEntropyLoss()",
    "optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)",
    "scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)",
))

cells.append(md("## Step 5 — Train"))
cells.append(code(
    "EPOCHS = 5  # bump to 8-10 for a bit more accuracy",
    "",
    "def run_epoch(loader, train=True):",
    "    model.train() if train else model.eval()",
    "    total, correct, loss_sum = 0, 0, 0.0",
    "    with torch.set_grad_enabled(train):",
    "        for imgs, labels in loader:",
    "            imgs, labels = imgs.to(device), labels.to(device)",
    "            if train: optimizer.zero_grad()",
    "            out = model(imgs)",
    "            loss = criterion(out, labels)",
    "            if train:",
    "                loss.backward(); optimizer.step()",
    "            loss_sum += loss.item() * imgs.size(0)",
    "            correct += (out.argmax(1) == labels).sum().item()",
    "            total += imgs.size(0)",
    "    return loss_sum/total, correct/total",
    "",
    "history = {'train_acc': [], 'val_acc': []}",
    "best_val = 0.0",
    "for epoch in range(EPOCHS):",
    "    t0 = time.time()",
    "    tr_loss, tr_acc = run_epoch(train_loader, True)",
    "    va_loss, va_acc = run_epoch(val_loader, False)",
    "    scheduler.step()",
    "    history['train_acc'].append(tr_acc); history['val_acc'].append(va_acc)",
    "    print(f'Epoch {epoch+1}/{EPOCHS}  train_acc={tr_acc:.3f}  val_acc={va_acc:.3f}  ({time.time()-t0:.0f}s)')",
    "    if va_acc > best_val:",
    "        best_val = va_acc",
    "        torch.save(model.state_dict(), 'plant_disease_model.pt')",
    "        print(f'  ✓ saved new best model (val_acc={va_acc:.3f})')",
    "print('Best validation accuracy:', round(best_val, 4))",
))
cells.append(code(
    "# Plot the learning curves",
    "plt.plot(history['train_acc'], label='train')",
    "plt.plot(history['val_acc'], label='val')",
    "plt.xlabel('epoch'); plt.ylabel('accuracy'); plt.legend(); plt.title('Accuracy'); plt.show()",
))

cells.append(md(
    "## Step 6 — Evaluate on the test set",
    "",
    "Load the best saved model and report overall test accuracy plus a confusion matrix.",
))
cells.append(code(
    "model.load_state_dict(torch.load('plant_disease_model.pt', map_location=device))",
    "model.eval()",
    "",
    "all_preds, all_labels = [], []",
    "with torch.no_grad():",
    "    for imgs, labels in test_loader:",
    "        imgs = imgs.to(device)",
    "        preds = model(imgs).argmax(1).cpu().numpy()",
    "        all_preds.extend(preds); all_labels.extend(labels.numpy())",
    "",
    "all_preds = np.array(all_preds); all_labels = np.array(all_labels)",
    "test_acc = (all_preds == all_labels).mean()",
    "print('Test accuracy:', round(float(test_acc), 4))",
))
cells.append(code(
    "from sklearn.metrics import confusion_matrix, classification_report",
    "cm = confusion_matrix(all_labels, all_preds)",
    "plt.figure(figsize=(12,10))",
    "plt.imshow(cm, cmap='Blues')",
    "plt.title('Confusion Matrix'); plt.xlabel('Predicted'); plt.ylabel('True')",
    "plt.colorbar(); plt.tight_layout(); plt.savefig('confusion_matrix.png', dpi=120); plt.show()",
    "",
    "print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))",
))

cells.append(md(
    "## Step 7 — Export the model + class names",
    "",
    "Download both files and drop them into your repo's `app/` folder to deploy the demo.",
))
cells.append(code(
    "with open('class_names.json', 'w') as f:",
    "    json.dump(class_names, f, indent=2)",
    "print('Saved class_names.json with', len(class_names), 'classes')",
    "",
    "from google.colab import files",
    "files.download('plant_disease_model.pt')",
    "files.download('class_names.json')",
    "files.download('confusion_matrix.png')",
))

cells.append(md(
    "## ✅ Done!",
    "",
    "Next: put `plant_disease_model.pt` and `class_names.json` in your repo's `app/` folder,",
    "then run `python app/app.py` locally or deploy `app/` to Hugging Face Spaces.",
    "See the project **README** for deployment steps.",
))

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out = os.path.join(os.path.dirname(__file__), "train_plant_disease.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("Wrote", out, "with", len(cells), "cells")
