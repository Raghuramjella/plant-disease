"""
Plant Disease Detector — Gradio web app.

Loads the MobileNetV2 model trained in notebooks/train_plant_disease.ipynb and serves
an upload-a-leaf-photo -> prediction UI.

Required files in this folder (produced by the training notebook):
  - plant_disease_model.pt   (trained weights)
  - class_names.json         (list of 38 class names, in training order)

Run locally:   python app/app.py
Deploy:        push this app/ folder to a Hugging Face Space (SDK: Gradio).
"""
import json
import os

import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "plant_disease_model.pt")
CLASSES_PATH = os.path.join(HERE, "class_names.json")

IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_class_names():
    if not os.path.exists(CLASSES_PATH):
        raise FileNotFoundError(
            f"{CLASSES_PATH} not found. Run the training notebook and place "
            "class_names.json in the app/ folder."
        )
    with open(CLASSES_PATH) as f:
        return json.load(f)


def load_model(num_classes):
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"{MODEL_PATH} not found. Run the training notebook and place "
            "plant_disease_model.pt in the app/ folder."
        )
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval().to(device)
    return model


class_names = load_class_names()
model = load_model(len(class_names))

transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]
)


def pretty(name: str) -> str:
    """'Tomato___Late_blight' -> 'Tomato — Late blight'."""
    crop, _, disease = name.partition("___")
    return f"{crop.replace('_', ' ')} — {disease.replace('_', ' ') or 'healthy'}"


def predict(image):
    if image is None:
        return {}
    x = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu()
    # Gradio Label component renders this dict as a ranked bar chart (top 3).
    return {pretty(class_names[i]): float(probs[i]) for i in range(len(class_names))}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload a leaf photo"),
    outputs=gr.Label(num_top_classes=3, label="Prediction"),
    title="🌱 Plant Disease Detector",
    description=(
        "Upload a photo of a plant leaf and the model predicts the disease "
        "(or healthy). Trained on the PlantVillage dataset (38 classes) using "
        "transfer learning with MobileNetV2."
    ),
    article=(
        "Built as a machine learning portfolio project. "
        "Model: MobileNetV2 fine-tuned on PlantVillage. "
        "For best results use a clear, well-lit photo of a single leaf."
    ),
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()
