# Plant Disease Detector — Presentation Outline

A ~10-slide deck for your project presentation. Each slide lists the title and the key
points / what to say. Drop your accuracy numbers and a demo screenshot where marked.

---

### Slide 1 — Title
- **Plant Disease Detection Using Deep Learning**
- Your name, course, date
- One leaf image as the background

### Slide 2 — The Problem
- Plant diseases destroy crops and threaten food security
- Expert diagnosis is scarce, especially for small farmers
- **Idea:** diagnose disease from a single phone photo

### Slide 3 — Goal
- Classify a leaf photo into healthy / disease (38 classes, 14 crops)
- Target ≥ 95% accuracy
- Deliver a usable web app, not just a notebook

### Slide 4 — Dataset
- PlantVillage: ~54,000 labeled leaf images, 38 classes
- Show a grid of sample images (from the notebook EDA cell)
- Split: 80/10/10 train/val/test

### Slide 5 — Approach: Transfer Learning
- Training from scratch = too much data/compute
- Reuse MobileNetV2 pre-trained on ImageNet
- Freeze backbone, train a new 38-class head
- Diagram: photo → MobileNetV2 → 38-class softmax → prediction

### Slide 6 — Training
- Adam optimizer, cross-entropy loss, 5 epochs, T4 GPU (free Colab)
- Data augmentation: flips, rotation, color jitter
- Show the accuracy learning-curve plot

### Slide 7 — Results
- **Test accuracy: 96.21%** | Best val: 96.41%
- Show the confusion matrix (mostly diagonal = good)
- Note where it confuses similar diseases

### Slide 8 — Live Demo
- Screenshot of the Gradio app with a real prediction
- **Demo it live if possible** (have the local app or HF Space open)
- Link: https://huggingface.co/spaces/Raghuram04/plant-disease

### Slide 9 — Limitations & Future Work
- Lab images vs. messy field photos
- Future: Grad-CAM explainability, field fine-tuning, mobile app

### Slide 10 — Conclusion
- Built a full pipeline: data → model → deployed app
- High accuracy with modest resources via transfer learning
- Thank you / questions

---

**Tip:** keep text minimal on slides; the screenshots, the confusion matrix, and a live
demo carry the presentation.
