# Deployment Guide

Two publishing steps remain. Both require logging into *your* accounts, so you run these.
Copy-paste each block from the project folder.

---

## A. Push the code to GitHub

### 1. Create an empty repo
Go to https://github.com/new → name it `plant-disease-detector` → **Create repository**
(do NOT add a README/license; the repo already has them).

### 2. Connect and push
From the project folder, run (replace `YOUR_USERNAME`):

```bash
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/plant-disease-detector.git
git push -u origin main
```

If prompted for a password, use a **GitHub Personal Access Token** (github.com →
Settings → Developer settings → Personal access tokens), not your account password.

✅ Your code, notebook, report, and confusion matrix are now on GitHub.
(The model file `.pt` is gitignored — that's intentional; it goes to Hugging Face below.)

---

## B. Deploy the live demo to Hugging Face Spaces

### Option 1 — Web upload (easiest, no terminal)
1. Go to https://huggingface.co/new-space → name it, **SDK: Gradio**, **Create Space**.
2. Click **Files → Add file → Upload files** and upload everything from the `app/` folder:
   - `app.py`
   - `requirements.txt`
   - `README.md`
   - `class_names.json`
   - `plant_disease_model.pt`  ← the trained model
3. Wait ~2–3 min for the build. You get a public URL.

### Option 2 — Command line (huggingface_hub is already installed)
```bash
# 1. Log in (paste a token from huggingface.co/settings/tokens, with WRITE access)
huggingface-cli login

# 2. Create the Space (replace YOUR_USERNAME)
huggingface-cli repo create plant-disease-detector --type space --space_sdk gradio -y

# 3. Upload the app folder contents to the Space
huggingface-cli upload YOUR_USERNAME/plant-disease-detector app/ . --repo-type space
```

✅ Your Space builds automatically and gives you a public demo URL.

---

## C. Final touches
Accuracy numbers are already filled in (test 96.21% / val 96.41%). After deploying,
just add your Space URL:

1. Paste the Space URL into:
   - `README.md` (the "Live demo" line at top)
   - `REPORT.md` (replace `{{HF_SPACE_URL}}`)
   - `SLIDES.md` (slide 8)
2. Commit and push the update:
   ```bash
   git add -A && git commit -m "Add live demo URL" && git push
   ```

🎉 Done — full project published: GitHub repo + live demo + report + slides.
