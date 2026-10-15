# Malaria Cell Classifier: Implementation Plan

*Made with ❤️ by Calli*

> **Problem statement (Biocon Ltd.):** Biocon is working on automated diagnosis of blood samples. Develop a CNN model to classify red blood cell images as healthy or malaria-infected, supporting faster and more accurate medical diagnostics.

---

## 0. How to use this file

**For Calli:** read Sections 1 to 4 once before starting. They explain the problem, the workflow and every tool in plain words. Then work through the phases in Section 6, one at a time.

**For the coding assistant:** this is the source of truth for scope and order. Build **one phase at a time** and stop for review after each.

**The deal:** the coding assistant writes the code. Calli reads it, runs it on Kaggle, and must be able to answer the "Explain it back" questions at the end of every phase before moving on. If he can't, the assistant explains again. That's what keeps this from being vibe coding.

---

## 1. The problem in plain words

- **Input:** a small colour photo of **one** red blood cell, taken under a microscope from a stained blood smear.
- **Output:** one of two labels: **Parasitized** (malaria-infected) or **Uninfected** (healthy).
- **Type of ML task:** *binary image classification*.
- **Why a CNN:** a Convolutional Neural Network learns to spot visual patterns by itself. Infected cells usually show a dark purple stained blob (the parasite) inside the cell. We never tell the model "look for the blob". It figures that out from thousands of labelled examples.

### Why this is a *medical* problem, not just an accuracy contest

The model can be wrong in two ways:

| Mistake | Meaning | How bad? |
|---|---|---|
| **False negative** | Infected cell called healthy | **Very bad.** A sick patient could go untreated. |
| **False positive** | Healthy cell called infected | Annoying. A human re-checks it. |

So besides accuracy we care most about **recall** (also called **sensitivity**): *out of all truly infected cells, how many did we catch?* The plan includes a step (Phase 5) where we deliberately tune the model to catch more infected cells, even if it costs a few extra false alarms.

---

## 2. The dataset

**NIH Malaria Cell Images**, hosted on Kaggle as `iarunava/cell-images-for-detecting-malaria`.

- About 27,558 images, roughly half **Parasitized** and half **Uninfected**.
- RGB `.png` files, **sizes vary** (roughly 46×46 up to ~385×395), so we resize everything.
- Folder layout: one folder per class. That folder name *is* the label.

### Known gotchas (the code must handle these)

1. **Nested duplicate folder:** some versions of the Kaggle upload contain the data twice (`cell_images/` and `cell_images/cell_images/`). Find images robustly and **remove duplicates**.
2. **Junk files:** non-image files like `Thumbs.db` sit inside the folders. Only load `.png`.
3. **Patient leakage:** many cells come from the **same patient's slide**. If cells from one patient land in both training and test sets, the model can "recognise the slide" instead of learning malaria, and the test score becomes fake-good. The filenames contain a slide/patient code, so we **split by patient**, not randomly. (Phase 2 verifies the filename pattern before trusting it.)

---

## 3. The workflow (where everything runs)

You will not train on your laptop. Kaggle gives a free GPU and already hosts the dataset.

```
┌──────────────────────┐   git push   ┌──────────┐   git clone   ┌──────────────────────────┐
│ Your laptop          │ ───────────► │  GitHub  │ ────────────► │ Kaggle Notebook (GPU)    │
│ AI assistant writes  │              │  (repo = │               │ dataset attached         │
│ code, runs quick     │ ◄─────────── │  your    │ ◄──────────── │ runs real training,      │
│ smoke tests on CPU   │   git pull   │  final   │  you download │ saves plots + metrics    │
└──────────────────────┘              │  submission)│  results &  └──────────────────────────┘
                                      └──────────┘  commit them
```

1. The coding assistant writes the code on your laptop and tests it on a **tiny fake dataset** (a "smoke test") so we know it runs without errors.
2. You push to GitHub.
3. On Kaggle, a short notebook clones your repo and runs the real training on the GPU with the real data.
4. You download the results (plots, metrics, model weights) and commit the small ones to the repo.

---

## 4. Technologies (and why each one is here)

| Tool | What it is, in one line | Why we need it |
|---|---|---|
| **Python 3.10+** | The programming language of ML | Everything is written in it |
| **PyTorch** | Deep learning library: tensors, layers, training | Builds and trains the CNN. Chosen because the training loop is written out explicitly, so you can read every step |
| **torchvision** | PyTorch's add-on for images | Image transforms (resize, flip, rotate, normalise) |
| **NumPy** | Fast number arrays | Basic maths on image data |
| **pandas** | Spreadsheet-like tables in Python | Keeps the list of files, labels and patient IDs tidy |
| **Pillow (PIL)** | Opens image files | Loads each `.png` |
| **scikit-learn** | Classic ML toolkit | Patient-grouped data splitting and evaluation metrics |
| **Matplotlib** | Plotting library | Training curves, confusion matrix, sample grids |
| **tqdm** | Progress bars | So you can see training progress |
| **grad-cam** (`pytorch-grad-cam`) | Explainability library | Heatmaps showing *where* in the cell the model looked |
| **pytest** | Testing tool | Small automatic checks that the code behaves |
| **Jupyter Notebooks** | Code + output + notes in one page | Data exploration and the Kaggle runner |
| **Kaggle Notebooks** | Free cloud notebooks with GPU | Where real training happens |
| **Git + GitHub** | Version control + online hosting | Your deliverable. Also how code moves to Kaggle |
| **GitHub Releases** | Attach big files to a repo version | Stores trained model weights (too big for normal commits) |
| **AI coding assistant** | An AI coding agent in your terminal | Writes the code, following this plan |
| *Optional:* **Gradio** | Quick web UI for ML models | Phase 9 demo: upload a cell, get a prediction |

**Version note:** Kaggle notebooks come with PyTorch, torchvision, NumPy, pandas, scikit-learn and Matplotlib preinstalled. Do **not** reinstall torch on Kaggle. Keep a small `requirements-kaggle.txt` with only the extras (e.g. `grad-cam`). For `requirements.txt` (local), pin versions after checking what Kaggle actually uses (`pip freeze` in a Kaggle cell) so local and cloud match.

---

## 5. Repository structure (target)

```
malaria-cnn/
├── README.md                  # Front page of the repo: problem, results, how to run
├── PLAN.md                    # This file
├── requirements.txt           # Local dependencies (pinned)
├── requirements-kaggle.txt    # Only the extras Kaggle doesn't have
├── .gitignore                 # Keeps data, big weights, caches out of git
├── src/
│   ├── config.py              # All settings in one place (image size, epochs, paths...)
│   ├── utils.py               # Seeds, device picking, small helpers
│   ├── data.py                # Find files, split by patient, Dataset + DataLoaders, transforms
│   ├── models.py              # SimpleCNN (ours)
│   ├── train.py               # Training loop, validation, early stopping, checkpoints
│   ├── evaluate.py            # Test metrics, confusion matrix, ROC, threshold tuning
│   ├── gradcam.py             # Explainability heatmaps
│   ├── predict.py             # Classify a single image or folder from the command line
│   └── make_fake_data.py      # Generates a tiny fake dataset for local smoke tests
├── notebooks/
│   ├── 01_explore_data.ipynb  # Look at the data before modelling
│   └── 02_run_on_kaggle.ipynb # Clone repo, train, evaluate, save outputs
├── tests/
│   ├── test_data.py
│   └── test_models.py
├── results/                   # Small outputs worth committing: plots, metrics.json
└── docs/
    └── learning_notes.md      # Plain-English notes, updated every phase
```

---

## 6. Phases

Each phase lists: **Goal**, **What gets built**, **Key decisions**, **Concepts you'll learn**, **Your job**, **Done when**, and **Explain it back**.

---

### Phase 0: Project setup

**Goal:** an empty but well-organised repo that runs.

**What gets built:**
- Folder structure from Section 5 (empty files with header docstrings are fine).
- `src/config.py` with all settings as clearly named constants and comments.
- `src/utils.py`: `set_seed()`, `get_device()` (CUDA → Apple MPS → CPU).
- `src/make_fake_data.py`: creates ~40 small coloured images per class in the same folder layout as the real data, with fake patient-style filenames.
- `.gitignore`, `requirements.txt`, a stub `README.md`, `docs/learning_notes.md`.

**Concepts:** virtual environments, why we fix random seeds (same results every run), what a "device" is (CPU vs GPU).

**Your job:** create the GitHub repo (public is easiest, since Kaggle can clone it without a token), push, and make a Kaggle account with phone verification (needed to turn on Internet and GPU in notebooks).

**Done when:** `python -m src.make_fake_data` runs and creates the fake folders.

**Explain it back:**
1. Why don't we train on the laptop?
2. Why does everything live in `config.py` instead of being typed into each file?

---

### Phase 1: Explore the data

**Goal:** understand the data *before* building anything. Every real ML project starts here.

**What gets built:** `notebooks/01_explore_data.ipynb`, with a markdown cell before every code cell explaining what we're about to check:
- Count images per class (is it balanced?).
- Detect and count duplicates (file hash), and junk files.
- Distribution of image widths/heights (justifies the resize size).
- A grid of 16 random cells per class.
- Parse the patient/slide code from filenames and count how many patients and cells per patient. **Print a sample of filenames first** so we confirm the pattern is real before relying on it.

**Concepts:** class balance, data leakage, why image size matters.

**Your job:** run it on Kaggle (attach the dataset via *Add Input*), look at the cell grids, and see if *you* can spot infected cells by eye.

**Done when:** you know how many images, how many patients, the size range, and the folder quirks are confirmed.

**Explain it back:**
1. What does an infected cell look like compared to a healthy one?
2. What is patient leakage, and why would it make our test score lie?

---

### Phase 2: Data pipeline

**Goal:** turn folders of images into batches of tensors the model can eat.

**What gets built:** `src/data.py`:
- `find_images(data_dir)` → a pandas table: `path, label, patient_id`, deduplicated, `.png` only.
- `split_by_patient(table)` → train / val / test at roughly **70 / 15 / 15**, using scikit-learn's `StratifiedGroupKFold` so **no patient appears in two splits** *and* each split keeps a similar Parasitized/Uninfected balance (a plain `GroupShuffleSplit` can come out lopsided, because healthy patients only contribute Uninfected cells). Print class counts per split.
- Save the split to **`splits.csv`** (`path, label, patient_id, split`) and reuse it in every later phase, so training, evaluation and Grad-CAM all use exactly the same test patients.
- `MalariaDataset` (a PyTorch `Dataset`) that loads one image and its label.
- Transforms:
  - **Train:** resize → random horizontal + vertical flip → random rotation (cells have no "up", so rotating is free extra data) → mild colour jitter (stain colour varies between labs) → to tensor → normalise.
  - **Val/Test:** resize → to tensor → normalise. **No augmentation** here; we test on clean images.
- `get_dataloaders(...)` returns the three `DataLoader`s.

`tests/test_data.py`: runs on fake data; checks no patient overlap between splits, every split contains both classes, correct tensor shape, labels are 0/1.

**Key decisions:**
- Image size: **128×128** for our own CNN. It lives in `config.py`.
- Label encoding: `Uninfected = 0`, `Parasitized = 1` (so "positive" means "infected").

**Concepts:** tensors, batches, train/validation/test sets and why there are three, data augmentation, normalisation.

**Done when:** `pytest` passes and a script prints one batch's shape, e.g. `(64, 3, 128, 128)`.

**Explain it back:**
1. What is the validation set for, and why can't we use the test set for that?
2. Why do we augment only the training images?
3. What do the four numbers in `(64, 3, 128, 128)` mean?

---

### Phase 3: Our own CNN + the training loop

**Goal:** a small CNN built from scratch, and a readable training loop.

**What gets built:** in `src/models.py`, `SimpleCNN`:
```
4 × [Conv 3×3 → BatchNorm → ReLU → MaxPool 2×2]   channels: 32 → 64 → 128 → 256
→ Global Average Pool → Dropout(0.3) → Linear(256 → 1)
```
- Output is **one number (a "logit")**. Sigmoid turns it into a probability of "infected".
- Every layer gets a comment saying what it does and the tensor shape after it.

In `src/train.py`, an **explicit** training loop (no high-level wrappers):
- Loss: `BCEWithLogitsLoss` (standard for yes/no problems).
- Optimiser: Adam, learning rate `1e-3`.
- Learning-rate scheduler: `ReduceLROnPlateau` (lowers the learning rate when progress stalls).
- Up to **30** epochs with **early stopping** (stop if validation loss hasn't improved for 4 epochs). *(Originally ~15; raised after the first Kaggle run reached its best validation loss at epoch 15 of 15.)*
- Save the **best** model (lowest validation loss) to `outputs/`.
- Log per epoch: train loss, val loss, val accuracy, val recall. Save history as JSON and plot loss/accuracy curves.
- Command line: `python -m src.train --model simple_cnn --data-dir <path> [--smoke-test]`. `--smoke-test` = 1 epoch on fake data, CPU-friendly.

**Sanity check (important, and it should be explained):** before full training, try to **overfit one small batch** (train on the same 16 images repeatedly). Loss should drop near zero. If it can't even memorise 16 images, something is broken. Add this as `tests/test_models.py` plus a `--overfit-batch` flag.

**Concepts:** convolution, filters/feature maps, pooling, ReLU, BatchNorm, dropout, logits and sigmoid, loss functions, gradient descent, epochs, learning rate, overfitting vs underfitting, early stopping.

**Done when:** smoke test runs end to end locally, overfit test passes.

**Explain it back:**
1. In one sentence each: what do a conv layer, a pooling layer and dropout do?
2. What happens in one iteration of the training loop (the 5 steps: forward, loss, zero grad, backward, step)?
3. How would you recognise overfitting from the loss curves?

---

### Phase 4: First real training run on Kaggle

**Goal:** train `SimpleCNN` on the full dataset with a GPU.

**What gets built:** `notebooks/02_run_on_kaggle.ipynb`, short and commented:
1. Setup instructions in markdown: *Settings → Accelerator → GPU*, *Internet → On*, *Add Input → the NIH dataset*.
2. `!git clone https://github.com/<you>/malaria-cnn.git` and `%cd malaria-cnn`.
3. `!pip install -q -r requirements-kaggle.txt`
4. Print GPU info (proves the GPU is on).
5. `!python -m src.train --model simple_cnn --data-dir /kaggle/input/...`
6. Copy outputs to `/kaggle/working/` so they appear in the notebook's Output tab.

**Your job:** run it. Download the curves, history JSON and best model. Note the final epoch numbers for the review.

**Done when:** training finishes and the curves look sensible.

**Explain it back:**
1. Looking at *your* curves: did the model overfit? How do you know?
2. Why did training stop when it did?

---

### Phase 5: Proper evaluation

**Goal:** honest numbers on the untouched test set, with a medical lens.

**What gets built:** `src/evaluate.py`:
- Loads the best checkpoint and runs it on the **test** set once.
- Reports: accuracy, **precision**, **recall (sensitivity)**, **specificity**, F1, ROC-AUC.
- Saves: confusion matrix plot, ROC curve, `metrics.json`.
- **Threshold tuning:** by default "infected" means probability > 0.5. Using the **validation** set, find the threshold that reaches recall ≥ 0.98, then report test metrics at both 0.5 and the tuned threshold, side by side.
- A grid of **misclassified** test images with their predicted probability (great for the README's "limitations").

**Concepts:** confusion matrix (TP, FP, TN, FN), precision vs recall trade-off, ROC curve, why tuning happens on validation and never on test.

**Your job:** run on Kaggle, look at the misclassified images. Would *you* have got them right?

**Explain it back:**
1. Define recall and specificity using "infected cells" and "healthy cells".
2. What did lowering the threshold do to recall and to false alarms?
3. Why must the threshold be chosen on validation data, not test data?

---

### Phase 6: (dropped)

This phase originally added a second, pretrained model and compared it with SimpleCNN. It was built, then removed from the project: SimpleCNN is the only model. Go straight from Phase 5 to Phase 7.

---

### Phase 7: Explainability with Grad-CAM

**Goal:** prove the model looks at the **parasite**, not at random background.

**What gets built:** `src/gradcam.py`: heatmap overlays for the model (last conv block of `SimpleCNN`). Save a grid: correct infected, correct healthy, and a few mistakes.

**Concepts:** why "black box" models are a problem in medicine, what Grad-CAM shows.

**Explain it back:**
1. On a correctly-classified infected cell, where is the heatmap hottest? Is that what we'd hope?
2. Why would a doctor trust a model more with these heatmaps?

---

### Phase 8: Prediction script + polished README

**Goal:** make the repo look and work like a finished project.

**What gets built:**
- `src/predict.py`: `python -m src.predict --weights best.pt --image cell.png` → prints label + probability. Also works on a folder.
- `README.md`: problem, dataset (with credit to NIH), approach, **results table and plots using the real numbers you paste in**, Grad-CAM examples, how to run (Kaggle and local), limitations, future work.
- Instructions for you to upload the trained weights to a **GitHub Release** (weights don't go in normal commits).

**Done when:** a stranger could open the repo, understand it in 2 minutes, and reproduce it.

**Explain it back:** walk through the whole pipeline out loud, from folder of images to prediction, in under 2 minutes. (This is basically your viva.)

---

### Phase 9 (optional stretch goals)

Pick any, only after Phase 8 is done:
- **Gradio demo** (`app.py`): upload a cell image, see the prediction and Grad-CAM. Can be hosted free on Hugging Face Spaces.
- **Real-world test:** crop a few dozen cells from the BBBC041 dataset and see how the model holds up on messier images. Great "limitations" material.
- **Lighter model:** try MobileNetV3 or EfficientNet-B0 for a faster, smaller model that could run on a clinic laptop.

---

## 7. What results to expect (and red flags)

- A decent model on this dataset usually lands **around 95% accuracy or higher**.
- **Red flag: 99%+ accuracy.** On this dataset that often means leakage (same patient in train and test, or duplicate images). Our patient split exists to prevent exactly this. If it happens anyway, investigate before celebrating.
- **Red flag:** training accuracy keeps rising while validation accuracy falls → overfitting. Early stopping and augmentation should handle it.
- **Red flag:** loss stuck at ~0.69 from the start → the model is guessing 50/50; something in the data pipeline or labels is wrong.

---

## 8. Mini glossary

| Term | Plain meaning |
|---|---|
| **Tensor** | A multi-dimensional array of numbers (an image is a 3D tensor: channels × height × width) |
| **Batch** | A group of images processed together (e.g. 64 at a time) |
| **Epoch** | One full pass over all training images |
| **Logit** | The model's raw output score before converting to a probability |
| **Loss** | A number measuring how wrong the model is; training tries to shrink it |
| **Gradient** | Which direction to nudge each weight to reduce the loss |
| **Learning rate** | How big each nudge is |
| **Overfitting** | Memorising training images instead of learning general patterns |
| **Augmentation** | Randomly flipping/rotating training images to create variety |
| **Recall / sensitivity** | Of all infected cells, the share we caught |
| **Specificity** | Of all healthy cells, the share we correctly called healthy |

---

## 9. Starter prompts for the coding assistant

Open a terminal in the project folder (with `PLAN.md` inside), start the coding assistant, then:

**To start:**
> Read PLAN.md fully. Then do Phase 0 only. Before writing code, tell me in a few lines what you're about to create and why.

**After reviewing a phase:**
> Phase N looks good. Commit it, then start Phase N+1. Same rules.

**When you don't understand something:**
> Explain [line / function / concept] in `src/xyz.py` like I'm a second-year student who has never done ML. Use an analogy, then show me exactly which lines do it.

**When pasting Kaggle results:**
> Here are my real results from Kaggle: [paste]. Check them for red flags from Section 7 of PLAN.md, then update results/ and the README.
