# Learning notes

Plain-English notes on every concept used in this project, updated every phase.
Each note points to the file (and function) where the idea lives in the code.

---

## Phase 0: Project setup

### Virtual environment
**What:** a private folder (`.venv/`) holding this project's own copy of Python libraries.
**Analogy:** a separate toolbox for each job. Your LiDAR project can use one version of a library and this project another, without either breaking the other.
**How:** `python3 -m venv .venv` creates it; `source .venv/bin/activate` switches your terminal to use it. It's listed in `.gitignore` because everyone builds their own from `requirements.txt`.

### Random seeds
**What:** computers make "random" numbers with a formula that starts from a number called the seed. Same seed → same sequence of "random" numbers.
**Why we care:** ML is full of randomness (which patients go in the test set, the starting weights, which images get flipped). Fixing the seed means running the code twice gives the same results, so if a number changes, it's because *we* changed something.
**Where:** `src/utils.py` → `set_seed()`. Python, NumPy and PyTorch each have their own generator, so all three are seeded. The seed value is `RANDOM_SEED` in `src/config.py`.

### Device (CPU vs GPU)
**What:** the hardware that does the maths.
**Analogy:** a CPU is a few very smart workers; a GPU is thousands of simpler workers. Training a CNN is millions of simple multiplications, which is exactly the GPU's kind of job, so it's often 10 to 50 times faster.
**Where:** `src/utils.py` → `get_device()`. It tries NVIDIA GPU (`cuda`, what Kaggle has) → Apple GPU (`mps`) → plain `cpu`.

### Why everything lives in `config.py`
**What:** every setting (image size, epochs, learning rate, paths) is defined once in `src/config.py`.
**Why:** if the image size were typed separately in `data.py` and `models.py`, changing one and forgetting the other would cause a confusing crash. One source of truth means one place to change.
**Where:** `src/config.py`. It also detects Kaggle (`IS_ON_KAGGLE`) and picks the output folder automatically.

### Smoke test and fake data
**What:** a "smoke test" is a tiny, fast run that only checks the code doesn't crash (the name comes from electronics: power it on and see if it smokes).
**Why:** the real dataset only lives on Kaggle. Fake cartoon cells let us test the whole pipeline on a laptop in seconds.
**Where:** `src/make_fake_data.py` → `main()`. It deliberately copies the real dataset's quirks (nested duplicate folder, `Thumbs.db` junk files, patient codes in filenames) so our data code gets tested against them.

---

## Phase 1: Explore the data

### Look at the data before modelling
**What:** count, measure and look at the images before writing any model code.
**Why:** problems in the data (duplicates, junk files, one patient everywhere) don't cause errors. They silently produce a model that *looks* great and isn't. Exploration is how you find them while they're cheap to fix.
**Where:** `notebooks/01_explore_data.ipynb`, the whole notebook. It even caught a bug in our own fake data generator: plain healthy cells of the same size came out pixel-identical. Fixed in `src/make_fake_data.py` → `draw_fake_cell()` with a random colour shift.

### Duplicates and file hashes
**What:** a hash is a short fingerprint calculated from every byte of a file. Identical files → identical hash; change one byte → completely different hash.
**Why:** if the same image is in both the training and the test set, the model has seen the answer, and the test score is fake-good. We also check for the same image under *both* labels, which would mean the data contradicts itself.
**Where:** notebook section 5, function `file_hash()`.

### Class balance
**What:** how many images each class has.
**Why:** with 95% healthy images, a model that always says "healthy" scores 95% accuracy while being useless. A roughly 50/50 dataset makes accuracy a more honest number (but we still watch recall, because missing infections is the dangerous mistake).
**Where:** notebook section 6.

### Why image size matters
**What:** a CNN needs every input image to be the same size, so we resize them all.
**Why:** too small and the parasite blurs away; too big and training is slow for no gain. The size histogram shows what we're starting from. Our choice (128×128 for our CNN, 224×224 for ResNet18) lives in `src/config.py`.
**Where:** notebook section 7, function `image_width_height()`.

### Data leakage (patient leakage)
**What:** information from the test set sneaking into training.
**Analogy:** a student who has seen the exam paper. Their score says nothing about what they learned. Here, if one patient's cells are in both training and test, the model can recognise that slide's stain and lighting instead of the parasite.
**How we prevent it:** split by patient, so each patient's cells all go to one split. The notebook first *checks* that the patient code in filenames is real and consistent before we rely on it.
**Where:** notebook sections 9 to 11 (`extract_code()`, `split_c_and_p()`). The split itself is built in Phase 2 (`src/data.py` → `split_by_patient()`).

### Regular expressions
**What:** a mini-language for describing text patterns. `^(.+?)_?thinF` means "from the start, capture the shortest text before an optional underscore and the word thinF".
**Where:** notebook section 10, `CODE_PATTERN`.

---

## Phase 2: Data pipeline

### Tensors
**What:** a tensor is a multi-dimensional array of numbers, PyTorch's basic data type. One colour image is a 3D tensor: `(channels, height, width)` = `(3, 128, 128)`.
**Analogy:** a stack of three spreadsheets, one each for red, green and blue, where each cell holds one pixel's brightness.
**Where:** `src/data.py` → `build_train_transform()`, the `transforms.ToTensor()` step.

### Batches
**What:** a group of images processed together. A batch of 64 images is a 4D tensor: `(64, 3, 128, 128)` = (images, colour channels, height, width).
**Why:** a GPU is fastest when it does many images at once, and averaging the error over a batch gives a steadier signal for learning than one image alone.
**Where:** `src/data.py` → `get_dataloaders()`. The batch size is `BATCH_SIZE` in `src/config.py`.

### Train, validation and test sets: why three?
**Analogy:** a student preparing for an exam.
- **Train** = the textbook exercises. The model learns from these.
- **Validation** = practice exams. We use them to make decisions: which epoch was best, which threshold to use. Because we make decisions with them, they slowly stop being "unseen".
- **Test** = the real exam, sat exactly once at the end. It's only honest if no decision was ever based on it.

**Where:** `src/data.py` → `split_by_patient()`. Sizes are `TRAIN_FRACTION`, `VAL_FRACTION` and `TEST_FRACTION` in `src/config.py`.

### Stratified group split
**What:** splitting so that (1) all of one patient's cells stay together (**group**), and (2) every split has a similar infected/healthy mix (**stratified**).
**Why both:** grouping prevents patient leakage. Stratifying matters because healthy volunteers only have healthy cells, so a purely random patient split could give a test set with very few infected cells. The test score would then be noisy, and recall would be measured on only a handful of cases.
**Where:** `src/data.py` → `hold_out_by_patient()` (uses scikit-learn's `StratifiedGroupKFold`) and `check_split_is_safe()`, which refuses any split with leakage or a missing class.

### Saving the split (`splits.csv`)
**What:** the split is made once, written to a CSV file, and reloaded on every later run.
**Why:** training, evaluation and the ResNet18 comparison must all be judged on the *same* test patients, otherwise comparing their scores is meaningless.
**Where:** `src/data.py` → `load_or_create_splits()`. Real split: `splits/splits.csv` (committed to git). Fake split: `data/fake/splits.csv`.

### Data augmentation
**What:** small random changes to training images every time they're loaded: flips, rotations and slight colour changes.
**Why:** it's free extra data. A cell rotated 90° is still the same cell, and stain colour really does vary between labs. It also makes memorising individual images harder, which fights overfitting.
**Why only on training images:** validation and test measure how the model does on *real* cells. Randomly changing them would make the score noisy and less honest.
**Where:** `src/data.py` → `build_train_transform()` vs `build_eval_transform()`. Strengths are set in `src/config.py` (`MAX_ROTATION_DEGREES`, `COLOUR_JITTER_*`).

### Normalisation
**What:** shifting and scaling each colour channel so the values are centred around 0 (roughly -2 to +2) instead of 0 to 1.
**Why:** networks learn faster and more stably when inputs are centred. We use ImageNet's mean and std because ResNet18 needs exactly those, and using the same for SimpleCNN keeps both models' inputs identical.
**Where:** `transforms.Normalize(...)` in `src/data.py`; the values are `NORMALISE_MEAN` and `NORMALISE_STD` in `src/config.py`.

### Dataset and DataLoader
**What:** a `Dataset` knows how to load **one** image and its label (`MalariaDataset.__getitem__`). A `DataLoader` asks the Dataset for many images, stacks them into a batch, and shuffles the order each epoch.
**Analogy:** the Dataset is a librarian who fetches one book on request; the DataLoader is a trolley that collects 64 books and wheels them over to you.
**Where:** `src/data.py` → `MalariaDataset` and `get_dataloaders()`.

### Tests that can fail
**What:** `tests/test_data.py` checks the pipeline automatically (`pytest`). We confirmed the tests are real by deliberately breaking the patient split: the leakage tests failed, as they should.

---

## Phase 3: Our own CNN and the training loop

### Convolution, filters and feature maps
**What:** a convolution slides a small filter (3×3 pixels) across the image. At each position it multiplies the filter's 9 numbers by the pixels underneath and adds them up. The result is a **feature map**: an image showing *where* that filter's pattern appears.
**Analogy:** a stencil you slide over a picture, which glows wherever the picture underneath matches its shape.
**Key point:** we never design the filters. They start random, and training shapes them into useful detectors (edges, then blobs, then "purple blob inside a round cell").
**Where:** `src/models.py` → `ConvBlock.__init__()`, `self.conv`.

### Pooling
**What:** max pooling keeps the largest value in every 2×2 square, halving height and width.
**Why:** less computation, a little tolerance to small shifts, and each later filter "sees" a larger area of the cell.
**Where:** `ConvBlock`, `self.pool`. Follow the shrinking shapes in the `SimpleCNN.forward()` comments: 128 → 64 → 32 → 16 → 8.

### ReLU
**What:** keeps positive numbers and turns negative ones into 0.
**Why:** without a bend like this, stacking layers would be no more powerful than a single layer (a chain of straight lines is still a straight line).
**Where:** `ConvBlock`, `self.relu`.

### BatchNorm
**What:** rescales each feature map to a steady range (average ~0, spread ~1).
**Why:** it keeps the numbers flowing through the network from growing or shrinking out of control, which makes training faster and more stable.
**A subtlety we saw:** in training mode it uses the current batch's statistics; in evaluation mode it uses long-run averages built up over training. With very few training steps those averages aren't ready yet. That's why, on the fake data (1 batch per epoch), validation looked broken until we tried 8 batches per epoch. The real data has ~300 batches per epoch.
**Where:** `ConvBlock`, `self.batch_norm`.

### Global average pooling and dropout
**Global average pooling:** averages each of the 256 final feature maps to one number: "how strongly does pattern k appear anywhere in this cell?" It also lets the model accept any image size.
**Dropout:** during training, randomly switches off 30% of those 256 numbers, so the decision can't hinge on any single feature. It's switched off automatically in `model.eval()`.
**Where:** `SimpleCNN.__init__()`, `self.global_pool` and `self.dropout`.

### Logits and sigmoid
**What:** the model outputs one raw score per image, the **logit** (any number, from -∞ to +∞). The **sigmoid** function squashes it into a probability between 0 and 1. A logit of 0 means 50%, positive means "probably infected", negative means "probably healthy".
**Where:** `SimpleCNN.forward()` returns logits; `src/train.py` → `validate()` applies `torch.sigmoid()`.

### Loss function
**What:** one number measuring how wrong the predictions are. We use **binary cross-entropy** (`BCEWithLogitsLoss`), the standard for yes/no questions. Being confidently wrong is punished heavily.
**Useful landmark:** random 50/50 guessing gives a loss of about **0.69**. A loss stuck there means the model isn't learning (PLAN.md Section 7).
**Where:** `src/train.py` → `train_model()`, `loss_function`.

### Gradient descent and the 5 training steps
**Analogy:** walking downhill in fog. You can't see the valley, but you can feel which way the ground slopes under your feet (the **gradient**) and take a step that way. The **learning rate** is how big each step is.
**The 5 steps, every batch** (`src/train.py` → `train_one_epoch()`):
1. **Forward:** the model scores the images.
2. **Loss:** measure how wrong the scores are.
3. **Zero grad:** clear the previous batch's gradients (PyTorch adds them up otherwise).
4. **Backward:** calculate the gradient for every weight.
5. **Step:** the optimiser (Adam) nudges every weight downhill.

### Epochs, learning rate scheduler and early stopping
**Epoch:** one full pass over all training images.
**Scheduler** (`ReduceLROnPlateau`): when validation loss stalls for 2 epochs, it halves the learning rate. Smaller steps help settle into the bottom of the valley.
**Early stopping:** if validation loss hasn't improved for 4 epochs, stop. We always keep the checkpoint from the best epoch, not the last one.
**Where:** `src/train.py` → `train_model()`. Settings: `LR_SCHEDULER_*`, `EARLY_STOPPING_PATIENCE` and `MAX_EPOCHS` in `src/config.py`.

### Overfitting vs underfitting
**Overfitting:** the model memorises the training images instead of learning general patterns. **On the curves:** training loss keeps falling while validation loss turns upward, opening a widening gap.
**Underfitting:** the model hasn't learned enough. **On the curves:** both losses stay high.
**Where:** `src/train.py` → `plot_history()` draws both losses side by side for exactly this check.

### The overfit-one-batch sanity check
**What:** train on the same 16 images 100 times. A working model *must* be able to memorise them (loss → ~0).
**Why:** it catches broken code (wrong labels, a disconnected layer, a learning rate of 0) in seconds, before you spend an hour of GPU time. We confirmed it works by setting the learning rate to 0: the test failed, as it should.
**Where:** `src/train.py` → `overfit_one_batch()`; run it with `python -m src.train --overfit-batch`. The test is in `tests/test_models.py`.

---

## Phase 4: First real training run on Kaggle

### Moving code to Kaggle with a zip
**What:** `git archive --format=zip --prefix=malaria-cnn/ -o malaria-cnn-code.zip HEAD` packs the files of the latest commit into a zip. On Kaggle, the zip becomes a private dataset that the notebook reads.
**Why `git archive` and not "right-click → compress":** it only includes *committed* files, so the zip never contains `.venv/`, fake data, old outputs or half-finished edits. Kaggle runs exactly the version you committed. That also means: **commit first, then zip**.
**Where:** setup steps at the top of `notebooks/02_run_on_kaggle.ipynb`.

### Read-only inputs, writable working folder
**What:** on Kaggle, `/kaggle/input/` (attached datasets) is read-only; `/kaggle/working/` is writable, and only files there appear in the notebook's Output tab.
**Why it matters:** the first run must *write* `splits/splits.csv`, so the notebook copies the code from the input into `/kaggle/working/malaria-cnn` and runs it there.
**Where:** notebook sections 2 and 3; the output folder is chosen in `src/config.py` (`IS_ON_KAGGLE`, `OUTPUT_DIR`).

### Data-loading workers
**What:** while the GPU trains on one batch, background CPU processes ("workers") are already loading and augmenting the next batches.
**Why:** a GPU is so fast that loading images from disk can become the bottleneck. One worker per CPU core keeps it fed.
**Where:** `NUM_DATA_WORKERS` in `src/config.py`, used by `get_dataloaders()` in `src/data.py`.

### Exit codes
**What:** every program finishes with a number: 0 = success, anything else = failure. `--overfit-batch` exits with 1 if the check fails.
**Why:** the notebook checks it (`_exit_code`) and stops before wasting GPU time on a broken setup.
**Where:** `src/train.py` → `main()` (`sys.exit`), and notebook sections 5 and 6.

---

## What the real data showed (Phase 1 notebook, run on Kaggle)

| Check | Result |
|---|---|
| Images | 55,116 `.png` files, but only **27,558 unique**: the nested `cell_images/cell_images/` folder holds a full second copy |
| Class balance | exactly **13,779 Parasitized / 13,779 Uninfected** (50/50) |
| Duplicates under different names | 0 |
| Same image under both labels | 0 |
| Junk files | 4 `Thumbs.db` |
| Image sizes | widths 46–394, heights 40–385, median 130 px, so 128×128 is a good fit |
| Kaggle mount path | `/kaggle/input/datasets/iarunava/cell-images-for-detecting-malaria/...`, which is why we search instead of hardcoding paths |
| Patients | 196 distinct codes: 148 with both classes, 48 with healthy cells only. This matches the dataset's published ~150 infected + ~50 healthy patients |

### Lesson: check your assumptions on the real data
Our first guess ("the patient code is everything before `thinF`") was wrong in two ways:
1. **356 filenames use `thin_original` instead of `thinF`.** The old code would have stopped with an error on the first real training run.
2. **Some codes have extra letters** (`C173P134N`, `C167P128Re`) that look like variants of plain codes (`C173P134`). Counting them as different patients could split one person across train and test: patient leakage.

**The fix:** the patient ID is just the leading **C number** (`C173P134N` → `C173`). The notebook showed every C number goes with exactly one P number, so this only merges variants of the same code, never two different patients. It's the safe choice.
**Where:** `src/data.py` → `extract_patient_id()`. The real filenames are now test cases in `tests/test_data.py`, and the fake data uses all five filename styles.
