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
