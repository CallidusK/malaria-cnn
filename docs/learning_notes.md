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

### What the real Phase 4 runs taught us
- **Reproducibility works.** The second run (max 30 epochs) produced the *same* split and the *same* first 15 epochs, to 4 decimal places, as the first run (max 15). That's `set_seed()` (`src/utils.py`) plus the saved `splits.csv` doing their job.
- **Why validation loss can be *lower* than training loss.** During training, dropout switches off 30% of the features and augmentation scrambles the images, so the task is deliberately harder. Validation runs in `model.eval()` mode on clean images. So "val below train" doesn't mean anything is wrong here.
- **The epoch limit vs early stopping.** With a 15-epoch limit, the best epoch was the last one: the limit stopped training, not the model. With 30, early stopping took over (best epoch 17, stopped at 21), and validation loss improved only a little (0.1077 → 0.1034). So the model had essentially finished learning.
- **Accuracy vs recall can move in opposite directions.** At epoch 6, recall rose while accuracy fell: the model called more cells "infected", catching more parasites but also raising more false alarms. Phase 5 controls this trade-off deliberately with the decision threshold.
- **Read your axes.** A 0-to-1 axis made 94% and 96% look identical. `plot_history()` (`src/train.py`) now zooms in on the real values. The flip side: a zoomed axis makes small differences *look* big, so always check the numbers on the axis.

**Results files:** `results/simple_cnn_history.json` (the real per-epoch numbers from Kaggle) and `results/simple_cnn_curves.png` (drawn from that file). The split is in `splits/splits.csv`. The trained weights (`outputs/simple_cnn_best.pt`, 1.6 MB) are kept out of git.

---

## Phase 5: Proper evaluation

### The confusion matrix: four kinds of outcome
For every test cell, the model is either right or wrong, about an infected or a healthy cell:

| | predicted healthy | predicted infected |
|---|---|---|
| **truly healthy** | true negative (TN): correctly cleared | false positive (FP): **false alarm**, a human re-checks |
| **truly infected** | false negative (FN): **missed infection**, the dangerous one | true positive (TP): caught |

**Where:** `src/evaluate.py` → `compute_metrics()` counts all four; `plot_confusion_matrices()` draws them.

### The metrics, in words
- **Recall (sensitivity)** = TP / (TP + FN): *of all infected cells, the share we caught.* Our top priority.
- **Specificity** = TN / (TN + FP): *of all healthy cells, the share we correctly called healthy.*
- **Precision** = TP / (TP + FP): *of all cells we called infected, the share that really were.*
- **F1:** one number balancing precision and recall (low if either one is low).
- **Accuracy** = (TP + TN) / all: *the share of all cells we got right.* It can hide a dangerous number of misses if healthy cells dominate, which is why we never look at it alone.

**Where:** `compute_metrics()`. `tests/test_evaluate.py` checks each formula on a 7-cell example worked out by hand, deliberately lopsided so that every metric has a *different* value. A symmetric example once let a recall/specificity mix-up slip through.

### The threshold and the precision-recall trade-off
**What:** the model outputs a probability; the **threshold** turns it into a decision (probability ≥ threshold → "infected").
**Analogy:** a smoke alarm's sensitivity dial. Turn it down (lower threshold) and it catches every fire, but also goes off when you make toast. Turn it up and the toast is fine, but a small fire might be missed.
**For malaria:** a missed infection is far worse than a false alarm, so we lower the threshold until we catch at least 98% of infected cells, and accept more false alarms.
**Where:** `choose_threshold_for_recall()` tries the infected cells' probabilities as thresholds, from highest to lowest, and keeps the **highest** one that reaches 98% recall: enough recall, with the fewest false alarms. (If recall is already above 98% at 0.5, this can even *raise* the threshold.)

### Why the threshold is chosen on validation, never on test
**Analogy:** if you set the pass mark *after* seeing the exam results, the pass rate tells you nothing about how the next class will do.
If we tuned the threshold on the test set, the test score would be the best case for *those exact cells*, not an honest estimate for new patients. So: choose on validation, then measure on test **once**. Test recall may land a little below the 98% target. That's the honest number, and we report it as it is.
**Where:** `evaluate_model()`: step 1 uses only `val_probabilities`; step 2 only measures `test_probabilities`.

### The ROC curve and AUC
**ROC curve:** recall (y) against false-alarm rate (x) for *every* possible threshold at once. Top-left corner = perfect; the diagonal = coin-flipping.
**AUC** (area under that curve): the chance that a random infected cell gets a higher probability than a random healthy cell. 1.0 = perfect ranking, 0.5 = random. It doesn't depend on any single threshold.
**Where:** `plot_roc_curve()`, using scikit-learn's `roc_curve` and `roc_auc_score`.

### Looking at the mistakes
**What:** a grid of misclassified test cells, the most confidently wrong first.
**Why:** numbers tell you *how often* the model is wrong; pictures tell you *why*. Blurry images, odd stains or mislabelled cells all go into the README's "limitations".
**Where:** `plot_misclassified()`.

### What the real Phase 5 results showed (test set: 29 patients, 3,726 cells)
| | threshold 0.5 | tuned 0.073 |
|---|---|---|
| missed infections | 132 | 61 |
| false alarms | 33 | 245 |
| recall | 92.1% | 96.4% |
| specificity | 98.4% | 88.1% |

ROC-AUC 0.985. Full numbers: `results/simple_cnn_metrics.json`.

- **The trade-off, for real:** lowering the threshold halved the missed infections (132 → 61), at the cost of about 7× more false alarms (33 → 245). Accuracy *fell* (95.6% → 91.8%), even though the model became safer for patients. That's why accuracy is the wrong thing to optimise here.
- **Validation doesn't transfer perfectly to new patients:** the threshold reached 98.05% recall on the validation patients, but 96.4% on the test patients. That gap is the honest answer to "how does it do on *new* people?". We report it as it is, and we must **not** lower the threshold further by looking at test results.
- **Some "mistakes" may be label errors.** In `results/simple_cnn_misclassified.png`, many "missed infections" (labelled Parasitized) show no visible parasite, and several "false alarms" (labelled Uninfected) show purple stained spots. Those spots could be parasites, or platelets/stain debris on a healthy cell; it takes an expert to tell. Either way, we never edit the test set to fit the model. These go into the README's limitations.

---

## Phase 6: Transfer learning (ResNet18) and comparison

### Transfer learning
**What:** instead of starting from random weights, start from a model already trained on a huge dataset (ResNet18 on ImageNet: 1.2 million everyday photos), and adapt it to our task.
**Why it helps, even though ImageNet has no blood cells:** a CNN's early layers learn *general* visual building blocks (edges, textures, colour blobs, round shapes) that are useful for almost any image. Only the last layers are specific to dogs-vs-cars. We keep the general knowledge and retrain the specific part.
**Analogy:** hiring an experienced photographer to learn microscopy, instead of someone who has never looked at an image. They still need training, but they already know how to *see*.
**We saw it happen:** in the overfit check, pretrained ResNet18 memorised 16 cells in about 10 steps; SimpleCNN from scratch needed about 40.
**Where:** `src/models.py` → `ResNet18Classifier`. Its last layer (`fc`: 512 features → 1,000 ImageNet classes) is replaced by `Linear(512 → 1)`.

### Freezing, then fine-tuning
**Stage 1, frozen** (3 epochs): every pretrained weight is locked (`requires_grad = False`); only the new last layer (513 numbers) learns.
**Why first:** the new layer starts random. If everything learned at once, its large, random errors would send large, damaging updates back through the pretrained layers and wreck what they knew.
**Stage 2, fine-tune:** everything unlocks (11.2 million numbers), with a 10× smaller learning rate (1e-4) and **AdamW** (Adam + *weight decay*, a gentle pull of every weight towards zero). Small steps adjust the pretrained knowledge to blood cells without throwing it away.
**Where:** `src/train.py` → `train_model()` (the two stages) and `run_training_stage()` (the epoch loop, shared with SimpleCNN). Helpers: `freeze_pretrained_layers()`, `unfreeze_all_layers()` in `src/models.py`.

### The BatchNorm trap
**What:** freezing weights does **not** freeze BatchNorm's *running statistics* (the average and spread it keeps for evaluation mode). In training mode they keep updating from every batch, so "frozen" layers would quietly change anyway.
**Fix:** during the frozen stage, the BatchNorm layers are switched back to evaluation mode right after `model.train()`.
**Where:** `keep_batchnorm_statistics_frozen()` in `src/models.py`, used by `train_one_epoch(..., batchnorm_frozen=True)`. A test checks that a real frozen training epoch leaves the statistics untouched. A first version of that test only checked the helper function, and **missed** a deliberately broken training loop; the improved test catches it.

### Comparing models fairly
**What:** `results/comparison.md` puts both models side by side: size, training time, speed, and test results at both thresholds.
**Fair means:** the same 29 test patients, thresholds tuned on validation only, and speed measured on the same GPU. `src/compare.py` *refuses* to compare models tested on different test sets.
**Not only accuracy:** a model for a clinic laptop must also be small and fast. ResNet18 has about 29× more parameters than SimpleCNN, and works on images about 3× larger (224×224 vs 128×128 pixels), so expect it to be slower per cell.
**Where:** `src/compare.py` → `benchmark_inference_speed()` (cells per second, using random images, since speed doesn't depend on what the weights learned) and `write_comparison()` (the table, built only from result files).

### Downloading pretrained weights on a Mac
Python installed from python.org on macOS can fail to download with `CERTIFICATE_VERIFY_FAILED` until its one-time certificate setup is run (`Install Certificates.command`, in the Python folder under Applications). Kaggle (Linux) doesn't have this problem.

### What the real Phase 6 results showed
Full table: `results/comparison.md` (rebuilt locally from the result files, and identical to the one made on Kaggle).

| tuned thresholds (test set, 29 patients) | SimpleCNN | ResNet18 |
|---|---|---|
| missed infections | 61 | 49 |
| false alarms | 245 | 66 |
| specificity | 88.1% | 96.8% |
| ROC-AUC | 0.985 | 0.994 |
| size of the weights | 1.6 MB | 44.7 MB |
| cells per second (T4 GPU) | 5,089 | 1,247 |

- **The biggest win was false alarms, not missed infections.** ResNet18 separates the two classes more cleanly (higher AUC), so it only had to lower its threshold to 0.20 to reach 98% validation recall. SimpleCNN had to go all the way down to 0.073, and paid with 3.7× more false alarms.
- **Small differences on one test set deserve caution.** 61 vs 49 missed infections, on 29 patients, is modest. The AUC and false-alarm gaps are larger and more convincing.
- **The price:** 29× more parameters and about 4× slower. Speed measurements vary by a few percent between runs (one run measured 4,988 / 1,171 cells per second), but the ratio stays the same. Speed on a laptop *CPU*, which a clinic might have, has **not** been measured.
- **Fine-tuning is where transfer learning pays off:** in the frozen stage, validation loss only reached 0.24; one epoch of fine-tuning took it to 0.11 (`results/resnet18_curves.png`, dashed line).
- **Both models get the same cells "wrong".** Several of ResNet18's most confident mistakes are the very images SimpleCNN got wrong: "infected" cells with no visible parasite, and "healthy" cells with purple spots. Two very different models confidently disagreeing with the same labels makes label errors more plausible, though only an expert could confirm it.

---

## Phase 7: Explainability with Grad-CAM

### Why "black box" models are a problem in medicine
A model that only says "infected, 97%" gives a doctor nothing to check. It might be right for the wrong reason: for example, reacting to a slide's stain colour or a smudge on the cell's edge instead of the parasite. Such a shortcut can score well on one dataset and fail on the next hospital's slides. Seeing *where* the model looked lets a human judge whether its reasons make sense.

### What Grad-CAM shows, and how
**Analogy:** asking a student "which part of the picture made you say that?" and having them circle it.
**How it works:**
1. The last convolutional layer's feature maps say *where* each learned pattern appears (ResNet18: 512 maps of 7×7; SimpleCNN: 256 maps of 16×16).
2. The *gradient* of the model's answer with respect to each map says how much that map mattered for this answer.
3. Add the maps together, weighted by how much they mattered, and keep the positive part. That gives a coarse heatmap, stretched to the image's size and laid over the cell. **Red = pushed the model towards its answer.**

**Why the last convolutional layer:** earlier layers find simple things (edges, colours) that say little about the decision. After the last convolution, the global average pool throws away *where* things are. The last convolutional layer is the final point that has both meaning and location.
**Where:** `src/gradcam.py` → `target_layer_for()` (which layer), `compute_heatmaps()` (Grad-CAM itself, using the `grad-cam` library), `overlay()` (drawing).

### Explaining the model's own answer
Each heatmap explains *what the model decided*: "why infected?" for cells it called infected, "why healthy?" for cells it called healthy. If we always asked "why infected?", a healthy cell would have almost no evidence anywhere. Grad-CAM scales every heatmap to 0..1, so that near-zero noise would be stretched into bright, misleading hot spots.
**Where:** `compute_heatmaps(..., explain_labels)`; the labels come from the tuned thresholds in `results/<model>_metrics.json`, the same ones as Phase 5, not re-tuned.

### A first sanity check (fake data)
Even a barely-trained smoke-test ResNet18 lit up on the purple blob in most correctly caught "infected" cartoon cells. On a missed one, it had looked at the plain middle of the cell instead of the blob near the edge. That's the kind of reasoning Grad-CAM makes visible.

### What the real heatmaps showed (test set, `results/*_gradcam.png`)
Before drawing them, both models were re-evaluated on the laptop with the downloaded images: thresholds, AUC, missed infections and false alarms matched the Kaggle results **exactly**. So these heatmaps explain the same decisions we reported.

*(Corrected after a recheck: see "Lesson: check your explanations too" below.)*

- **Correctly caught infections: the heat sits on the purple stained parasite**, for both models, in the examples shown. SimpleCNN's heatmaps are sharper (a 16×16 grid vs ResNet18's 7×7). One honest exception from the first run: a SimpleCNN cell called infected with p = 0.078 (only just above its very low 0.073 threshold) had its heat on a tiny speck at the image's corner, not on a clear parasite.
- **"Why healthy?" heatmaps: trustworthy for SimpleCNN only.** SimpleCNN's spread over the body of the cell ("no spot anywhere"), and they pass the rotation check. ResNet18's do **not** pass it, so they're hidden (see below).
- **False alarms: both models fire on purple spots** in cells labelled healthy. Whether those spots are a parasite the label missed, or a platelet or stain speck on a healthy cell, only an expert can say.
- **Missed infections: there's no spot to find.** SimpleCNN's heat is spread over the cell, just as for truly healthy cells, and to the eye most of these "infected" cells show no clear parasite (one has faint marks in a pale centre).
- **78 test cells fooled both models.** In 5 of the 6 shown, the label says infected but no clear parasite is visible; in the other, the label says healthy but two purple blobs are there. This makes **label errors** a likely explanation for part of the "mistakes", but it doesn't prove it.
- **Practical note:** Grad-CAM for both models on the full test set, including the rotation check, takes about 15 to 20 minutes on the laptop.

### Lesson: check your explanations too (the rotation check)
**What happened:** at first, I described some of ResNet18's "why healthy?" heatmaps as peaking at the picture's bottom-right corner, and guessed it might be a shortcut. A guess isn't good enough, so we tested it.
**The test:** rotate each cell 180° and run Grad-CAM again. A heatmap that shows something *in the cell* turns with the cell. A heatmap that sits at a fixed spot *of the picture* doesn't.
**The results** (agreement after rotating back: 1.0 = perfectly turns with the cell, 0 = unrelated), from `results/gradcam_checks.json`:

| | "why infected?" | "why healthy?" |
|---|---|---|
| SimpleCNN | 0.96 ✓ | 0.93 ✓ |
| ResNet18 | 0.97 ✓ | **0.07 ✗** |

In a first test on 300 cells, 91% of ResNet18's "why healthy?" heatmaps peaked in the bottom-right quarter **whether the cell was rotated or not**.
**What it means:**
- It is **not** a shortcut in ResNet18's *decisions*: rotating cells changed its probabilities by only 0.018 on average (3 of 300 decisions changed).
- It **is** a failure of the *explanation*. "Healthy" means "no parasite anywhere", so there's no single place to point to. Grad-CAM then falls back on a quirk of how ResNet18's strided layers process the picture, which favours one corner. Those heatmaps say nothing about the cell.

**The fix:** every kind of heatmap must now pass the rotation check (agreement ≥ 0.5, `GRADCAM_MIN_AGREEMENT` in `src/config.py`); failing ones are **hidden** in the figures, with a note saying why. **Where:** `src/gradcam.py` → `content_agreement()` and `rotation_check()`.
**The bigger lesson:** an explanation method can be wrong too. Heatmaps look convincing whatever they show, so they need checking like any other result.

---

## Phase 8: Prediction script and README

### Using the model: `predict.py`
**What:** the step a user would actually run. Give it a cell image (or a folder), and get "Parasitized" or "Uninfected" plus the probability of infection.
**Two details that matter:**
- **The same preparation as in testing.** It uses exactly `build_eval_transform()` (resize + normalise, no augmentation). A test confirmed that `predict.py` gives *identical* probabilities to the evaluation code on 200 real test cells, for both models. Otherwise its answers wouldn't match the results we report.
- **The tuned threshold by default.** It reads the threshold chosen on validation in Phase 5 (`results/<model>_metrics.json`) and prints which threshold it used and why. `--threshold` overrides it.

**Where:** `src/predict.py` → `predict_files()` (the work), `choose_threshold()`, `find_image_files()` (skips junk like `Thumbs.db`).

### Checking a README against the results
**What:** every number in the README was copied from `results/`, then a script pulled out all 161 numbers and looked for each one in the result files.
**What it caught:**
- one number (0.018) that came from a one-off check rather than from `results/`, which made the README's line "every number comes from the result files" untrue. Both were reworded.
- a rounding slip: "about 0.10" where the history file says 0.107 (rounded, 0.11).

**Why bother:** a README is the only part of a project most people read. A wrong number there, even a small one, undermines trust in all the others.

### Model weights and GitHub Releases
**What:** trained weights (`outputs/*_best.pt`) are binary files that change completely with every training run. Git is built for text that changes a little at a time, so weights go into a **GitHub Release** (a downloadable attachment to a tagged version) instead of into the commit history.
**Where:** `README.md` → "Trained weights", which has the steps to publish and to download them.
