"""
config.py: every setting for the project, in one place.

Where this fits in the pipeline:
    data -> model -> train -> evaluate -> predict
Every one of those steps reads its settings from here. If you want to change
the image size, the number of epochs or a file path, change it HERE and
nowhere else. That way two files can never disagree about a setting.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Where are we running? (laptop or Kaggle)
# ---------------------------------------------------------------------------

# Kaggle sets this environment variable inside its notebooks. If it exists,
# we know we're on Kaggle and should save outputs where Kaggle can see them.
IS_ON_KAGGLE: bool = "KAGGLE_KERNEL_RUN_TYPE" in os.environ

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# The project's top folder (the one containing src/, tests/, PLAN.md ...).
# __file__ is this file; .parent is src/; .parent again is the project root.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Tiny fake dataset for quick local tests, made by src/make_fake_data.py.
FAKE_DATA_DIR: Path = PROJECT_ROOT / "data" / "fake"

# Where the real dataset lives on Kaggle. We only give the top folder;
# src/data.py searches inside it, because the exact layout can vary.
KAGGLE_INPUT_DIR: Path = Path("/kaggle/input")

# Where trained models, plots and metrics are written.
# On Kaggle, only files under /kaggle/working/ show up in the Output tab.
if IS_ON_KAGGLE:
    OUTPUT_DIR: Path = Path("/kaggle/working/outputs")
else:
    OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"

# Small, final results worth committing to git (plots, metrics.json).
RESULTS_DIR: Path = PROJECT_ROOT / "results"

# The train/val/test split, saved once and reused by every later step, so
# training, evaluation and the ResNet18 comparison all use the SAME test patients.
# The real split is committed to git; the fake one is regenerated with the fake data.
SPLITS_CSV: Path = PROJECT_ROOT / "splits" / "splits.csv"
FAKE_SPLITS_CSV: Path = FAKE_DATA_DIR / "splits.csv"

# ---------------------------------------------------------------------------
# Classes and labels
# ---------------------------------------------------------------------------

# The folder names in the dataset. The folder an image is in IS its label.
CLASS_NAMES: list[str] = ["Uninfected", "Parasitized"]

# Turning class names into numbers, because models only understand numbers.
# Parasitized = 1 so that "positive" means "infected" (the thing we hunt for).
LABEL_UNINFECTED: int = 0
LABEL_PARASITIZED: int = 1

# Only files with this extension are images. Everything else (e.g. Thumbs.db)
# is junk and gets ignored.
IMAGE_EXTENSION: str = ".png"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

# A "seed" fixes the starting point of the random number generator, so every
# random choice (data split, weight init, augmentation) repeats exactly.
# 42 has no special meaning; any fixed number works.
RANDOM_SEED: int = 42

# ---------------------------------------------------------------------------
# Data split (by patient, never randomly by image)
# ---------------------------------------------------------------------------

# Fractions of the data for each split. They must add up to 1.0.
TRAIN_FRACTION: float = 0.70  # the model learns from these
VAL_FRACTION: float = 0.15    # used to pick the best epoch and the threshold
TEST_FRACTION: float = 0.15   # touched once, at the very end, for honest scores

# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

# Every image is resized to a square of this many pixels per side.
# Real cell images range from about 46 to 395 pixels, so we must pick one size.
SIMPLE_CNN_IMAGE_SIZE: int = 128  # small enough to train fast, big enough to see the parasite
RESNET_IMAGE_SIZE: int = 224      # ResNet18 was pretrained on 224x224 images

# Number of colour channels: red, green, blue.
NUM_CHANNELS: int = 3

# Normalisation: shift and scale each colour channel so values sit around 0
# instead of 0..1. Networks train faster and more stably on centred inputs.
# These are the average (mean) and spread (std) of red, green and blue over the
# ImageNet photo collection. ResNet18 REQUIRES them, because it was pretrained
# on images normalised this way; we use them for SimpleCNN too, so both models
# see identically prepared images.
NORMALISE_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
NORMALISE_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

# ---------------------------------------------------------------------------
# Data augmentation (training images only)
# ---------------------------------------------------------------------------

# Augmentation = random small changes to training images, so the model sees
# a slightly different version of each cell every epoch.

# A cell has no "up", so any rotation is a realistic new view. 180 means
# "rotate by a random angle between -180 and +180 degrees".
MAX_ROTATION_DEGREES: float = 180.0

# Stain colour and lighting vary between labs and microscopes. Small random
# colour changes teach the model not to depend on one exact shade.
# Each value is the maximum change, as a fraction (0.1 = up to 10%).
COLOUR_JITTER_BRIGHTNESS: float = 0.1
COLOUR_JITTER_CONTRAST: float = 0.1
COLOUR_JITTER_SATURATION: float = 0.1
COLOUR_JITTER_HUE: float = 0.02  # hue shifts colours around the colour wheel; keep it tiny so purple stays purple

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

# How many images the model looks at before each weight update.
BATCH_SIZE: int = 64

# An "epoch" is one full pass over all training images. This is the maximum;
# early stopping usually ends training sooner.
# Raised from 15 to 30 after the first Kaggle run: its best epoch was the last
# one (validation loss still improving), so 15 cut training short. With 30,
# early stopping decides when to stop instead of the limit.
MAX_EPOCHS: int = 30

# Stop training if validation loss hasn't improved for this many epochs.
EARLY_STOPPING_PATIENCE: int = 4

# Learning rate: how big each weight update step is.
SIMPLE_CNN_LEARNING_RATE: float = 1e-3

# The scheduler lowers the learning rate when validation loss stops improving.
# Its patience must be SHORTER than early stopping's, otherwise training would
# stop before the learning rate ever got a chance to drop.
LR_SCHEDULER_PATIENCE: int = 2
LR_SCHEDULER_FACTOR: float = 0.5  # new learning rate = old learning rate x 0.5

# Dropout randomly switches off this fraction of neurons during training,
# so the model can't rely too heavily on any single one.
DROPOUT_PROBABILITY: float = 0.3

# SimpleCNN: how many filters (feature detectors) each of its 4 blocks has.
# Early blocks find simple things (edges, colour patches); later blocks combine
# them into complex things (a purple blob inside a round cell). Later blocks get
# more filters because there are more combinations worth detecting.
SIMPLE_CNN_CHANNELS: tuple[int, int, int, int] = (32, 64, 128, 256)

# Size of the square window each convolution filter looks at, in pixels.
CONV_KERNEL_SIZE: int = 3

# Max pooling shrinks height and width by this factor in every block.
POOL_SIZE: int = 2

# ---------------------------------------------------------------------------
# Overfit-one-batch sanity check (Phase 3)
# ---------------------------------------------------------------------------

# Train repeatedly on ONE small batch. A working model and training loop
# can memorise it, driving the loss close to zero. If it can't, something is broken.
OVERFIT_BATCH_SIZE: int = 16
OVERFIT_STEPS: int = 100
OVERFIT_TARGET_LOSS: float = 0.05  # "close to zero" (random guessing scores about 0.69)

# How many CPU worker processes load images in the background while the GPU
# trains. On Kaggle, one per CPU core, so the GPU isn't left waiting for images.
# Locally, 0 = load in the main process (slower, but simplest and safest).
if IS_ON_KAGGLE:
    NUM_DATA_WORKERS: int = os.cpu_count() or 1  # cpu_count() can return None; fall back to 1
else:
    NUM_DATA_WORKERS: int = 0

# ---------------------------------------------------------------------------
# Transfer learning (ResNet18, Phase 6)
# ---------------------------------------------------------------------------

FROZEN_EPOCHS: int = 3               # stage 1: train only the new last layer
FINE_TUNE_LEARNING_RATE: float = 1e-4  # stage 2: small steps so we don't wreck pretrained knowledge

# ---------------------------------------------------------------------------
# Evaluation (Phase 5)
# ---------------------------------------------------------------------------

# Default rule: if the predicted probability of "infected" is above this, say infected.
DEFAULT_THRESHOLD: float = 0.5

# We tune the threshold on the validation set so we catch at least this share
# of infected cells. Missing a sick patient is worse than a false alarm.
TARGET_RECALL: float = 0.98

# ---------------------------------------------------------------------------
# Smoke test and fake data (local testing only)
# ---------------------------------------------------------------------------

# A "smoke test" is a tiny run that only checks the code doesn't crash.
SMOKE_TEST_EPOCHS: int = 1

FAKE_IMAGES_PER_CLASS: int = 40
# Enough patients that every split (train/val/test) gets several of them.
FAKE_NUM_PATIENTS: int = 20
# Like the real data, some patients are healthy volunteers with no infected cells.
FAKE_NUM_HEALTHY_ONLY_PATIENTS: int = 4
FAKE_IMAGE_MIN_SIZE: int = 50    # real cell images also vary in size,
FAKE_IMAGE_MAX_SIZE: int = 140   # so the fake ones do too
