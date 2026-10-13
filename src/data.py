"""
data.py: turns folders of cell images into batches of tensors the model can learn from.

Where this fits in the pipeline:
    [data] -> model -> train -> evaluate -> predict

The journey of the data, in the order of this file:
    1. find_images()           folders of .png files  -> a table (path, label, patient_id)
    2. split_by_patient()      table -> train / val / test, never splitting a patient
    3. load_or_create_splits() save that split to splits.csv once, reuse it forever after
    4. transforms              resize, augment (training only), turn into normalised tensors
    5. MalariaDataset          loads ONE image + its label when asked
    6. get_dataloaders()       groups images into shuffled batches, e.g. (64, 3, 128, 128)

Try it:
    python -m src.make_fake_data
    python -m src.data
"""

import argparse
import hashlib
import re
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src import config
from src.utils import set_seed

# Folder name -> number the model is trained on.
CLASS_NAME_TO_LABEL: dict[str, int] = {
    "Uninfected": config.LABEL_UNINFECTED,
    "Parasitized": config.LABEL_PARASITIZED,
}

# The three splits, in a fixed order (used for printing and for the dataloaders).
SPLIT_NAMES: list[str] = ["train", "val", "test"]


# ===========================================================================
# 1. Find the images
# ===========================================================================

# A regular expression (a pattern for matching text). It reads:
#   ^       from the start of the filename,
#   (.+?)   capture the shortest run of characters (this is the patient code),
#   _?      then an optional underscore,
#   thinF   then the text "thinF" (a thin blood film slide), in any upper/lower case.
# Example: "C33P1thinF_IMG_20150619_114756a_cell_179.png" -> "C33P1"
PATIENT_CODE_PATTERN = re.compile(r"^(.+?)_?thinF", re.IGNORECASE)


def extract_patient_id(filename: str) -> str:
    """
    Read the patient/slide code from the start of a filename.

    For now the WHOLE code (e.g. "C33P1") counts as one patient. If the Phase 1
    notebook shows that a patient is really identified differently (e.g. only
    by the "P" number), this is the only function that needs to change.

    A filename that doesn't fit the pattern raises an error on purpose: if we
    can't tell which patient an image belongs to, we can't split it safely.
    """
    match = PATIENT_CODE_PATTERN.match(filename)
    if match is None:
        raise ValueError(
            f"Can't find a patient code in the filename '{filename}'. "
            "Check the filename pattern in src/data.py -> extract_patient_id()."
        )
    # Upper case, so "c33p1" and "C33P1" count as the same patient.
    return match.group(1).upper()


def file_hash(path: Path) -> str:
    """
    A fingerprint of the file's contents. Identical files give identical hashes.
    MD5 is fast and fine for spotting duplicates (it's not used for security here).
    """
    return hashlib.md5(path.read_bytes()).hexdigest()


def remove_duplicates(table: pd.DataFrame) -> pd.DataFrame:
    """
    Keep exactly one copy of every unique image.

    Duplicates must go: if one copy landed in training and another in the test
    set, the model would be tested on an image it has already seen.
    Images that appear under BOTH labels are dropped completely, because we
    can't know which label is right.
    """
    labels_per_hash = table.groupby("hash")["label"].nunique()
    conflicting_hashes = labels_per_hash[labels_per_hash > 1].index
    if len(conflicting_hashes) > 0:
        print(f"WARNING: {len(conflicting_hashes)} images appear under BOTH labels. Dropping them.")
    table = table[~table["hash"].isin(conflicting_hashes)]

    number_before = len(table)
    table = table.drop_duplicates(subset="hash", keep="first")
    print(f"Removed {number_before - len(table)} duplicate copies; {len(table)} unique images remain.")
    return table.reset_index(drop=True)


def find_images(data_dir: Path) -> pd.DataFrame:
    """
    Search data_dir (and every folder inside it) for labelled cell images.

    Returns a table with one row per unique image:
        path        location relative to data_dir (so the table works on any computer)
        label       0 = Uninfected, 1 = Parasitized
        patient_id  e.g. "C33P1"

    We search instead of using one fixed path, because the dataset's folder
    layout differs between versions (e.g. a nested cell_images/cell_images/).
    """
    rows = []
    # sorted() so files are always visited in the same order, on any computer.
    for path in sorted(data_dir.rglob("*")):
        is_image = path.is_file() and path.suffix.lower() == config.IMAGE_EXTENSION
        if not is_image:
            continue  # junk like Thumbs.db, or a folder

        class_name = path.parent.name
        if class_name not in CLASS_NAME_TO_LABEL:
            continue  # an image outside a class folder has no label

        rows.append({
            "path": path.relative_to(data_dir).as_posix(),
            "label": CLASS_NAME_TO_LABEL[class_name],
            "patient_id": extract_patient_id(path.name),
            "hash": file_hash(path),
        })

    if not rows:
        raise FileNotFoundError(f"No labelled .png images found under {data_dir}")

    table = pd.DataFrame(rows)
    print(f"Found {len(table)} labelled .png files under {data_dir}")
    table = remove_duplicates(table)
    return table.drop(columns="hash")  # the hash was only needed for de-duplication


# ===========================================================================
# 2. Split by patient
# ===========================================================================

def hold_out_by_patient(
    table: pd.DataFrame, fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Set aside roughly `fraction` of the images, taking whole patients only.

    StratifiedGroupKFold cuts the data into N equal "folds" (portions) where:
      - "Group":      all images of one patient stay in the same fold, and
      - "Stratified": each fold keeps a similar Parasitized/Uninfected mix.
    One fold is about 1/N of the data, so to hold out `fraction` we use
    N = round(1 / fraction), and keep just the first fold.

    Returns (kept, held_out).
    """
    num_folds = round(1 / fraction)
    splitter = StratifiedGroupKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    folds = splitter.split(X=table, y=table["label"], groups=table["patient_id"])
    kept_rows, held_out_rows = next(folds)  # we only need the first fold

    kept = table.iloc[kept_rows].reset_index(drop=True)
    held_out = table.iloc[held_out_rows].reset_index(drop=True)
    return kept, held_out


def split_by_patient(table: pd.DataFrame, seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """
    Divide the images into train / val / test (about 70 / 15 / 15), by patient.

    Returns the same table with an extra "split" column.
    We split in two steps: first take the test set out of everything, then
    take the validation set out of what's left.
    """
    total_fraction = config.TRAIN_FRACTION + config.VAL_FRACTION + config.TEST_FRACTION
    if abs(total_fraction - 1.0) > 1e-6:
        raise ValueError(f"Split fractions in config.py must add up to 1.0, not {total_fraction}")

    # Step 1: test set = ~15% of ALL images.
    rest, test = hold_out_by_patient(table, config.TEST_FRACTION, seed)

    # Step 2: validation set = ~15% of ALL images, but we now pick from the
    # remaining 85%. 15 / 85 is about 0.18, i.e. 18% of what's left.
    val_fraction_of_rest = config.VAL_FRACTION / (1 - config.TEST_FRACTION)
    train, val = hold_out_by_patient(rest, val_fraction_of_rest, seed)

    train["split"] = "train"
    val["split"] = "val"
    test["split"] = "test"
    return pd.concat([train, val, test], ignore_index=True)


def check_split_is_safe(split_table: pd.DataFrame) -> None:
    """
    Stop with a clear error if the split could give misleading results:
      - a patient appears in more than one split (patient leakage), or
      - a split is missing one of the two classes (can't measure recall or specificity).
    """
    splits_per_patient = split_table.groupby("patient_id")["split"].nunique()
    leaking_patients = splits_per_patient[splits_per_patient > 1].index.tolist()
    if leaking_patients:
        raise ValueError(f"Patient leakage! These patients are in several splits: {leaking_patients[:10]}")

    for split_name in SPLIT_NAMES:
        labels_in_split = set(split_table.loc[split_table["split"] == split_name, "label"])
        if labels_in_split != set(CLASS_NAME_TO_LABEL.values()):
            raise ValueError(f"The {split_name} split doesn't contain both classes: {labels_in_split}")


def print_split_summary(split_table: pd.DataFrame) -> None:
    """Print how many images, patients and infected cells each split has."""
    print(f"\n{'split':<6} {'images':>7} {'patients':>9} {'uninfected':>11} {'parasitized':>12} {'% of all':>9}")
    for split_name in SPLIT_NAMES:
        rows = split_table[split_table["split"] == split_name]
        num_uninfected = (rows["label"] == config.LABEL_UNINFECTED).sum()
        num_parasitized = (rows["label"] == config.LABEL_PARASITIZED).sum()
        share_of_all = 100 * len(rows) / len(split_table)
        print(f"{split_name:<6} {len(rows):>7} {rows['patient_id'].nunique():>9} "
              f"{num_uninfected:>11} {num_parasitized:>12} {share_of_all:>8.1f}%")
    print()


# ===========================================================================
# 3. Save the split once, reuse it forever
# ===========================================================================

def default_splits_csv_for(data_dir: Path) -> Path:
    """The fake data gets its own splits file, so it never overwrites the real one."""
    if data_dir.resolve() == config.FAKE_DATA_DIR.resolve():
        return config.FAKE_SPLITS_CSV
    return config.SPLITS_CSV


def check_files_exist(split_table: pd.DataFrame, data_dir: Path) -> None:
    """Make sure a saved splits.csv really belongs to the data in data_dir."""
    missing = [path for path in split_table["path"] if not (data_dir / path).exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} images listed in the splits file are not in {data_dir} "
            f"(e.g. {missing[0]}). Is --data-dir pointing at the right dataset?"
        )


def load_or_create_splits(data_dir: Path, splits_csv: Path) -> pd.DataFrame:
    """
    Load the train/val/test split from splits_csv if it exists; otherwise make it and save it.

    Saving the split means every later run (training today, evaluation next
    week, the ResNet18 comparison) uses exactly the same test patients,
    even if a library version changes how "random" splitting works.
    """
    if splits_csv.exists():
        print(f"Loading the saved split from {splits_csv}")
        split_table = pd.read_csv(splits_csv, dtype={"patient_id": str})
        check_files_exist(split_table, data_dir)
    else:
        print(f"No saved split at {splits_csv}. Creating one.")
        split_table = split_by_patient(find_images(data_dir))
        splits_csv.parent.mkdir(parents=True, exist_ok=True)
        split_table.to_csv(splits_csv, index=False)
        print(f"Saved the split to {splits_csv}")

    check_split_is_safe(split_table)
    print_split_summary(split_table)
    return split_table


# ===========================================================================
# 4. Transforms: from an image file to a tensor
# ===========================================================================

def build_train_transform(image_size: int) -> transforms.Compose:
    """
    Steps applied to every TRAINING image, every time it's loaded.

    The random steps (flips, rotation, colour jitter) are "data augmentation":
    the model sees a slightly different version of each cell every epoch, which
    acts like extra training data and makes it harder to just memorise images.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),  # every image the same size
        transforms.RandomHorizontalFlip(),            # mirror left-right, 50% of the time
        transforms.RandomVerticalFlip(),              # mirror top-bottom, 50% of the time
        # Rotate by a random angle. Corners that appear are filled with black
        # (fill=0), which matches the black background around real cells.
        transforms.RandomRotation(degrees=config.MAX_ROTATION_DEGREES, fill=0),
        transforms.ColorJitter(
            brightness=config.COLOUR_JITTER_BRIGHTNESS,
            contrast=config.COLOUR_JITTER_CONTRAST,
            saturation=config.COLOUR_JITTER_SATURATION,
            hue=config.COLOUR_JITTER_HUE,
        ),
        transforms.ToTensor(),  # PIL image -> tensor of shape (3, height, width), values 0..1
        transforms.Normalize(config.NORMALISE_MEAN, config.NORMALISE_STD),  # centre values around 0
    ])


def build_eval_transform(image_size: int) -> transforms.Compose:
    """
    Steps applied to VALIDATION and TEST images: no randomness at all.

    We want to measure the model on the real, unmodified cells, and get the
    same score every time we evaluate.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),  # (3, height, width), values 0..1
        transforms.Normalize(config.NORMALISE_MEAN, config.NORMALISE_STD),
    ])


# ===========================================================================
# 5. Dataset: loads one image at a time
# ===========================================================================

class MalariaDataset(Dataset):
    """
    A PyTorch Dataset: something you can ask for "image number i" and get back
    (image tensor, label). It only loads an image from disk when it's asked for,
    so all 27,000 images never need to fit in memory at once.
    """

    def __init__(self, split_table: pd.DataFrame, data_dir: Path, transform: transforms.Compose):
        self.image_paths: list[Path] = [data_dir / path for path in split_table["path"]]
        self.labels: list[int] = split_table["label"].tolist()
        self.transform = transform

    def __len__(self) -> int:
        """How many images are in this dataset."""
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load image number `index`, transform it, and return it with its label."""
        # convert("RGB") guarantees 3 colour channels, even if a file was saved
        # with an extra transparency channel or in greyscale.
        with Image.open(self.image_paths[index]) as image:
            image = image.convert("RGB")

        image_tensor = self.transform(image)  # (3, image_size, image_size)

        # The label is a float (0.0 or 1.0), because the loss function we use
        # in training (BCEWithLogitsLoss) compares it with a float prediction.
        label = torch.tensor(self.labels[index], dtype=torch.float32)  # shape () = a single number
        return image_tensor, label


# ===========================================================================
# 6. DataLoaders: group images into batches
# ===========================================================================

def get_dataloaders(
    split_table: pd.DataFrame,
    data_dir: Path,
    image_size: int,
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = config.NUM_DATA_WORKERS,
) -> dict[str, DataLoader]:
    """
    Build one DataLoader per split: {"train": ..., "val": ..., "test": ...}.

    A DataLoader hands the model a "batch" of images at a time, e.g. a tensor
    of shape (64, 3, 128, 128) = 64 images, 3 colour channels, 128 x 128 pixels.
    Training images are shuffled every epoch (so the model can't learn the order);
    validation and test images are not (order doesn't matter when only measuring).
    """
    # A seeded random generator, so the shuffle order is repeatable.
    shuffle_generator = torch.Generator()
    shuffle_generator.manual_seed(config.RANDOM_SEED)

    dataloaders = {}
    for split_name in SPLIT_NAMES:
        is_training = split_name == "train"
        if is_training:
            transform = build_train_transform(image_size)
        else:
            transform = build_eval_transform(image_size)

        rows = split_table[split_table["split"] == split_name]
        dataset = MalariaDataset(rows, data_dir, transform)

        dataloaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_training,
            generator=shuffle_generator if is_training else None,
            num_workers=num_workers,
            # pin_memory speeds up copying batches to an NVIDIA GPU; useless without one.
            pin_memory=torch.cuda.is_available(),
        )
    return dataloaders


# ===========================================================================
# Run this file directly to check the whole data pipeline
# ===========================================================================

def main() -> None:
    """Build (or load) the split, then print the shape of one training batch."""
    parser = argparse.ArgumentParser(description="Check the data pipeline.")
    parser.add_argument("--data-dir", type=Path, default=config.FAKE_DATA_DIR,
                        help="Folder containing the cell images (default: the fake dataset)")
    parser.add_argument("--splits-csv", type=Path, default=None,
                        help="Where the split is saved (default: chosen from --data-dir)")
    parser.add_argument("--image-size", type=int, default=config.SIMPLE_CNN_IMAGE_SIZE)
    args = parser.parse_args()

    if not args.data_dir.exists():
        raise FileNotFoundError(f"{args.data_dir} doesn't exist. For fake data, run: python -m src.make_fake_data")

    set_seed(config.RANDOM_SEED)
    splits_csv = args.splits_csv or default_splits_csv_for(args.data_dir)
    split_table = load_or_create_splits(args.data_dir, splits_csv)
    dataloaders = get_dataloaders(split_table, args.data_dir, args.image_size)

    images, labels = next(iter(dataloaders["train"]))
    print(f"One training batch of images: {tuple(images.shape)}  (batch, channels, height, width)")
    print(f"Its labels:                   {tuple(labels.shape)}  e.g. {labels[:8].tolist()}")
    print(f"Pixel values after normalising: min {images.min():.2f}, max {images.max():.2f}")


if __name__ == "__main__":
    main()
