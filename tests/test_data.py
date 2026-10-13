"""
test_data.py: automatic checks for the data pipeline (src/data.py), run on the fake dataset.

Where this fits in the pipeline:
    [data] -> model -> train -> evaluate -> predict
These tests guard the data step. If one fails, the model would be trained or
tested on wrong data, which is far harder to notice later than a crash.

Run them with:
    pytest
"""

import shutil
from pathlib import Path

import pandas as pd
import pytest
import torch

from src import config, data
from src.make_fake_data import create_fake_dataset

# The fake dataset has this many unique images; the nested duplicate folder
# holds a second copy of each, which find_images() must remove.
NUM_UNIQUE_FAKE_IMAGES = 2 * config.FAKE_IMAGES_PER_CLASS


@pytest.fixture(scope="module")
def fake_data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """
    A fresh fake dataset in a temporary folder, shared by all tests in this file.
    (A pytest "fixture" is setup code that tests receive as an argument.)
    Using a temporary folder means the tests never touch data/fake.
    """
    folder = tmp_path_factory.mktemp("fake_data")
    create_fake_dataset(folder)
    return folder


@pytest.fixture(scope="module")
def split_table(fake_data_dir: Path) -> pd.DataFrame:
    """The fake dataset, found and split by patient."""
    return data.split_by_patient(data.find_images(fake_data_dir))


# ---------------------------------------------------------------------------
# Finding images
# ---------------------------------------------------------------------------

def test_patient_id_is_read_from_a_real_style_filename() -> None:
    filename = "C33P1thinF_IMG_20150619_114756a_cell_179.png"
    assert data.extract_patient_id(filename) == "C33P1"


def test_filename_without_patient_code_raises_an_error() -> None:
    with pytest.raises(ValueError):
        data.extract_patient_id("random_cell_picture.png")


def test_find_images_skips_junk_and_removes_duplicates(fake_data_dir: Path) -> None:
    table = data.find_images(fake_data_dir)
    # 160 .png files on disk (80 + 80 nested copies) and 4 Thumbs.db files:
    # only the 80 unique images should remain.
    assert len(table) == NUM_UNIQUE_FAKE_IMAGES
    assert table["path"].str.endswith(".png").all()


def test_image_under_both_labels_is_dropped(tmp_path: Path) -> None:
    create_fake_dataset(tmp_path)
    # Copy one infected image into the Uninfected folder: now the same picture
    # carries both labels, so neither copy can be trusted.
    infected_image = next((tmp_path / "cell_images" / "Parasitized").glob("*.png"))
    shutil.copy(infected_image, tmp_path / "cell_images" / "Uninfected" / infected_image.name)

    table = data.find_images(tmp_path)
    assert len(table) == NUM_UNIQUE_FAKE_IMAGES - 1
    assert not table["path"].str.endswith("/" + infected_image.name).any()


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def test_no_patient_appears_in_two_splits(split_table: pd.DataFrame) -> None:
    splits_per_patient = split_table.groupby("patient_id")["split"].nunique()
    assert (splits_per_patient == 1).all()


def test_every_split_contains_both_classes(split_table: pd.DataFrame) -> None:
    for split_name in data.SPLIT_NAMES:
        labels = set(split_table.loc[split_table["split"] == split_name, "label"])
        assert labels == {config.LABEL_UNINFECTED, config.LABEL_PARASITIZED}, split_name


def test_every_image_is_in_exactly_one_split(split_table: pd.DataFrame) -> None:
    assert len(split_table) == NUM_UNIQUE_FAKE_IMAGES
    assert split_table["path"].is_unique


def test_split_is_the_same_every_time(fake_data_dir: Path, split_table: pd.DataFrame) -> None:
    split_again = data.split_by_patient(data.find_images(fake_data_dir))
    pd.testing.assert_frame_equal(split_table, split_again)


def test_saved_split_is_reloaded_unchanged(fake_data_dir: Path, tmp_path: Path) -> None:
    splits_csv = tmp_path / "splits.csv"
    created = data.load_or_create_splits(fake_data_dir, splits_csv)  # creates the file
    assert splits_csv.exists()
    reloaded = data.load_or_create_splits(fake_data_dir, splits_csv)  # reads the file
    pd.testing.assert_frame_equal(created, reloaded)


def test_leaking_split_is_rejected(split_table: pd.DataFrame) -> None:
    leaky_table = split_table.copy()
    # Move ONE training image of some patient into the test set: that patient
    # is now in two splits, which check_split_is_safe() must refuse.
    first_train_row = leaky_table.index[leaky_table["split"] == "train"][0]
    leaky_table.loc[first_train_row, "split"] = "test"
    with pytest.raises(ValueError, match="leakage"):
        data.check_split_is_safe(leaky_table)


# ---------------------------------------------------------------------------
# Dataset and DataLoaders
# ---------------------------------------------------------------------------

def test_dataset_returns_image_tensor_and_label(fake_data_dir: Path, split_table: pd.DataFrame) -> None:
    image_size = config.SIMPLE_CNN_IMAGE_SIZE
    dataset = data.MalariaDataset(split_table, fake_data_dir, data.build_eval_transform(image_size))
    image, label = dataset[0]
    assert image.shape == (config.NUM_CHANNELS, image_size, image_size)  # (3, 128, 128)
    assert label.dtype == torch.float32
    assert label.item() in (0.0, 1.0)


def test_dataloader_batches_have_the_right_shape(fake_data_dir: Path, split_table: pd.DataFrame) -> None:
    image_size = config.SIMPLE_CNN_IMAGE_SIZE
    batch_size = 8
    dataloaders = data.get_dataloaders(split_table, fake_data_dir, image_size, batch_size=batch_size)

    images, labels = next(iter(dataloaders["train"]))
    assert images.shape == (batch_size, config.NUM_CHANNELS, image_size, image_size)  # (8, 3, 128, 128)
    assert labels.shape == (batch_size,)
    assert set(labels.tolist()) <= {0.0, 1.0}


def test_resnet_image_size_works_too(fake_data_dir: Path, split_table: pd.DataFrame) -> None:
    image_size = config.RESNET_IMAGE_SIZE
    dataloaders = data.get_dataloaders(split_table, fake_data_dir, image_size, batch_size=4)
    images, _ = next(iter(dataloaders["val"]))
    assert images.shape == (4, config.NUM_CHANNELS, image_size, image_size)  # (4, 3, 224, 224)
