"""
test_gradcam.py: automatic checks for the Grad-CAM heatmaps (src/gradcam.py).

Where this fits in the pipeline:
    data -> model -> train -> [evaluate: explainability] -> predict
Grad-CAM can quietly go wrong (wrong layer, explaining the wrong answer), so
these tests check the pieces, and then the whole thing on fake data.

Run them with:
    pytest
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from src import config, data, evaluate, gradcam, train
from src.make_fake_data import create_fake_dataset
from src.models import SimpleCNN
from src.utils import set_seed

CPU = torch.device("cpu")


def test_target_layer_is_the_last_convolutional_block() -> None:
    simple_cnn = SimpleCNN()
    assert gradcam.target_layer_for(simple_cnn) is simple_cnn.block4.relu


def test_heatmaps_have_one_value_per_pixel_between_0_and_1() -> None:
    set_seed(config.RANDOM_SEED)
    model = SimpleCNN()
    size = config.SIMPLE_CNN_IMAGE_SIZE
    images = torch.randn(2, config.NUM_CHANNELS, size, size)  # (2, 3, 128, 128)
    heatmaps = gradcam.compute_heatmaps(model, images, explain_labels=[1, 0])
    assert heatmaps.shape == (2, size, size)
    assert heatmaps.min() >= 0.0 and heatmaps.max() <= 1.0


def test_examples_are_sorted_into_the_right_groups() -> None:
    # cell:          0     1     2     3     4     5
    labels = np.array([1, 1, 1, 0, 0, 0])
    probabilities = np.array([0.9, 0.1, 0.05, 0.2, 0.8, 0.95])
    # At threshold 0.5: cell 0 caught; 1, 2 missed; 3 cleared; 4, 5 false alarms.
    examples = gradcam.choose_examples(labels, probabilities, 0.5, per_group=4, rng=np.random.default_rng(0))
    assert examples["caught"].tolist() == [0]
    assert examples["cleared"].tolist() == [3]
    assert examples["missed"].tolist() == [2, 1]        # most confidently wrong (lowest p) first
    assert examples["false_alarm"].tolist() == [5, 4]   # most confidently wrong (highest p) first


def test_gradcam_figures_are_made(tmp_path: Path) -> None:
    """Train and evaluate SimpleCNN for one epoch on fake data, then draw every figure."""
    set_seed(config.RANDOM_SEED)
    data_dir = tmp_path / "fake_data"
    output_dir = tmp_path / "outputs"
    create_fake_dataset(data_dir)
    split_table = data.split_by_patient(data.find_images(data_dir))

    train.train_model("simple_cnn", split_table, data_dir, output_dir, num_epochs=1, device=CPU)
    evaluate.evaluate_model(output_dir / "simple_cnn_best.pt", split_table, data_dir, output_dir, CPU)

    saved = gradcam.run_gradcam(split_table, data_dir, output_dir, output_dir, output_dir, CPU)

    names = sorted(path.name for path in saved)
    assert names == ["gradcam_checks.json", "simple_cnn_gradcam.png"]
    for path in saved:
        assert path.stat().st_size > 0


# ---------------------------------------------------------------------------
# The rotation check
# ---------------------------------------------------------------------------

def blob_at(row: int, column: int, size: int = 8) -> np.ndarray:
    """A made-up heatmap: all zero except one hot square at (row, column)."""
    heatmap = np.zeros((size, size))
    heatmap[row:row + 2, column:column + 2] = 1.0
    return heatmap


def test_heatmaps_that_turn_with_the_cell_pass_the_rotation_check() -> None:
    # Original: heat at the top-left. Rotated cell: heat at the bottom-right,
    # exactly where the top-left content ends up after a 180° turn.
    original = np.stack([blob_at(1, 1)])
    of_rotated = np.stack([np.rot90(blob_at(1, 1), k=2)])
    agreement = gradcam.content_agreement(original, of_rotated)
    assert agreement == pytest.approx(1.0)
    assert gradcam.passes_check(agreement)


def test_heatmaps_stuck_in_one_corner_fail_the_rotation_check() -> None:
    # Heat at the bottom-right corner of the picture, whether the cell is
    # rotated or not: it says nothing about the cell.
    original = np.stack([blob_at(5, 5)])
    of_rotated = np.stack([blob_at(5, 5)])
    agreement = gradcam.content_agreement(original, of_rotated)
    assert not gradcam.passes_check(agreement)


def test_flat_heatmaps_are_skipped_by_the_rotation_check() -> None:
    flat = np.zeros((1, 8, 8))
    assert gradcam.content_agreement(flat, flat) == 0.0
