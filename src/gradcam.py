"""
gradcam.py: heatmaps showing WHERE in a cell image a model looked to make its decision.

Where this fits in the pipeline:
    data -> model -> train -> [evaluate: explainability] -> predict

Why this matters: a model that just says "infected" is a black box. A doctor
can't trust it without knowing what it based that on. If the heatmap lights up
on the purple parasite, good. If it lights up on the cell's edge or the black
background, the model learned a shortcut, and its scores can't be trusted.

How Grad-CAM works, in plain words:
    1. The image goes through the model, and we keep the feature maps of its
       last convolutional layer (SimpleCNN: 256 maps of 16x16). Each map shows
       where one learned pattern appears in the cell.
    2. We ask: if a map got a little stronger, how much would the model's answer
       change? (That's the map's gradient.) Maps that push the answer up matter.
    3. Add the maps together, each weighted by how much it matters, and keep
       only the positive part: a coarse heatmap (16x16), stretched to the image's
       size and laid over the cell. Red = "this pushed me towards my answer".

Can we trust a heatmap? Not automatically. A heatmap can light up the same
spot of the picture (say, its bottom-right corner) for almost every cell, rotated
or not: a quirk of the model's machinery, not evidence about the cell. So every
kind of heatmap has to pass a ROTATION CHECK: rotate the cell 180°, and the heat
must move with it. Heatmaps that fail are hidden.

What happens, in the order of this file:
    1. target_layer_for()      which layer's feature maps to use
    2. choose_examples()       pick cells: caught, cleared, missed, false alarms
    3. compute_heatmaps()      Grad-CAM itself (using the grad-cam library)
    3b. rotation_check()       does each kind of heatmap turn with the cell?
    4. plot_model_gradcam()    one grid of examples, each shown as cell | heatmap
    5. run_gradcam()           all of the above, for every trained model found

How to run it:
    python -m src.gradcam --smoke-test               # fake data, smoke-test models
    python -m src.gradcam --data-dir data/real       # real data (laptop, after downloading it)
"""

import argparse
import json
from pathlib import Path

import matplotlib

# Draw plots straight into image files (no screen needed).
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must come after matplotlib.use)

import numpy as np
import pandas as pd
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
from torch import nn

from src import config
from src.data import (
    MalariaDataset,
    build_eval_transform,
    default_splits_csv_for,
    get_dataloaders,
    load_or_create_splits,
)
from src.evaluate import load_checkpoint, predict_infected, predict_probabilities
from src.models import MODEL_NAMES, SimpleCNN
from src.utils import get_device, set_seed

# How each model's name is written in figure titles.
DISPLAY_NAMES: dict[str, str] = {"simple_cnn": "SimpleCNN"}

# The four groups of cells in each model's grid, in display order.
GROUP_TITLES: dict[str, str] = {
    "caught": "Correctly caught infections",
    "cleared": "Correctly cleared healthy cells",
    "missed": "Missed infections (labelled infected, called healthy)",
    "false_alarm": "False alarms (labelled healthy, called infected)",
}

# Every cell is drawn at this size (see config.py).
DISPLAY_SIZE = config.GRADCAM_DISPLAY_SIZE


# ===========================================================================
# 1. Which layer to explain
# ===========================================================================

def target_layer_for(model: nn.Module) -> nn.Module:
    """
    The layer whose feature maps Grad-CAM uses: the last convolutional one,
    because it holds the most "meaningful" patterns while still knowing WHERE they are.
    """
    if isinstance(model, SimpleCNN):
        # Block 4's ReLU output: 256 maps of 16x16 (for 128x128 input), taken just
        # BEFORE the final pooling halves them to 8x8. A finer grid = a sharper heatmap.
        return model.block4.relu
    raise ValueError(f"No Grad-CAM layer defined for {type(model).__name__}")


# ===========================================================================
# 2. Which cells to show
# ===========================================================================

def choose_examples(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float, per_group: int, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """
    Pick up to `per_group` test cells for each group, as indices into the test set.
      caught / cleared:      picked at random, so they're typical examples
      missed / false alarms: the most confidently wrong first (like the misclassified grid)
    """
    predictions = predict_infected(probabilities, threshold)
    is_infected = labels == config.LABEL_PARASITIZED
    is_called_infected = predictions == config.LABEL_PARASITIZED

    caught = np.where(is_infected & is_called_infected)[0]
    cleared = np.where(~is_infected & ~is_called_infected)[0]
    missed = np.where(is_infected & ~is_called_infected)[0]
    false_alarms = np.where(~is_infected & is_called_infected)[0]

    missed = missed[np.argsort(probabilities[missed])]                      # lowest probability first
    false_alarms = false_alarms[np.argsort(-probabilities[false_alarms])]   # highest probability first

    return {
        "caught": rng.permutation(caught)[:per_group],
        "cleared": rng.permutation(cleared)[:per_group],
        "missed": missed[:per_group],
        "false_alarm": false_alarms[:per_group],
    }


# ===========================================================================
# 3. Grad-CAM
# ===========================================================================

def load_display_image(path: Path) -> np.ndarray:
    """The cell as plain RGB pixels between 0 and 1 (no normalisation), for drawing."""
    with Image.open(path) as image:
        resized = image.convert("RGB").resize((DISPLAY_SIZE, DISPLAY_SIZE))
    return np.asarray(resized, dtype=np.float32) / 255.0  # (height, width, 3)


def load_model_inputs(test_rows: pd.DataFrame, indices: np.ndarray, data_dir: Path, image_size: int) -> torch.Tensor:
    """The chosen cells exactly as the model sees them: resized and normalised."""
    chosen_rows = test_rows.iloc[indices]
    dataset = MalariaDataset(chosen_rows, data_dir, build_eval_transform(image_size))
    image_list = []
    for position in range(len(dataset)):
        image, _ = dataset[position]
        image_list.append(image)
    return torch.stack(image_list)  # (num_cells, 3, image_size, image_size)


def compute_heatmaps(model: nn.Module, images: torch.Tensor, explain_labels: list[int]) -> np.ndarray:
    """
    Grad-CAM heatmaps, one per image, each scaled to 0..1.

    explain_labels says which answer to explain for each image: 1 = "why infected?",
    0 = "why healthy?". We always explain the model's OWN answer, because for a
    healthy prediction the evidence for "infected" is close to zero everywhere,
    and stretching that to 0..1 would draw misleading hot spots out of noise.
    """
    model.eval()
    device = next(model.parameters()).device
    targets = [BinaryClassifierOutputTarget(label) for label in explain_labels]
    # "with" makes sure Grad-CAM removes its hooks from the model afterwards.
    with GradCAM(model=model, target_layers=[target_layer_for(model)]) as cam:
        heatmaps = cam(input_tensor=images.to(device), targets=targets)  # (num_cells, image_size, image_size)
    return heatmaps


def resize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """Stretch a heatmap to DISPLAY_SIZE, so it lines up with the displayed cell."""
    heatmap_image = Image.fromarray(np.uint8(heatmap * 255))
    resized = heatmap_image.resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32) / 255.0


def overlay(display_image: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    """The heatmap laid over the cell: half cell, half heatmap colours (blue = low, red = high)."""
    return show_cam_on_image(display_image, resize_heatmap(heatmap), use_rgb=True)


# ===========================================================================
# 3b. The rotation check: can we trust a heatmap?
# ===========================================================================

def content_agreement(heatmaps: np.ndarray, heatmaps_of_rotated: np.ndarray) -> float:
    """
    How well heatmaps "turn with the cell" (1.0 = perfectly, 0 = not at all).

    heatmaps:            Grad-CAM for the original cells          (num_cells, height, width)
    heatmaps_of_rotated: Grad-CAM for the same cells rotated 180° (num_cells, height, width)

    We rotate the second set back and measure how similar each pair is (a
    correlation). If the heatmap shows something IN the cell, rotating the cell
    moves the heat with it, so after rotating back the two match. If the heat
    sits at a fixed spot of the picture whatever the cell looks like (a quirk of
    the model's machinery, not evidence), they don't.
    """
    rotated_back = np.rot90(heatmaps_of_rotated, k=2, axes=(1, 2))
    correlations = []
    for original, returned in zip(heatmaps, rotated_back):
        if original.std() == 0 or returned.std() == 0:
            continue  # a completely flat heatmap has no "where", so it can't be compared
        correlations.append(np.corrcoef(original.ravel(), returned.ravel())[0, 1])
    if not correlations:
        return 0.0
    return float(np.mean(correlations))


def rotation_check(
    model: nn.Module,
    test_rows: pd.DataFrame,
    data_dir: Path,
    image_size: int,
    probabilities: np.ndarray,
    threshold: float,
    rng: np.random.Generator,
) -> dict[str, float]:
    """
    Run the rotation check separately for cells the model called infected and
    cells it called healthy, because the two kinds of heatmap can behave differently.
    Returns {"infected": agreement, "healthy": agreement}.
    """
    called_infected = predict_infected(probabilities, threshold)
    agreements = {}
    for answer_name, answer in (("infected", 1), ("healthy", 0)):
        candidates = np.where(called_infected == answer)[0]
        chosen = rng.permutation(candidates)[:config.GRADCAM_CHECK_CELLS]
        if len(chosen) == 0:
            agreements[answer_name] = 0.0  # the model never gave this answer: nothing to check or show
            continue
        heatmaps = []
        heatmaps_of_rotated = []
        # Small chunks, so memory use stays low.
        for start in range(0, len(chosen), config.BATCH_SIZE):
            chunk = chosen[start:start + config.BATCH_SIZE]
            images = load_model_inputs(test_rows, chunk, data_dir, image_size)  # (chunk, 3, size, size)
            rotated = torch.rot90(images, k=2, dims=(2, 3))                        # each picture turned 180°
            explain_labels = [answer] * len(chunk)
            heatmaps.append(compute_heatmaps(model, images, explain_labels))
            heatmaps_of_rotated.append(compute_heatmaps(model, rotated, explain_labels))
        agreements[answer_name] = content_agreement(np.concatenate(heatmaps), np.concatenate(heatmaps_of_rotated))
    return agreements


def passes_check(agreement: float) -> bool:
    """True if heatmaps of this kind can be trusted (they turn with the cell)."""
    return agreement >= config.GRADCAM_MIN_AGREEMENT


def answer_for_group(group: str) -> str:
    """Which answer the model gave for a group of cells: caught and false alarms were called infected."""
    if group in ("caught", "false_alarm"):
        return "infected"
    return "healthy"


def show_hidden_heatmap(axis: plt.Axes) -> None:
    """Draw a short note instead of a heatmap that failed the rotation check."""
    axis.text(0.5, 0.5, "heatmap hidden:\nfailed the\nrotation check", ha="center", va="center",
              fontsize=9, transform=axis.transAxes)


# ===========================================================================
# 4. Plots
# ===========================================================================

def plot_model_gradcam(
    model_name: str,
    model: nn.Module,
    image_size: int,
    examples: dict[str, np.ndarray],
    test_rows: pd.DataFrame,
    data_dir: Path,
    probabilities: np.ndarray,
    threshold: float,
    agreements: dict[str, float],
    path: Path,
) -> None:
    """
    One grid for the model: a row per group, each cell shown as original | heatmap.
    Each row shows the rotation check for its kind of heatmap; failing heatmaps are hidden.
    """
    per_group = config.GRADCAM_EXAMPLES_PER_GROUP
    figure = plt.figure(figsize=(2.2 * 2 * per_group, 2.9 * len(GROUP_TITLES)), layout="constrained")
    figure.suptitle(f"{DISPLAY_NAMES[model_name]}: where the model looked (threshold {threshold:.3f}). "
                    "Red = pushed the model towards its answer. p = predicted probability of infection.\n"
                    f"Rotation check: does the heatmap turn with the cell? (1.0 = perfectly; "
                    f"below {config.GRADCAM_MIN_AGREEMENT} = not trusted, so hidden)",
                    fontsize=13)
    row_figures = figure.subfigures(len(GROUP_TITLES), 1)

    for row_figure, (group, group_title) in zip(row_figures, GROUP_TITLES.items()):
        indices = examples[group]
        answer = answer_for_group(group)
        agreement = agreements[answer]
        verdict = "trusted" if passes_check(agreement) else "NOT trusted"
        row_figure.suptitle(f"{group_title} ({len(indices)} shown). "
                            f"Rotation check for '{answer}' answers: {agreement:.2f}, {verdict}",
                            fontweight="bold")
        axes = row_figure.subplots(1, 2 * per_group)
        for axis in axes:
            axis.axis("off")
        if len(indices) == 0:
            axes[0].text(0, 0.5, "none", va="center", fontsize=12)
            continue

        images = load_model_inputs(test_rows, indices, data_dir, image_size)
        explain_labels = predict_infected(probabilities[indices], threshold).tolist()
        heatmaps = compute_heatmaps(model, images, explain_labels)

        for position, index in enumerate(indices):
            display_image = load_display_image(data_dir / test_rows["path"].iloc[index])
            original_axis = axes[2 * position]
            heatmap_axis = axes[2 * position + 1]
            original_axis.imshow(display_image)
            original_axis.set_title(f"p = {probabilities[index]:.3f}", fontsize=10)
            if passes_check(agreement):
                heatmap_axis.imshow(overlay(display_image, heatmaps[position]))
                heatmap_axis.set_title("heatmap", fontsize=10)
            else:
                show_hidden_heatmap(heatmap_axis)

    figure.savefig(path, dpi=100)
    plt.close(figure)


# ===========================================================================
# 5. Put it all together
# ===========================================================================

def tuned_threshold_for(model_name: str, results_dir: Path) -> float:
    """The threshold chosen on VALIDATION data in Phase 5 (never re-tuned here)."""
    metrics_path = results_dir / f"{model_name}_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"{metrics_path} not found: evaluate {model_name} first (python -m src.evaluate).")
    return json.loads(metrics_path.read_text())["tuned_threshold"]


def run_gradcam(
    split_table: pd.DataFrame,
    data_dir: Path,
    checkpoint_dir: Path,
    results_dir: Path,
    output_dir: Path,
    device: torch.device,
) -> list[Path]:
    """Make the Grad-CAM figures for every model that has a trained checkpoint. Returns the saved files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    test_rows = split_table[split_table["split"] == "test"].reset_index(drop=True)  # same order as the loader
    rng = np.random.default_rng(config.RANDOM_SEED)
    saved_paths = []
    checks = {}  # each model's rotation-check scores

    for model_name in MODEL_NAMES:
        checkpoint_path = checkpoint_dir / f"{model_name}_best.pt"
        if not checkpoint_path.exists():
            print(f"Skipping {model_name}: no trained model at {checkpoint_path}")
            continue

        model, checkpoint = load_checkpoint(checkpoint_path, device)
        image_size = checkpoint["image_size"]
        threshold = tuned_threshold_for(model_name, results_dir)
        print(f"{DISPLAY_NAMES[model_name]}: predicting {len(test_rows):,} test cells (threshold {threshold:.4f})")
        test_loader = get_dataloaders(split_table, data_dir, image_size)["test"]
        probabilities, labels = predict_probabilities(model, test_loader, device)

        agreements = rotation_check(model, test_rows, data_dir, image_size, probabilities, threshold, rng)
        for answer, agreement in agreements.items():
            verdict = "trusted" if passes_check(agreement) else "NOT trusted, hidden in the figures"
            print(f"  rotation check, '{answer}' heatmaps: {agreement:.2f} ({verdict})")

        examples = choose_examples(labels, probabilities, threshold, config.GRADCAM_EXAMPLES_PER_GROUP, rng)
        figure_path = output_dir / f"{model_name}_gradcam.png"
        plot_model_gradcam(model_name, model, image_size, examples, test_rows, data_dir,
                           probabilities, threshold, agreements, figure_path)
        print(f"  saved {figure_path}")
        saved_paths.append(figure_path)

        checks[model_name] = {
            "agreement_for_infected_answers": agreements["infected"],
            "agreement_for_healthy_answers": agreements["healthy"],
            "minimum_to_trust": config.GRADCAM_MIN_AGREEMENT,
            "cells_checked_per_answer": config.GRADCAM_CHECK_CELLS,
        }

    # Save the rotation-check scores, so the figures' trust labels can be traced back.
    checks_path = output_dir / "gradcam_checks.json"
    checks_path.write_text(json.dumps(checks, indent=2))
    saved_paths.append(checks_path)
    return saved_paths


# ===========================================================================
# Command line
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Draw Grad-CAM heatmaps for the trained models.")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Folder with the cell images. Not needed with --smoke-test.")
    parser.add_argument("--checkpoint-dir", type=Path, default=None,
                        help="Folder with <model>_best.pt files (default: outputs/)")
    parser.add_argument("--results-dir", type=Path, default=None,
                        help="Folder with <model>_metrics.json files, for the thresholds (default: results/)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Where to save the figures (default: outputs/)")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Use the fake data and the smoke-test models.")
    args = parser.parse_args()

    if args.smoke_test:
        data_dir = args.data_dir or config.FAKE_DATA_DIR
        smoke_dir = config.OUTPUT_DIR / "smoke_test"
        checkpoint_dir = args.checkpoint_dir or smoke_dir
        results_dir = args.results_dir or smoke_dir
        output_dir = args.output_dir or smoke_dir
    else:
        if args.data_dir is None:
            parser.error("--data-dir is required (or use --smoke-test).")
        data_dir = args.data_dir
        checkpoint_dir = args.checkpoint_dir or config.OUTPUT_DIR
        results_dir = args.results_dir or config.RESULTS_DIR
        output_dir = args.output_dir or config.OUTPUT_DIR

    # Must reuse the split the models were trained and evaluated with.
    splits_csv = default_splits_csv_for(data_dir)
    if not splits_csv.exists():
        raise FileNotFoundError(f"No saved split at {splits_csv}.")

    set_seed(config.RANDOM_SEED)
    device = get_device()
    print(f"Device: {device}")
    split_table = load_or_create_splits(data_dir, splits_csv)
    run_gradcam(split_table, data_dir, checkpoint_dir, results_dir, output_dir, device)


if __name__ == "__main__":
    main()
