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
       last convolutional layer (e.g. ResNet18: 512 maps of 7x7). Each map shows
       where one learned pattern appears in the cell.
    2. We ask: if a map got a little stronger, how much would the model's answer
       change? (That's the map's gradient.) Maps that push the answer up matter.
    3. Add the maps together, each weighted by how much it matters, and keep
       only the positive part: a coarse heatmap (7x7), stretched to the image's
       size and laid over the cell. Red = "this pushed me towards my answer".

What happens, in the order of this file:
    1. target_layer_for()      which layer's feature maps to use, per model
    2. choose_examples()       pick cells: caught, cleared, missed, false alarms
    3. compute_heatmaps()      Grad-CAM itself (using the grad-cam library)
    4. plots                   one grid per model, plus the cells BOTH models got wrong
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
from src.compare import DISPLAY_NAMES
from src.data import (
    MalariaDataset,
    build_eval_transform,
    default_splits_csv_for,
    get_dataloaders,
    load_or_create_splits,
)
from src.evaluate import load_checkpoint, predict_infected, predict_probabilities
from src.models import MODEL_NAMES, ResNet18Classifier, SimpleCNN
from src.utils import get_device, set_seed

# The four groups of cells in each model's grid, in display order.
GROUP_TITLES: dict[str, str] = {
    "caught": "Correctly caught infections",
    "cleared": "Correctly cleared healthy cells",
    "missed": "Missed infections (labelled infected, called healthy)",
    "false_alarm": "False alarms (labelled healthy, called infected)",
}

# Every cell is drawn at this size, whichever model looked at it.
DISPLAY_SIZE = config.RESNET_IMAGE_SIZE


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
    if isinstance(model, ResNet18Classifier):
        # layer4: ResNet18's last convolutional stage, 512 maps of 7x7 (for 224x224 input).
        return model.backbone.layer4
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
    path: Path,
) -> None:
    """One grid per model: a row per group, each cell shown as original | heatmap."""
    per_group = config.GRADCAM_EXAMPLES_PER_GROUP
    figure = plt.figure(figsize=(2.2 * 2 * per_group, 2.9 * len(GROUP_TITLES)), layout="constrained")
    figure.suptitle(f"{DISPLAY_NAMES[model_name]}: where the model looked (threshold {threshold:.3f}). "
                    "Red = pushed the model towards its answer. p = predicted probability of infection.",
                    fontsize=13)
    row_figures = figure.subfigures(len(GROUP_TITLES), 1)

    for row_figure, (group, group_title) in zip(row_figures, GROUP_TITLES.items()):
        indices = examples[group]
        row_figure.suptitle(f"{group_title} ({len(indices)} shown)", fontweight="bold")
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
            heatmap_axis.imshow(overlay(display_image, heatmaps[position]))
            heatmap_axis.set_title("heatmap", fontsize=10)

    figure.savefig(path, dpi=100)
    plt.close(figure)


def plot_shared_mistakes(
    results_by_model: dict[str, dict], test_rows: pd.DataFrame, data_dir: Path, path: Path
) -> int:
    """
    Cells that EVERY model got wrong: one row per cell, showing
    original | each model's heatmap. Returns how many shared mistakes there were.
    """
    labels = None
    wrong_for_all = None
    for results in results_by_model.values():
        predictions = predict_infected(results["probabilities"], results["threshold"])
        is_wrong = predictions != results["labels"]
        if wrong_for_all is None:
            wrong_for_all = is_wrong
            labels = results["labels"]
        else:
            wrong_for_all = wrong_for_all & is_wrong
    shared_indices = np.where(wrong_for_all)[0]

    # Most confidently wrong first, judged by the last model (ResNet18).
    last_results = list(results_by_model.values())[-1]
    wrongness = np.abs(last_results["probabilities"][shared_indices] - labels[shared_indices])
    shown_indices = shared_indices[np.argsort(-wrongness)][:config.GRADCAM_SHARED_MISTAKES]

    model_names = list(results_by_model)
    num_columns = 1 + len(model_names)
    num_rows = max(len(shown_indices), 1)
    figure, axes = plt.subplots(num_rows, num_columns, figsize=(3 * num_columns, 3 * num_rows + 0.8),
                                squeeze=False, layout="constrained")
    figure.suptitle(f"Cells every model got wrong: {len(shared_indices)} in total, "
                    f"{len(shown_indices)} shown\n(red = pushed the model towards its answer)", fontsize=12)
    for axis in axes.flat:
        axis.axis("off")
    if len(shown_indices) == 0:
        axes[0, 0].text(0, 0.5, "none", va="center", fontsize=12)
        figure.savefig(path, dpi=100)
        plt.close(figure)
        return 0

    # One model at a time: compute its heatmaps for all shown cells at once.
    heatmaps_by_model = {}
    for model_name, results in results_by_model.items():
        images = load_model_inputs(test_rows, shown_indices, data_dir, results["image_size"])
        explain_labels = predict_infected(results["probabilities"][shown_indices], results["threshold"]).tolist()
        heatmaps_by_model[model_name] = compute_heatmaps(results["model"], images, explain_labels)

    for row, index in enumerate(shown_indices):
        display_image = load_display_image(data_dir / test_rows["path"].iloc[index])
        label_name = "infected" if labels[index] == config.LABEL_PARASITIZED else "healthy"
        axes[row, 0].imshow(display_image)
        axes[row, 0].set_title(f"labelled {label_name}", fontsize=10)
        for column, model_name in enumerate(model_names, start=1):
            probability = results_by_model[model_name]["probabilities"][index]
            axes[row, column].imshow(overlay(display_image, heatmaps_by_model[model_name][row]))
            axes[row, column].set_title(f"{DISPLAY_NAMES[model_name]}: p = {probability:.3f}", fontsize=10)

    figure.savefig(path, dpi=100)
    plt.close(figure)
    return len(shared_indices)


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
    results_by_model = {}

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

        examples = choose_examples(labels, probabilities, threshold, config.GRADCAM_EXAMPLES_PER_GROUP, rng)
        figure_path = output_dir / f"{model_name}_gradcam.png"
        plot_model_gradcam(model_name, model, image_size, examples, test_rows, data_dir,
                           probabilities, threshold, figure_path)
        print(f"  saved {figure_path}")
        saved_paths.append(figure_path)

        results_by_model[model_name] = {
            "model": model, "image_size": image_size, "threshold": threshold,
            "probabilities": probabilities, "labels": labels,
        }

    if len(results_by_model) >= 2:
        shared_path = output_dir / "shared_mistakes_gradcam.png"
        num_shared = plot_shared_mistakes(results_by_model, test_rows, data_dir, shared_path)
        print(f"Cells every model got wrong: {num_shared}. Saved {shared_path}")
        saved_paths.append(shared_path)
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
